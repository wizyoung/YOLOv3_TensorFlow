# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import logging
from tqdm import trange

import args

from utils.data_utils import get_batch_data
from utils.misc_utils import shuffle_and_overwrite, make_summary, config_learning_rate, config_optimizer, AverageMeter
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec
from utils.nms_utils import gpu_nms

from model import yolov3

# setting loggers
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename=args.progress_log_path, filemode='w')

# setting placeholders
is_training = tf.placeholder(tf.bool, name="phase_train")
handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
# register the gpu nms operation here for the following evaluation scheme
pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, args.class_num, args.nms_topk, args.score_threshold, args.nms_threshold)

##################
# tf.data pipeline
##################
train_dataset = tf.data.TextLineDataset(args.train_file)
train_dataset = train_dataset.shuffle(args.train_img_cnt)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.map(
    lambda x: tf.py_func(get_batch_data,
                         inp=[x, args.class_num, args.img_size, args.anchors, 'train', args.multi_scale_train, args.use_mix_up, args.letterbox_resize],
                         Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads
)
train_dataset = train_dataset.prefetch(args.prefetech_buffer)

val_dataset = tf.data.TextLineDataset(args.val_file)
val_dataset = val_dataset.batch(1)
val_dataset = val_dataset.map(
    lambda x: tf.py_func(get_batch_data,
                         inp=[x, args.class_num, args.img_size, args.anchors, 'val', False, False, args.letterbox_resize],
                         Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads
)
val_dataset.prefetch(args.prefetech_buffer)

iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_init_op = iterator.make_initializer(train_dataset)
val_init_op = iterator.make_initializer(val_dataset)

# get an element from the chosen dataset iterator
image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
y_true = [y_true_13, y_true_26, y_true_52]

# tf.data pipeline will lose the data `static` shape, so we need to set it manually
image_ids.set_shape([None])
image.set_shape([None, None, None, 3])
for y in y_true:
    y.set_shape([None, None, None, None, None])

##################
# Model definition
##################
yolo_model = yolov3(args.class_num, args.anchors, args.use_label_smooth, args.use_focal_loss, args.batch_norm_decay, args.weight_decay, use_static_shape=False)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image, is_training=is_training)
loss = yolo_model.compute_loss(pred_feature_maps, y_true)
y_pred = yolo_model.predict(pred_feature_maps)

l2_loss = tf.losses.get_regularization_loss()

# setting restore parts and vars to update
saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=args.restore_include, exclude=args.restore_exclude))
update_vars = tf.contrib.framework.get_variables_to_restore(include=args.update_part)

tf.summary.scalar('train_batch_statistics/total_loss', loss[0])
tf.summary.scalar('train_batch_statistics/loss_xy', loss[1])
tf.summary.scalar('train_batch_statistics/loss_wh', loss[2])
tf.summary.scalar('train_batch_statistics/loss_conf', loss[3])
tf.summary.scalar('train_batch_statistics/loss_class', loss[4])
tf.summary.scalar('train_batch_statistics/loss_l2', l2_loss)
tf.summary.scalar('train_batch_statistics/loss_ratio', l2_loss / loss[0])

global_step = tf.Variable(float(args.global_step), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
if args.use_warm_up:
    learning_rate = tf.cond(tf.less(global_step, args.train_batch_num * args.warm_up_epoch), 
                            lambda: args.learning_rate_init * global_step / (args.train_batch_num * args.warm_up_epoch),
                            lambda: config_learning_rate(args, global_step - args.train_batch_num * args.warm_up_epoch))
else:
    learning_rate = config_learning_rate(args, global_step)
tf.summary.scalar('learning_rate', learning_rate)

if not args.save_optimizer:
    saver_to_save = tf.train.Saver()
    saver_best = tf.train.Saver()

optimizer = config_optimizer(args.optimizer_name, learning_rate)

# set dependencies for BN ops
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # train_op = optimizer.minimize(loss[0] + l2_loss, var_list=update_vars, global_step=global_step)
    # apply gradient clip to avoid gradient exploding
    gvs = optimizer.compute_gradients(loss[0] + l2_loss, var_list=update_vars)
    clip_grad_var = [gv if gv[0] is None else [
          tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
    train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)

if args.save_optimizer:
    print('Saving optimizer parameters to checkpoint! Remember to restore the global_step in the fine-tuning afterwards.')
    saver_to_save = tf.train.Saver()
    saver_best = tf.train.Saver()

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver_to_restore.restore(sess, args.restore_path)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    print('\n----------- start to train -----------\n')

    best_mAP = -np.Inf

    for epoch in range(args.total_epoches):

        sess.run(train_init_op)
        loss_total, loss_xy, loss_wh, loss_conf, loss_class = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        for i in trange(args.train_batch_num):
            _, summary, __y_pred, __y_true, __loss, __global_step, __lr = sess.run(
                [train_op, merged, y_pred, y_true, loss, global_step, learning_rate],
                feed_dict={is_training: True})

            writer.add_summary(summary, global_step=__global_step)

            loss_total.update(__loss[0], len(__y_pred[0]))
            loss_xy.update(__loss[1], len(__y_pred[0]))
            loss_wh.update(__loss[2], len(__y_pred[0]))
            loss_conf.update(__loss[3], len(__y_pred[0]))
            loss_class.update(__loss[4], len(__y_pred[0]))

            if __global_step % args.train_evaluation_step == 0 and __global_step > 0:
                # recall, precision = evaluate_on_cpu(__y_pred, __y_true, args.class_num, args.nms_topk, args.score_threshold, args.nms_threshold)
                recall, precision = evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __y_pred, __y_true, args.class_num, args.nms_threshold)

                info = "Epoch: {}, global_step: {} | loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f} | ".format(
                        epoch, int(__global_step), loss_total.average, loss_xy.average, loss_wh.average, loss_conf.average, loss_class.average)
                info += 'Last batch: rec: {:.3f}, prec: {:.3f} | lr: {:.5g}'.format(recall, precision, __lr)
                print(info)
                logging.info(info)

                writer.add_summary(make_summary('evaluation/train_batch_recall', recall), global_step=__global_step)
                writer.add_summary(make_summary('evaluation/train_batch_precision', precision), global_step=__global_step)

                if np.isnan(loss_total.average):
                    print('****' * 10)
                    raise ArithmeticError(
                        'Gradient exploded! Please train again and you may need modify some parameters.')

        # NOTE: this is just demo. You can set the conditions when to save the weights.
        if epoch % args.save_epoch == 0 and epoch > 0:
            if loss_total.average <= 2.:
                saver_to_save.save(sess, args.save_dir + 'model-epoch_{}_step_{}_loss_{:.4f}_lr_{:.5g}'.format(epoch, int(__global_step), loss_total.average, __lr))

        # switch to validation dataset for evaluation
        if epoch % args.val_evaluation_epoch == 0 and epoch >= args.warm_up_epoch:
            sess.run(val_init_op)

            val_loss_total, val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class = \
                AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

            val_preds = []

            for j in trange(args.val_img_cnt):
                __image_ids, __y_pred, __loss = sess.run([image_ids, y_pred, loss],
                                                         feed_dict={is_training: False})
                pred_content = get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __image_ids, __y_pred)
                val_preds.extend(pred_content)
                val_loss_total.update(__loss[0])
                val_loss_xy.update(__loss[1])
                val_loss_wh.update(__loss[2])
                val_loss_conf.update(__loss[3])
                val_loss_class.update(__loss[4])

            # calc mAP
            rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
            gt_dict = parse_gt_rec(args.val_file, args.img_size, args.letterbox_resize)

            info = '======> Epoch: {}, global_step: {}, lr: {:.6g} <======\n'.format(epoch, __global_step, __lr)

            for ii in range(args.class_num):
                npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, ii, iou_thres=args.eval_threshold, use_07_metric=args.use_voc_07_metric)
                info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(ii, rec, prec, ap)
                rec_total.update(rec, npos)
                prec_total.update(prec, nd)
                ap_total.update(ap, 1)

            mAP = ap_total.average
            info += 'EVAL: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n'.format(rec_total.average, prec_total.average, mAP)
            info += 'EVAL: loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f}\n'.format(
                val_loss_total.average, val_loss_xy.average, val_loss_wh.average, val_loss_conf.average, val_loss_class.average)
            print(info)
            logging.info(info)

            if mAP > best_mAP:
                best_mAP = mAP
                saver_best.save(sess, args.save_dir + 'best_model_Epoch_{}_step_{}_mAP_{:.4f}_loss_{:.4f}_lr_{:.7g}'.format(
                                   epoch, int(__global_step), best_mAP, val_loss_total.average, __lr))

            writer.add_summary(make_summary('evaluation/val_mAP', mAP), global_step=epoch)
            writer.add_summary(make_summary('evaluation/val_recall', rec_total.average), global_step=epoch)
            writer.add_summary(make_summary('evaluation/val_precision', prec_total.average), global_step=epoch)
            writer.add_summary(make_summary('validation_statistics/total_loss', val_loss_total.average), global_step=epoch)
            writer.add_summary(make_summary('validation_statistics/loss_xy', val_loss_xy.average), global_step=epoch)
            writer.add_summary(make_summary('validation_statistics/loss_wh', val_loss_wh.average), global_step=epoch)
            writer.add_summary(make_summary('validation_statistics/loss_conf', val_loss_conf.average), global_step=epoch)
            writer.add_summary(make_summary('validation_statistics/loss_class', val_loss_class.average), global_step=epoch)

