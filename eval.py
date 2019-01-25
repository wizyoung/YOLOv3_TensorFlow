# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse

from utils.data_utils import parse_data
from utils.misc_utils import parse_anchors, read_class_names, shuffle_and_overwrite, update_dict, make_summary, config_learning_rate, config_optimizer, list_add
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu
from utils.nms_utils import gpu_nms

from model import yolov3

#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="YOLO-V3 eval procedure.")
# some paths
parser.add_argument("--eval_file", type=str, default="./data/my_data/val.txt",
                    help="The path of the validation or test txt file.")

parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")

parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")

parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")

# some numbers
parser.add_argument("--batch_size", type=int, default=20,
                    help="The batch size for training.")

parser.add_argument("--img_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image to `img_size`, size format: [width, height]")

parser.add_argument("--num_threads", type=int, default=10,
                    help="Number of threads for image processing used in tf.data pipeline.")

parser.add_argument("--prefetech_buffer", type=int, default=3,
                    help="Prefetech_buffer used in tf.data pipeline.")

args = parser.parse_args()

# args params
args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.class_num = len(args.classes)
args.img_cnt = len(open(args.eval_file, 'r').readlines())
args.batch_num = int(np.ceil(float(args.img_cnt) / args.batch_size))

# setting placeholders
is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')

##################
# tf.data pipeline
##################

dataset = tf.data.TextLineDataset(args.eval_file)
dataset = dataset.apply(tf.contrib.data.map_and_batch(
    lambda x: tf.py_func(parse_data, [x, args.class_num, args.img_size, args.anchors, 'val'], [tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads, batch_size=args.batch_size))
dataset = dataset.prefetch(args.prefetech_buffer)

iterator = dataset.make_one_shot_iterator()

# get an element from the dataset iterator
image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
y_true = [y_true_13, y_true_26, y_true_52]

# tf.data pipeline will lose the data shape, so we need to set it manually
image.set_shape([None, args.img_size[1], args.img_size[0], 3])
for y in y_true:
    y.set_shape([None, None, None, None, None])

##################
# Model definition
##################

# define yolo-v3 model here
yolo_model = yolov3(args.class_num, args.anchors)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image, is_training=is_training)
loss = yolo_model.compute_loss(pred_feature_maps, y_true)
y_pred = yolo_model.predict(pred_feature_maps)

################
# register the gpu nms operation here for the following evaluation scheme
pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, args.class_num)
################

saver_to_restore = tf.train.Saver()

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer()])
    saver_to_restore.restore(sess, args.restore_path)

    print('\n----------- start to eval -----------\n')

    true_positive_dict, true_labels_dict, pred_labels_dict = {}, {}, {}
    val_loss = [0., 0., 0., 0., 0.]

    for j in range(args.batch_num):
        y_pred_, y_true_, loss_ = sess.run([y_pred, y_true, loss], feed_dict={is_training: False})
        true_positive_dict_tmp, true_labels_dict_tmp, pred_labels_dict_tmp = \
            evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag,
                            y_pred_, y_true_, args.class_num, calc_now=False)
        true_positive_dict = update_dict(true_positive_dict, true_positive_dict_tmp)
        true_labels_dict = update_dict(true_labels_dict, true_labels_dict_tmp)
        pred_labels_dict = update_dict(pred_labels_dict, pred_labels_dict_tmp)

        val_loss = list_add(val_loss, loss_)

    # make sure there is at least ground truth an object in each image
    # avoid divided by 0
    recall = float(sum(true_positive_dict.values())) / (sum(true_labels_dict.values()) + 1e-6)
    precision = float(sum(true_positive_dict.values())) / (sum(pred_labels_dict.values()) + 1e-6)

    print("recall: {:.3f}, precision: {:.3f}".format(recall, precision))
    print("total_loss: {:.3f}, loss_xy: {:.3f}, loss_wh: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}".format(
        val_loss[0] / args.img_cnt, val_loss[1] / args.img_cnt, val_loss[2] / args.img_cnt, val_loss[3] / args.img_cnt, val_loss[4] / args.img_cnt))