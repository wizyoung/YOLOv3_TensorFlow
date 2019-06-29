# coding: utf-8

from __future__ import division, print_function

import numpy as np
import cv2
from collections import Counter

from utils.nms_utils import cpu_nms, gpu_nms
from utils.data_utils import parse_line


def calc_iou(pred_boxes, true_boxes):
    '''
    Maintain an efficient way to calculate the ios matrix using the numpy broadcast tricks.
    shape_info: pred_boxes: [N, 4]
                true_boxes: [V, 4]
    return: IoU matrix: shape: [N, V]
    '''

    # [N, 1, 4]
    pred_boxes = np.expand_dims(pred_boxes, -2)
    # [1, V, 4]
    true_boxes = np.expand_dims(true_boxes, 0)

    # [N, 1, 2] & [1, V, 2] ==> [N, V, 2]
    intersect_mins = np.maximum(pred_boxes[..., :2], true_boxes[..., :2])
    intersect_maxs = np.minimum(pred_boxes[..., 2:], true_boxes[..., 2:])
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [N, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [N, 1, 2]
    pred_box_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
    # shape: [N, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # [1, V, 2]
    true_boxes_wh = true_boxes[..., 2:] - true_boxes[..., :2]
    # [1, V]
    true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]

    # shape: [N, V]
    iou = intersect_area / (pred_box_area + true_boxes_area - intersect_area + 1e-10)

    return iou


def evaluate_on_cpu(y_pred, y_true, num_classes, calc_now=True, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    '''
    Given y_pred and y_true of a batch of data, get the recall and precision of the current batch.
    '''

    num_images = y_true[0].shape[0]
    true_labels_dict = {i: 0 for i in range(num_classes)}  # {class: count}
    pred_labels_dict = {i: 0 for i in range(num_classes)}
    true_positive_dict = {i: 0 for i in range(num_classes)}

    for i in range(num_images):
        true_labels_list, true_boxes_list = [], []
        for j in range(3):  # three feature maps
            # shape: [13, 13, 3, 80]
            true_probs_temp = y_true[j][i][..., 5:-1]
            # shape: [13, 13, 3, 4] (x_center, y_center, w, h)
            true_boxes_temp = y_true[j][i][..., 0:4]

            # [13, 13, 3]
            object_mask = true_probs_temp.sum(axis=-1) > 0

            # [V, 3] V: Ground truth number of the current image
            true_probs_temp = true_probs_temp[object_mask]
            # [V, 4]
            true_boxes_temp = true_boxes_temp[object_mask]

            # [V], labels
            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            # [V, 4] (x_center, y_center, w, h)
            true_boxes_list += true_boxes_temp.tolist()

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items():
                true_labels_dict[cls] += count

        # [V, 4] (xmin, ymin, xmax, ymax)
        true_boxes = np.array(true_boxes_list)
        box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]
        true_boxes[:, 0:2] = box_centers - box_sizes / 2.
        true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes

        # [1, xxx, 4]
        pred_boxes = y_pred[0][i:i + 1]
        pred_confs = y_pred[1][i:i + 1]
        pred_probs = y_pred[2][i:i + 1]

        # pred_boxes: [N, 4]
        # pred_confs: [N]
        # pred_labels: [N]
        # N: Detected box number of the current image
        pred_boxes, pred_confs, pred_labels = cpu_nms(pred_boxes, pred_confs * pred_probs, num_classes,
                                                      max_boxes=max_boxes, score_thresh=score_thresh, iou_thresh=iou_thresh)

        # len: N
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        if pred_labels_list == []:
            continue

        # calc iou
        # [N, V]
        iou_matrix = calc_iou(pred_boxes, true_boxes)
        # [N]
        max_iou_idx = np.argmax(iou_matrix, axis=-1)

        correct_idx = []
        correct_conf = []
        for k in range(max_iou_idx.shape[0]):
            pred_labels_dict[pred_labels_list[k]] += 1
            match_idx = max_iou_idx[k]  # V level
            if iou_matrix[k, match_idx] > iou_thresh and true_labels_list[match_idx] == pred_labels_list[k]:
                if match_idx not in correct_idx:
                    correct_idx.append(match_idx)
                    correct_conf.append(pred_confs[k])
                else:
                    same_idx = correct_idx.index(match_idx)
                    if pred_confs[k] > correct_conf[same_idx]:
                        correct_idx.pop(same_idx)
                        correct_conf.pop(same_idx)
                        correct_idx.append(match_idx)
                        correct_conf.append(pred_confs[k])

        for t in correct_idx:
            true_positive_dict[true_labels_list[t]] += 1

    if calc_now:
        # avoid divided by 0
        recall = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
        precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)

        return recall, precision
    else:
        return true_positive_dict, true_labels_dict, pred_labels_dict


def evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, y_pred, y_true, num_classes, iou_thresh=0.5, calc_now=True):
    '''
    Given y_pred and y_true of a batch of data, get the recall and precision of the current batch.
    This function will perform gpu operation on the GPU.
    '''

    num_images = y_true[0].shape[0]
    true_labels_dict = {i: 0 for i in range(num_classes)}  # {class: count}
    pred_labels_dict = {i: 0 for i in range(num_classes)}
    true_positive_dict = {i: 0 for i in range(num_classes)}

    for i in range(num_images):
        true_labels_list, true_boxes_list = [], []
        for j in range(3):  # three feature maps
            # shape: [13, 13, 3, 80]
            true_probs_temp = y_true[j][i][..., 5:-1]
            # shape: [13, 13, 3, 4] (x_center, y_center, w, h)
            true_boxes_temp = y_true[j][i][..., 0:4]

            # [13, 13, 3]
            object_mask = true_probs_temp.sum(axis=-1) > 0

            # [V, 80] V: Ground truth number of the current image
            true_probs_temp = true_probs_temp[object_mask]
            # [V, 4]
            true_boxes_temp = true_boxes_temp[object_mask]

            # [V], labels, each from 0 to 79
            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            # [V, 4] (x_center, y_center, w, h)
            true_boxes_list += true_boxes_temp.tolist()

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items():
                true_labels_dict[cls] += count

        # [V, 4] (xmin, ymin, xmax, ymax)
        true_boxes = np.array(true_boxes_list)
        box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]
        true_boxes[:, 0:2] = box_centers - box_sizes / 2.
        true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes

        # [1, xxx, 4]
        pred_boxes = y_pred[0][i:i + 1]
        pred_confs = y_pred[1][i:i + 1]
        pred_probs = y_pred[2][i:i + 1]

        # pred_boxes: [N, 4]
        # pred_confs: [N]
        # pred_labels: [N]
        # N: Detected box number of the current image
        pred_boxes, pred_confs, pred_labels = sess.run(gpu_nms_op,
                                                       feed_dict={pred_boxes_flag: pred_boxes,
                                                                  pred_scores_flag: pred_confs * pred_probs})
        # len: N
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        if pred_labels_list == []:
            continue

        # calc iou
        # [N, V]
        iou_matrix = calc_iou(pred_boxes, true_boxes)
        # [N]
        max_iou_idx = np.argmax(iou_matrix, axis=-1)

        correct_idx = []
        correct_conf = []
        for k in range(max_iou_idx.shape[0]):
            pred_labels_dict[pred_labels_list[k]] += 1
            match_idx = max_iou_idx[k]  # V level
            if iou_matrix[k, match_idx] > iou_thresh and true_labels_list[match_idx] == pred_labels_list[k]:
                if match_idx not in correct_idx:
                    correct_idx.append(match_idx)
                    correct_conf.append(pred_confs[k])
                else:
                    same_idx = correct_idx.index(match_idx)
                    if pred_confs[k] > correct_conf[same_idx]:
                        correct_idx.pop(same_idx)
                        correct_conf.pop(same_idx)
                        correct_idx.append(match_idx)
                        correct_conf.append(pred_confs[k])

        for t in correct_idx:
            true_positive_dict[true_labels_list[t]] += 1

    if calc_now:
        # avoid divided by 0
        recall = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
        precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)

        return recall, precision
    else:
        return true_positive_dict, true_labels_dict, pred_labels_dict


def get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, image_ids, y_pred):
    '''
    Given the y_pred of an input image, get the predicted bbox and label info.
    return:
        pred_content: 2d list.
    '''
    image_id = image_ids[0]

    # keep the first dimension 1
    pred_boxes = y_pred[0][0:1]
    pred_confs = y_pred[1][0:1]
    pred_probs = y_pred[2][0:1]

    boxes, scores, labels = sess.run(gpu_nms_op,
                                     feed_dict={pred_boxes_flag: pred_boxes,
                                                pred_scores_flag: pred_confs * pred_probs})

    pred_content = []
    for i in range(len(labels)):
        x_min, y_min, x_max, y_max = boxes[i]
        score = scores[i]
        label = labels[i]
        pred_content.append([image_id, x_min, y_min, x_max, y_max, score, label])

    return pred_content


gt_dict = {}  # key: img_id, value: gt object list
def parse_gt_rec(gt_filename, target_img_size, letterbox_resize=True):
    '''
    parse and re-organize the gt info.
    return:
        gt_dict: dict. Each key is a img_id, the value is the gt bboxes in the corresponding img.
    '''

    global gt_dict

    if not gt_dict:
        new_width, new_height = target_img_size
        with open(gt_filename, 'r') as f:
            for line in f:
                img_id, pic_path, boxes, labels, ori_width, ori_height = parse_line(line)

                objects = []
                for i in range(len(labels)):
                    x_min, y_min, x_max, y_max = boxes[i]
                    label = labels[i]

                    if letterbox_resize:
                        resize_ratio = min(new_width / ori_width, new_height / ori_height)

                        resize_w = int(resize_ratio * ori_width)
                        resize_h = int(resize_ratio * ori_height)

                        dw = int((new_width - resize_w) / 2)
                        dh = int((new_height - resize_h) / 2)

                        objects.append([x_min * resize_ratio + dw,
                                        y_min * resize_ratio + dh,
                                        x_max * resize_ratio + dw,
                                        y_max * resize_ratio + dh,
                                        label])
                    else:
                        objects.append([x_min * new_width / ori_width,
                                        y_min * new_height / ori_height,
                                        x_max * new_width / ori_width,
                                        y_max * new_height / ori_height,
                                        label])
                gt_dict[img_id] = objects
    return gt_dict


# The following two functions are modified from FAIR's Detectron repo to calculate mAP:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/voc_eval.py
def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(gt_dict, val_preds, classidx, iou_thres=0.5, use_07_metric=False):
    '''
    Top level function that does the PASCAL VOC evaluation.
    '''
    # 1.obtain gt: extract all gt objects for this class
    class_recs = {}
    npos = 0
    for img_id in gt_dict:
        R = [obj for obj in gt_dict[img_id] if obj[-1] == classidx]
        bbox = np.array([x[:4] for x in R])
        det = [False] * len(R)
        npos += len(R)
        class_recs[img_id] = {'bbox': bbox, 'det': det}

    # 2. obtain pred results
    pred = [x for x in val_preds if x[-1] == classidx]
    img_ids = [x[0] for x in pred]
    confidence = np.array([x[-2] for x in pred])
    BB = np.array([[x[1], x[2], x[3], x[4]] for x in pred])

    # 3. sort by confidence
    sorted_ind = np.argsort(-confidence)
    try:
        BB = BB[sorted_ind, :]
    except:
        print('no box, ignore')
        return 1e-6, 1e-6, 0, 0, 0
    img_ids = [img_ids[x] for x in sorted_ind]

    # 4. mark TPs and FPs
    nd = len(img_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        # all the gt info in some image
        R = class_recs[img_ids[d]]
        bb = BB[d, :]
        ovmax = -np.Inf
        BBGT = R['bbox']

        if BBGT.size > 0:
            # calc iou
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (BBGT[:, 2] - BBGT[:, 0] + 1.) * (
                        BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > iou_thres:
            # gt not matched yet
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    # return rec, prec, ap
    return npos, nd, tp[-1] / float(npos), tp[-1] / float(nd), ap
