import tensorflow as tf
import numpy as np
import argparse
import cv2
import glob
import matplotlib.pyplot as plt
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import batch_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize
from model import yolov3

anchors = parse_anchors("./data/yolo_anchors.txt")
classes = read_class_names("./data/coco.names")
num_class = len(classes)

color_table = get_color_table(num_class)

with tf.Session() as sess:
    # build graph
    input_data = tf.placeholder(tf.float32, [None, None, None, 3], name='input')
    yolo_model = yolov3(num_class, anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    pred_scores = pred_confs * pred_probs
    boxes, scores, labels, num_dects = batch_nms(pred_boxes, pred_scores, max_boxes=20, score_thresh=0.5, nms_thresh=0.5)
    # restore weight
    saver = tf.train.Saver()
    saver.restore(sess, "./data/darknet_weights/yolov3.ckpt")
    # save
    output_node_names = [
        "output/boxes",
        "output/scores",
        "output/labels",
        "output/num_detections",
        "input",
    ]
    output_node_names = ",".join(output_node_names)

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        output_node_names.split(",")
    )

    with tf.gfile.GFile('./data/darknet_weights/yolov3_frozen_graph_batch.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("{} ops written to {}.".format(len(output_graph_def.node), './data/darknet_weights/yolov3_frozen_graph_batch.pb'))

