# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
ce = tf.ceil
fl = tf.floor

def conv2d(inputs, filters, kernel_size, strides=1):
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs
    if strides > 1: 
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs

def _padding_layer(inputs, region, layer_size, history_res):
    xmin = region[0]
    ymin = region[1]
    xmax = region[2]
    ymax = region[3]
    xmin_2 = tf.maximum(xmin - 2, 0)
    ymin_2 = tf.maximum(ymin - 2, 0)
    xmax_2 = tf.minimum(xmax + 2, layer_size - 1)
    ymax_2 = tf.minimum(ymax + 2, layer_size - 1)
    if xmin_2 != xmin or xmax_2 != xmax or ymin_2 != ymin or ymax_2 != ymax:
        temp1 = history_res[: , xmin_2 : xmin, ymin : ymax , :]
        inputs = tf.concat([temp1, inputs], axis = 2)
        temp1 = history_res[: , xmax + 1 : xmax_2 + 1, ymin : ymax , :]
        inputs = tf.concat([inputs, temp1], axis = 2)
        temp1 = history_res[: , xmin_2 : xmax_2 + 1, ymin_2 : ymin , :]
        inputs = tf.concat([temp1, inputs], axis = 1)
        temp1 = history_res[:, xmin_2 : xmax_2 + 1, ymax + 1: ymax_2 + 1, :]
        inputs = tf.concat([inputs, temp1], axis = 1)
    temp = tf.shape(inputs)[3]       # the fourth dim (# of channels)
    res = tf.zeros(layer_size * layer_size * temp)
    res = res.reshape([1, layer_size, layer_size, temp])
    res[ymin_2 : ymax_2 + 1,xmin_2 : xmax_2 + 1] = inputs
    return res, [ymin_2, ymax_2, xmin_2, xmax_2]

def new_region(region, layer_type=1):
    xmin = region[0]
    ymin = region[1]
    xmax = region[2]
    ymax = region[3]
    if layer_type == 1:
        xmin = xmin + 1
        ymin = ymin + 1
        xmax = xmax - 1
        ymax = ymax - 1
    elif layer_type == 3:
        xmin = ce((xmin + 1) / 2)
        ymin = ce((ymin + 1) / 2)
        xmax = fl((xmax - 1) / 2)
        ymax = fl((ymax - 1) / 2)
        xmin = tf.cast(xmin, tf.int32)
        ymin = tf.cast(ymin, tf.int32)
        xmax = tf.cast(xmax, tf.int32)
        ymax = tf.cast(ymax, tf.int32)
    region = [xmin, ymin, xmax, ymax]
    return region

def same_region(region):
    return region

def same_net_region(net, region):
    return net, region

def darknet53_body(inputs, pad, input_n1, input_n2, input_n3, input_n4, input_n5, input_n6, input_n7, input_n8, input_n9, input_n10, input_n11, input_n12, input_n13, input_n14, input_n15, input_n16, input_n17, input_n18, input_n19, input_n20, input_n21, input_n22, input_n23, input_n24, input_n25, input_n26, input_n27, input_n28, input_n29, region):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)

        net = net + shortcut

        return net

    # first two conv2d layers
    
    net1 = conv2d(inputs, 32,  3, strides=1)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    net1, region = tf.cond(pad, lambda:same_net_region(net1, region), lambda:_padding_layer(net1, region, 416, input_n1))
    
    net2 = conv2d(net1, 64,  3, strides=2)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region,layer_type=3))
    net2, region = tf.cond(pad, lambda:same_net_region(net2, region), lambda:_padding_layer(net2, region, 208, input_n2))
    
    # res_block * 1
    net3 = res_block(net2, 32)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    net3, region = tf.cond(pad, lambda:same_net_region(net3, region), lambda:_padding_layer(net3, region, 208, input_n3))
    
    net4 = conv2d(net3, 128, 3, strides=2)
    region = tf.cond(pad, lambda:same_region(region), lambda:new_region(region, layer_type=3))
    net4, region = tf.cond(pad, lambda: same_net_region(net4, region), lambda: _padding_layer(net4, region, 104, input_n4))

    # res_block * 2
    #for i in range(2):
    net5 = res_block(net4, 64)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    net5, region = tf.cond(pad, lambda: same_net_region(net5, region), lambda: _padding_layer(net5, region, 104, input_n5))

    net6 = res_block(net5, 64)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    net6, region = tf.cond(pad, lambda: same_net_region(net6, region), lambda: _padding_layer(net6, region, 104, input_n6))

    net7 = conv2d(net6, 256, 3, strides=2)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region, layer_type=3))
    net7, region = tf.cond(pad, lambda: same_net_region(net7, region), lambda: _padding_layer(net7, region, 52, input_n7))

    # res_block * 8
    net8 = res_block(net7, 128)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    net8, region = tf.cond(pad, lambda: same_net_region(net8, region), lambda: _padding_layer(net8, region, 52, input_n8))

    net9 = res_block(net8, 128)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    net9, region = tf.cond(pad, lambda: same_net_region(net9, region), lambda: _padding_layer(net9, region, 52, input_n9))

    net10 = res_block(net9, 128)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    net10, region = tf.cond(pad, lambda: same_net_region(net10, region), lambda: _padding_layer(net10, region, 52, input_n10))

    net11 = res_block(net10, 128)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    net11, region = tf.cond(pad, lambda: same_net_region(net11, region), lambda: _padding_layer(net11, region, 52, input_n11))

    net12 = res_block(net11, 128)
    region = tf.cond(pad, same_region(region), new_region(region))
    net12, region = tf.cond(pad, lambda: same_net_region(net12, region), lambda: _padding_layer(net12, region, 52, input_n12))

    net13 = res_block(net12, 128)
    region = tf.cond(pad, lambda: same_region(region), new_region(region))
    net13, region = tf.cond(pad, lambda: same_net_region(net13, region), lambda: _padding_layer(net13, region, 52, input_n13))

    net14 = res_block(net13, 128)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    net14, region = tf.cond(pad, lambda: same_net_region(net14, region), lambda: _padding_layer(net14, region, 52, input_n14))

    net15 = res_block(net14, 128)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    net15, region = tf.cond(pad, lambda: same_net_region(net15, region), lambda: _padding_layer(net15, region, 52, input_n15))

    #route_1 = net15
    region_1 = region
    
    net16 = conv2d(net15, 512, 3, strides=2)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region, layer_type=3))
    net16, region = tf.cond(pad, lambda: same_net_region(net16, region), lambda: _padding_layer(net16, region, 26, input_n16))

    # res_block * 8
    net17 = res_block(net16, 256)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    net17, region = tf.cond(pad, lambda: same_net_region(net17, region), lambda: _padding_layer(net17, region, 26, input_n17))

    net18 = res_block(net17, 256)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    net18, region = tf.cond(pad, lambda: same_net_region(net18, region), lambda: _padding_layer(net18, region, 26, input_n18))

    net19 = res_block(net18, 256)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    net19, region = tf.cond(pad, lambda: same_net_region(net19, region), lambda: _padding_layer(net19, region, 26, input_n19))

    net20 = res_block(net19, 256)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    net20, region = tf.cond(pad, lambda: same_net_region(net20, region), lambda: _padding_layer(net20, region, 26, input_n20))

    net21 = res_block(net20, 256)
    region = tf.cond(pad, lambda: same_region(region), new_region(region))
    net21, region = tf.cond(pad, lambda: same_net_region(net21, region), lambda: _padding_layer(net21, region, 26, input_n21))

    net22 = res_block(net21, 256)
    region = tf.cond(pad, lambda: same_region(region), new_region(region))
    net22, region = tf.cond(pad, lambda: same_net_region(net22, region), lambda: _padding_layer(net22, region, 26, input_n22))

    net23 = res_block(net22, 256)
    region = tf.cond(pad, lambda: same_region(region), new_region(region))
    net23, region = tf.cond(pad, lambda: same_net_region(net23, region), lambda: _padding_layer(net23, region, 26, input_n23))

    net24 = res_block(net23, 256)
    region = tf.cond(pad, lambda: same_region(region), new_region(region))
    net24, region = tf.cond(pad, lambda: same_net_region(net24, region), lambda: _padding_layer(net24, region, 26, input_n24))

    #route_2 = net24
    region_2 = region

    net25 = conv2d(net24, 1024, 3, strides=2)
    region = tf.cond(pad, lambda: same_region(region), new_region(region, layer_type=3))
    net25, region = tf.cond(pad, lambda: same_net_region(net25, region), lambda: _padding_layer(net25, region, 13, input_n25))

    # res_block * 4
    net26 = res_block(net25, 512)
    region = tf.cond(pad, lambda: same_region(region), new_region(region))
    net26, region = tf.cond(pad, lambda: same_net_region(net26, region), lambda: _padding_layer(net26, region, 13, input_n26))

    net27 = res_block(net26, 512)
    region = tf.cond(pad, lambda: same_region(region), new_region(region))
    net27, region = tf.cond(pad, lambda: same_net_region(net27, region), lambda: _padding_layer(net27, region, 13, input_n27))

    net28 = res_block(net27, 512)
    region = tf.cond(pad, lambda: same_region(region), new_region(region))
    net28, region = tf.cond(pad, lambda: same_net_region(net28, region), lambda: _padding_layer(net28, region, 13, input_n28))

    net29 = res_block(net28, 512)
    region = tf.cond(pad, lambda: same_region(region), new_region(region))
    net29, region = tf.cond(pad, lambda: same_net_region(net29, region), lambda: _padding_layer(net29, region, 13, input_n29))

    #route_3 = net29
    region_3 = region

    return region_1, region_2, region_3, net1, net2, net3, net4, net5, net6, net7, net8, net9, net10, net11, net12, net13, net14, net15, net16, net17, net18, net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29


def yolo_block(inputs, filters, his1, his3, his5, layersize, region, pad):
    yb_net1 = conv2d(inputs, filters * 1, 1)
    yb_net1, region = tf.cond(pad, lambda: same_net_region(yb_net1, region), lambda: _padding_layer(yb_net1, region, layersize, his1))

    yb_net2 = conv2d(yb_net1, filters * 2, 3)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))

    yb_net3 = conv2d(yb_net2, filters * 1, 1)
    yb_net3, region = tf.cond(pad, lambda: same_net_region(yb_net3, region), _padding_layer(yb_net3, region, layersize, his3))

    yb_net4 = conv2d(yb_net3, filters * 2, 3)
    region = tf.cond(pad, lambda: same_region(region), new_region(region))

    yb_net5 = conv2d(yb_net4, filters * 1, 1)
    region = tf.cond(pad, lambda: same_region(region), lambda: new_region(region))
    yb_net5, region = tf.cond(pad, lambda: same_net_region(yb_net5, region), _padding_layer(yb_net5, region, layersize, his5))
    xmin = region[0]
    ymin = region[1]
    xmax = region[1]
    ymax = region[2]
    yb_region = [2*xmin, 2*ymin, 2*xmax + 1, 2*ymax + 1]
    #route = yb_net5
    yb_net6 = conv2d(yb_net5, filters * 2, 3)
    region = tf.cond(pad, lambda: same_region(region), new_region(region))

    return yb_net1, yb_net3, yb_net5, yb_region, yb_net6, region


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    # TODO: Do we need to set `align_corners` as True?
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    return inputs

def _full_pad(inputs, region, his_info):
    xmin = region[0]
    ymin = region[1]
    xmax = region[2]
    ymax = region[3]
    res = his_info
    res[ymin : ymax + 1,xmin : xmax + 1] = inputs
    return res

def _zero_pad(inputs, region, size):
    xmin = region[0]
    ymin = region[1]
    xmax = region[2]
    ymax = region[3]
    temp = tf.shape(inputs)[3]       # the fourth dim (# of channels)
    res = tf.zeros(size * size * temp)
    res = res.reshape([1, size, size, temp])
    res[ymin : ymax + 1,xmin : xmax + 1] = inputs
    return res
def _none_pad(inputs):
    return inputs

