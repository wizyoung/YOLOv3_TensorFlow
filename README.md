#  YOLOv3_TensorFlow

### 1. Introduction

This is my implementation of [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) in pure TensorFlow. It contains the full pipeline of training and evaluation on your own dataset. The keys features of this repo are:

- Efficient tf.data pipeline

- Weights converter (converting pretrained darknet weights on COCO dataset to TensorFlow checkpoint.)
- Extremely fast GPU non maximum supression.
- Full training pipeline.
- Kmeans algorithm to select prior anchor boxes.

### 2. Requirements

- tensorflow >= 1.8.0 (lower versions may work too)
- opencv-python

### 3. Weights convertion

The pretrained darknet weights file can be downloaded [here](https://pjreddie.com/media/files/yolov3.weights). Place this weights file under directory `./data/darknet_weights/` and then run:

```shell
python convert_weight.py
```

Then the converted TensorFlow checkpoint file will be saved to `./data/darknet_weights/` directory.

You can also download the converted TensorFlow checkpoint file by me via [Google Drive link](https://drive.google.com/drive/folders/1mXbNgNxyXPi7JNsnBaxEv1-nWr7SVoQt?usp=sharing) and then place it to the same directory.

### 4. Running demos

There are some demo images and videos under the `./data/demo_data/`. You can run the demo by:

Single image test demo:

```shell
python test_single_image.py ./data/demo_data/messi.jpg
```

Video test demo:

```shell
python video_test.py ./data/demo_data/video.mp4
```

Some results:

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/data/demo_data/results/dog.jpg?raw=true)

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/data/demo_data/results/messi.jpg?raw=true)

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/data/demo_data/results/kite.jpg?raw=true)

(The kite result is under image resolution 1344x896)

#### 5. Inference speed

How fast is the inference speed? With images scaled to 416*416:


| Backbone              |   GPU    | Time(ms) |
| :-------------------- | :------: | :------: |
| Darknet-53 (paper)    | Titan X  |    29    |
| Darknet-53 (my impl.) | Titan XP |   ~23    |

### 6. Training

#### 6.1 Data preparation 

(1) annotation file

Generate `train.txt/val.txt/test.txt` files under `./data/my_data/` directory. One line for one image, in the format like `image_absolute_path box_1 box_2 ... box_n`. Box_format: `label_index x_min y_min x_max y_max`.

For example:

```
xxx/xxx/1.jpg 0 453 369 473 391 1 588 245 608 268
xxx/xxx/2.jpg 1 466 403 485 422 2 793 300 809 320
...
```

**NOTE**: **You should leave a blank line at the end of each txt file.**

(2)  class_names file:

Generate the `data.names` file under `./data/my_data/` directory. Each line represents a class name.

For example:

```
bird
person
bike
...
```

The COCO dataset class names file is placed at `./data/coco.names`.

(3) prior anchor file:

Using the kmeans algorithm to get the prior anchors:

```
python get_kmeans.py
```

Then you will get 9 anchors and the average IOU. Save the anchors to a txt file.

The COCO dataset anchors offered by YOLO v3 author is placed at `./data/yolo_anchors.txt`, you can use that one too.

**NOTE: The yolo anchors should be scaled to the rescaled new image size. Suppose your image size is [W, H], and the image will be rescale to 416*416 as input, for each generated anchor [anchor_w, anchor_h], you should apply the transformation anchor_w = anchor_w / W * 416, anchor_h = anchor_g / H * 416.**

#### 6.2 Training

Using `train.py`. The parameters are as following:

```shell
$ python train.py -h
usage: train.py [-h] [--train_file TRAIN_FILE] 
				[--val_file VAL_FILE]
                [--restore_path RESTORE_PATH] 
                [--save_dir SAVE_DIR]
                [--log_dir LOG_DIR] 
                [--progress_log_path PROGRESS_LOG_PATH]
                [--anchor_path ANCHOR_PATH]
                [--class_name_path CLASS_NAME_PATH] [--batch_size BATCH_SIZE]
                [--img_size [IMG_SIZE [IMG_SIZE ...]]]
                [--total_epoches TOTAL_EPOCHES]
                [--train_evaluation_freq TRAIN_EVALUATION_FREQ]
                [--val_evaluation_freq VAL_EVALUATION_FREQ]
                [--save_freq SAVE_FREQ] [--num_threads NUM_THREADS]
                [--prefetech_buffer PREFETECH_BUFFER]
                [--optimizer_name OPTIMIZER_NAME]
                [--save_optimizer SAVE_OPTIMIZER]
                [--learning_rate_init LEARNING_RATE_INIT] [--lr_type LR_TYPE]
                [--lr_decay_freq LR_DECAY_FREQ]
                [--lr_decay_factor LR_DECAY_FACTOR]
                [--lr_lower_bound LR_LOWER_BOUND]
                [--restore_part [RESTORE_PART [RESTORE_PART ...]]]
                [--update_part [UPDATE_PART [UPDATE_PART ...]]]
```

Check the `train.py` for more details. You should set the parameters yourself. 

Some training tricks in my experiment:

(1) Apply the two-stage training strategy:

First stage: Restore `darknet53_body` part weights from COCO checkpoints, train the `yolov3_head` with big learning rate like 1e-3 until the loss reaches to a low level, like less than 1.

Second stage: Restore the weights from the first stage, then train the whole model with small learning rate like 1e-4 or smaller. At this stage remember to restore the optimizer parameters if you use optimizers like adam.

(2) Quick train:

If you want to obtain good results in a short time like in 10 minutes. You can use the coco names but substitute several with real class names in your dataset. In this way you restore the whole pretrained COCO model and get a 80 class classification model, but you only care the class names from your dataset.

### 7. Evaluation

Using `eval.py` to evaluate the validation or test dataset. The parameters are as following:

```shell
$ python eval.py -h
usage: eval.py [-h] [--eval_file EVAL_FILE] 
			   [--restore_path RESTORE_PATH]
               [--anchor_path ANCHOR_PATH] 
               [--class_name_path CLASS_NAME_PATH]
               [--batch_size BATCH_SIZE]
               [--img_size [IMG_SIZE [IMG_SIZE ...]]]
               [--num_threads NUM_THREADS]
               [--prefetech_buffer PREFETECH_BUFFER]
```

Check the `eval.py` for more details. You should set the parameters yourself. 

You will get the loss, recall and precision metrics results, like:

```shell
recall: 0.927, precision: 0.945
total_loss: 0.210, loss_xy: 0.010, loss_wh: 0.025, loss_conf: 0.125, loss_class: 0.050
```

### 8. Other skills

There are many skills you can try during training:

(1) Data augmentation: You can implement your data augmentation like color jittering under `data_augmentation` method in `./utils/data_utils.py`.

(2) Mutil-scale training: You can change the input image scales periodically like the author does in the original paper.

### 9. TODO

- [ ] Multi-GPU training with sync batch norm. 

-------

### Credits:

I refer to many fantastic repos during the implementation:

https://github.com/YunYang1994/tensorflow-yolov3

https://github.com/qqwweee/keras-yolo3

https://github.com/eriklindernoren/PyTorch-YOLOv3

https://github.com/pjreddie/darknet





 