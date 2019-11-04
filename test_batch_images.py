import tensorflow as tf

class YoloV3:
    """Class to load ssd model and run inference."""
    INPUT_NAME = 'input:0'
    BOXES_NAME = 'output/boxes:0'
    CLASSES_NAME = 'output/labels:0'
    SCORES_NAME = 'output/scores:0'
    NUM_DETECTIONS_NAME = 'output/num_detections:0'
    def __init__(self, frozen_graph):
        self.graph = tf.Graph()
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)

        if graph_def is None:
              raise RuntimeError('Cannot find inference graph.')

        with self.graph.as_default():
              tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)
    def run(self, image):
        """
        image should be normalized to [0,1] and RGB order
        """
        boxes, classes, scores, num_detections = self.sess.run(
                    [self.BOXES_NAME, self.CLASSES_NAME, self.SCORES_NAME, self.NUM_DETECTIONS_NAME],
                    feed_dict={self.INPUT_NAME: image})
        return boxes, classes.astype(np.int64), scores, num_detections.astype(np.int64)

if __name__ == '__main__':
    import os
    import glob
    import numpy as np
    import cv2

    from utils.plot_utils import get_color_table, plot_one_box
    from utils.misc_utils import parse_anchors, read_class_names

    model = YoloV3('./data/darknet_weights/yolov3_frozen_graph_batch.pb')
    classes = read_class_names("./data/coco.names")
    color_table = get_color_table(80)
    files = glob.glob('./data/demo_data/*.jpg')
    images = []
    vis_images = []
    for file in files:
        image = cv2.imread(file)
        image = cv2.resize(image, (640, 640))
        vis_images.append(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255. #important!
        images.append(image)
    images = np.array(images)
    # inference
    boxes_,labels_,scores_, num_dect_= model.run(images)
    # visualize
    for idx, image in enumerate(vis_images):
        for i in range(len(boxes_[idx])):
            x0, y0, x1, y1 = boxes_[idx][i]
            plot_one_box(image, [x0, y0, x1, y1], label=classes[labels_[idx][i]] + ', {:.2f}%'.format(scores_[idx][i] * 100), color=color_table[labels_[idx][i]])
        out_name = os.path.join('./data/demo_data/results', 'batch_output_' + os.path.basename(files[idx]))
        cv2.imwrite(out_name, image)