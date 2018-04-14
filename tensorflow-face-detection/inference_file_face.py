#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import os
import glob
import numpy as np
import cv2 as cv
import PIL.Image
import time

import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils_color as vis_util

import readheadPose

def rotate(img, deg):
    """rotate anti-clockwise
    img: numpy image
    deg:
    """
    pilImg = PIL.Image.fromarray(np.uint8(img))
    rotated = pilImg.rotate(deg)
    return np.asarray(rotated)+0

def isInside(point, leftTop, rightBottom):
    """
    return True if point is in the rectangle define by leftTop and rightBottom
    """

    if not (leftTop[0] < point[0] < rightBottom[0]):
        return False
    if not (leftTop[1] < point[1] < rightBottom[1]):
        return False
    return True

def centerIsInRect(shape, leftTop, rightBottom):
    center = (shape[1]/2, shape[0]/2)
    return isInside(center, leftTop, rightBottom)

def scaledImage(img, scale, keepFullSize=True):
    """
    return resized image
    """

    [h, w] = img.shape[:2]
    scaledSize = (int(round(w*scale)), int(round(h*scale)))

    scaledImg = cv.resize(img, scaledSize)

    if not keepFullSize:
        return scaledImg
    else:
        newImg = np.zeros((img.shape), dtype=np.uint8)
        newImg[0:scaledSize[1], 0:scaledSize[0], :] = scaledImg
        return newImg


def getGategoryIndex():
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = './protos/face_label_map.pbtxt'

    NUM_CLASSES = 2

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

class TensoflowFaceDector(object):
    def __init__(self):
        """Tensorflow detector
        """

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'


        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(graph=self.detection_graph, config=config) as self.sess:

                self.windowNotSet = True


    def run(self, frame):
        """frame: bgr frame
        return (boxes, scores, classes, num_detections)
        """

        image_np = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # the array based representation of the frame will be used later in order to prepare the
        # result frame with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the frame where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result frame, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

#        print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)

def processDatabase(dataset, names, deg=0, scale=1.0, min_score_thresh=0.7, showImg=True):
    """run face detection for named dataset as names.
    dataset:
    names:
    deg: angle (anti-clockwise)
    """
    if dataset == "headPose":
        import readheadPose
        d = readheadPose.getTruePosition()


    log = open("log_%s_%d.csv" % (dataset, deg), "wt")
    log.write("name,num,truePositives,falsePositives\n")

    tDetector = TensoflowFaceDector()
    category_index = getGategoryIndex()

    windowNotSet = True

    for p in names:
        dstDir = "result"
        dstname = os.path.join(dstDir, p)
        dirname = os.path.dirname(dstname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        frame = cv.imread(p)
        if deg != 0:
            frame = rotate(frame, deg)

        [h, w] = frame.shape[:2]
        scaledImg = scaledImage(frame, scale)
        frame = scaledImg


        cols = frame.shape[1]
        rows = frame.shape[0]
        [h, w] = frame.shape[:2]
        imgCenter = [cols/2, rows/2]

        (boxes, scores, classes, num_detections) = tDetector.run(frame)


#        vis_util.visualize_boxes_and_labels_on_image_array(
#            frame,
#            boxes,
#            classes.astype(np.int32),
#            scores,
#            category_index,
#            use_normalized_coordinates=True,
#            line_thickness=4)


        trueDetection = {True:0, False:0}

        if dataset in ("lfw", ):
            center = imgCenter
        elif dataset == "headPose":
            v = d[p]
            center = (v[0], v[1])
            center = readheadPose.getRotatedPoint(center, deg, imgCenter)
            #　ここで縮小したことによる画像の点の扱いを修正すること
            center = (int(scale*center[0]), int(scale*center[1]))

            r = int(50*scale)

            cv.circle(frame, center, r, (0, 255, 0))
        else:
            center = imgCenter


        boxes_shape = boxes.shape
        for i in range(boxes_shape[0]):
            ymin, xmin, ymax, xmax = boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3]
            yLeftTop, xLeftTop, yRightBottom, xRightBottom = ymin * h, xmin * w, ymax * h, xmax * w
            yLeftTop, xLeftTop, yRightBottom, xRightBottom = int(yLeftTop), int(xLeftTop), int(yRightBottom), int(xRightBottom)

            if scores[i] <= min_score_thresh:
                continue

            isPositive = isInside(center, (xLeftTop, yLeftTop), (xRightBottom, yRightBottom))

            trueDetection[isPositive] += 1

            cv.circle(frame, (xLeftTop, yLeftTop), 5, (0, 255, 0))
            cv.circle(frame, (xRightBottom, yRightBottom), 5, (0, 255, 0))

            color = {True:(0, 255, 0), False:(0, 0, 255)}[isPositive]
            cv.rectangle(frame, (xLeftTop, yLeftTop), (xRightBottom, yRightBottom),
                         color, 5)

        found = trueDetection[True] + trueDetection[False]
        log.write("%s, %d, %d, %d\n" % (p, found, trueDetection[True], trueDetection[False]))

        if windowNotSet is True:
            cv.namedWindow("tensorflow based (%d, %d)" % (w, h), cv.WINDOW_NORMAL)
            windowNotSet = False

        if showImg:
            cv.imshow("tensorflow based (%d, %d)" % (w, h), frame)
            k = cv.waitKey(1) & 0xff
            if k == ord('q') or k == 27:
                break

    log.close()
    cv.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) == 0:
        print """usage: %s (headPose | lfw | cnn)
        """ % sys.argv[0]
        exit()



    dataset = "headPose"
#    dataset = "lfw"
#    dataset = "cnn"
#    dataset = "att"

    if dataset == "headPose":
        names = glob.glob("headPose/Person*/*.jpg")
    elif dataset == "lfw":
        names = glob.glob("lfw/lfw/*/*.jpg")
    elif dataset == "cnn":
        names = glob.glob("cnn*/*/*.jpg")
    elif dataset == "att":
        names = glob.glob("att*/*/*.pgm")


    names.sort()
    processDatabase(dataset, names, 20, scale=0.5)