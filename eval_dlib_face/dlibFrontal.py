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

import dlib

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


class DlibFrontalDetector(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def run(self, frame):
        dets = self.detector(frame, 1)
        return dets

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

    detector = dlib.get_frontal_face_detector()

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

        dets, scores, idx = detector.run(frame)
        # ここで検出結果の枠の扱いなどをそろえること

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


        for i in range(len(dets)):
            yLeftTop, xLeftTop, yRightBottom, xRightBottom = dets[i].top(), dets[i].left(), dets[i].bottom(), dets[i].right()
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
        names = glob.glob("headPose/Person*/*.jpg")[:20]
    elif dataset == "lfw":
        names = glob.glob("lfw/lfw/*/*.jpg")
    elif dataset == "cnn":
        names = glob.glob("cnn*/*/*.jpg")
    elif dataset == "att":
        names = glob.glob("att*/*/*.pgm")


    names.sort()
    processDatabase(dataset, names, 20)