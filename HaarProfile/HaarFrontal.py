#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

# Python 2/3 compatibility
from __future__ import print_function

import sys
import os
import glob
import numpy as np
import cv2 as cv

import readheadPose

import helper


class HaarCascadeDetector(object):
    def __init__(self):
		self.face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
		self.face_cascadeP = cv.CascadeClassifier('haarcascade_profileface.xml')

    def run(self, frame):
        dets = self.face_cascade.detectMultiScale(frame, 1.3, 5)
        detsP = self.face_cascadeP.detectMultiScale(frame, 1.3, 5)
        if len(dets) > 0 and len(detsP) > 0:
            detsnew = np.r_[dets, detsP]
            return detsnew, "noScore", "noIdx"
        elif len(dets) == 0 and len(detsP) > 0:
            return detsP, "noScore", "noIdx"
        elif len(dets) > 0 and len(detsP) == 0:
            return dets, "noScore", "noIdx"
        else:
            return dets, "noScore", "noIdx"


def processDatabase(dataset, names, deg=0, scale=1.0, min_score_thresh=0.7, showImg=True):
    """run face detection for named dataset as names.
    dataset:
    names:
    deg: angle (anti-clockwise)
    """
    if dataset == "headPose":
        import readheadPose
        d = readheadPose.getTruePosition()


    log = open("log_%s_%d_%f.csv" % (dataset, deg, scale), "wt")
    log.write("name,num,truePositives,falsePositives,meanSize\n")

    detector = HaarCascadeDetector()

    windowNotSet = True

    for p in names:
        dstDir = "result"
        dstname = os.path.join(dstDir, p)
        dirname = os.path.dirname(dstname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        frame = cv.imread(p)
        if deg != 0:
            frame = helper.rotate(frame, deg)

        [h, w] = frame.shape[:2]
        scaledImg = helper.scaledImage(frame, scale)
        frame = scaledImg


        cols = frame.shape[1]
        rows = frame.shape[0]
        imgCenter = [cols/2, rows/2]

        dets, scores, idx = detector.run(frame)
        # ここで検出結果の枠の扱いなどをそろえること

        trueDetection = {True:0, False:0}

        if dataset in ("lfw", ):
            center = imgCenter
            center = (int(scale*center[0]), int(scale*center[1]))

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
            center = (int(scale*center[0]), int(scale*center[1]))

        trueSizes = []

        for i in range(len(dets)):
            x, y, w, h = dets[i]
            yLeftTop, xLeftTop, yRightBottom, xRightBottom = y, x, y+h, x+w
            yLeftTop, xLeftTop, yRightBottom, xRightBottom = int(yLeftTop), int(xLeftTop), int(yRightBottom), int(xRightBottom)

            if scores[i] <= min_score_thresh:
                continue

            isPositive = helper.isInside(center, (xLeftTop, yLeftTop), (xRightBottom, yRightBottom))

            trueDetection[isPositive] += 1
            trueSizes.append(xRightBottom - xLeftTop)

            cv.circle(frame, (xLeftTop, yLeftTop), 5, (0, 255, 0))
            cv.circle(frame, (xRightBottom, yRightBottom), 5, (0, 255, 0))

            color = {True:(0, 255, 0), False:(0, 0, 255)}[isPositive]
            cv.rectangle(frame, (xLeftTop, yLeftTop), (xRightBottom, yRightBottom),
                         color, 5)

        found = trueDetection[True] + trueDetection[False]
        log.write("%s, %d, %d, %d, %s\n" % (p, found, trueDetection[True], trueDetection[False], `np.mean(trueSizes)`))

        if windowNotSet is True:
            cv.namedWindow("tensorflow based (%d, %d)" % (cols, rows), cv.WINDOW_NORMAL)
            windowNotSet = False

        if showImg:
            cv.imshow("tensorflow based (%d, %d)" % (cols, rows), frame)
            k = cv.waitKey(1) & 0xff
            if k == ord('q') or k == 27:
                break

    log.close()
    cv.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) == 0:
        print("""usage: %s (headPose | lfw | cnn)
        """ % sys.argv[0])
        exit()



    dataset = "headPose"
#    dataset = "lfw"
#    dataset = "cnn"
#    dataset = "att"

    if dataset == "headPose":
        names = glob.glob("headPose/Person*/*.jpg")
    elif dataset == "lfw":
        names = glob.glob("lfw/lfw/*/*.jpg")[:20]
    elif dataset == "cnn":
        names = glob.glob("cnn*/*/*.jpg")
    elif dataset == "att":
        names = glob.glob("att*/*/*.pgm")


    names.sort()
#    print names
    processDatabase(dataset, names, 20, scale=0.5)
