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
import resnetFaceDetector

import helper

def processDatabase(dataset, names, deg=0, scale=1.0, confThreshold=0.5, showImg=True):
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

    detector = resnetFaceDetector.ResnetFaceDetector()

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
        [h, w] = frame.shape[:2]
        imgCenter = [cols/2, rows/2]

        detections, perf_stats = detector.run(frame)

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
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confThreshold:
                xLeftTop = int(detections[0, 0, i, 3] * cols)
                yLeftTop = int(detections[0, 0, i, 4] * rows)
                xRightBottom = int(detections[0, 0, i, 5] * cols)
                yRightBottom = int(detections[0, 0, i, 6] * rows)
                width = xRightBottom - xLeftTop

                isPositive = helper.isInside(center, (xLeftTop, yLeftTop), (xRightBottom, yRightBottom))
                trueDetection[isPositive] += 1
                trueSizes.append(width)

                cv.circle(frame, (xLeftTop, yLeftTop), 5, (0, 255, 0))
                cv.circle(frame, (xRightBottom, yRightBottom), 5, (0, 255, 0))

                color = {True:(0, 255, 0), False:(0, 0, 128)}[isPositive]
                cv.rectangle(frame, (xLeftTop, yLeftTop), (xRightBottom, yRightBottom),
                             color)

                label = "face: %.4f" % confidence
                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv.rectangle(frame, (xLeftTop, yLeftTop - labelSize[1]),
                                    (xLeftTop + labelSize[0], yLeftTop + baseLine),
                                    (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (xLeftTop, yLeftTop),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        found = trueDetection[True] + trueDetection[False]
        log.write("%s, %d, %d, %d, %s\n" % (p, found, trueDetection[True], trueDetection[False], `np.mean(trueSizes)`))
        cv.imwrite(dstname, frame)

        if showImg:
            cv.imshow("resnet based (%d, %d)" % (w, h), frame)
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
        names = glob.glob("lfw/lfw/*/*.jpg")
    elif dataset == "cnn":
        names = glob.glob("cnn*/*/*.jpg")
    elif dataset == "att":
        names = glob.glob("att*/*/*.pgm")


    names.sort()
    print(names)

    processDatabase(dataset, names, scale=1.0, deg=30)