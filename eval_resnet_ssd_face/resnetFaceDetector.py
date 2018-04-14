#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101
import cv2 as cv
from cv2 import dnn

class ResnetFaceDetector(object):
    def __init__(self):
        """ResnetFaceDetecor class rewrite version
        """

        prototxt = 'face_detector/deploy.prototxt'
        caffemodel = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
        self.inWidth = 300
        self.inHeight = 300
        self.net = dnn.readNetFromCaffe(prototxt, caffemodel)

    def run(self, image):
        """
        image: input image
        return: detections, perf_stats
        """

        self.net.setInput(dnn.blobFromImage(image, 1.0, (self.inWidth, self.inHeight), (104.0, 177.0, 123.0), False, False))
        self.detections = self.net.forward()

        self.perf_stats = self.net.getPerfProfile()
        return self.detections, self.perf_stats

if __name__ == '__main__':
    detector = ResnetFaceDetector()

    confThreshold = 0.5

    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret is None:
            continue

        cols = frame.shape[1]
        rows = frame.shape[0]
        detections, perf_stats = detector.run(frame)

        print('Inference time, ms: %.2f' % (perf_stats[0] / cv.getTickFrequency() * 1000))

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confThreshold:
                xLeftTop = int(detections[0, 0, i, 3] * cols)
                yLeftTop = int(detections[0, 0, i, 4] * rows)
                xRightBottom = int(detections[0, 0, i, 5] * cols)
                yRightBottom = int(detections[0, 0, i, 6] * rows)

                cv.circle(frame, (xLeftTop, yLeftTop), 5, (0, 255, 0))
                cv.circle(frame, (xRightBottom, yRightBottom), 5, (0, 255, 0))

                cv.rectangle(frame, (xLeftTop, yLeftTop), (xRightBottom, yRightBottom),
                             (0, 255, 0))
                label = "face: %.4f" % confidence
                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv.rectangle(frame, (xLeftTop, yLeftTop - labelSize[1]),
                                    (xLeftTop + labelSize[0], yLeftTop + baseLine),
                                    (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (xLeftTop, yLeftTop),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("detections", frame)
        k = cv.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break
