#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import numpy as np
import PIL.Image
import cv2 as cv

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