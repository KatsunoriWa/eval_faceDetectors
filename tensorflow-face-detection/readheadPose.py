#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101
import glob
import os
import math
import numpy as np

def getCenterAndWH(name):
    lines = open(name, "rt").readlines()
    centerAndWH = [int(line) for line in lines[3:7]]
    return centerAndWH

def getTruePosition():
    txtnames = glob.glob("headPose/Per*/*.txt")
    d = {}
    for p in txtnames:
        jpgname = p.replace(".txt", ".jpg")
        d[jpgname] = getCenterAndWH(p)

    return d

def  cosd(deg):
    return math.cos(math.pi*deg/180.0)

def  sind(deg):
    return math.sin(math.pi*deg/180.0)


def getRotatedPoint(point, deg, imgCenter):
    a = np.array(point) - np.array(imgCenter)
    deg = -deg
    rotated = np.array((cosd(deg)*a[0] -sind(deg)*a[1], sind(deg)*a[0] +cosd(deg)*a[1]), dtype=np.int)
    r = rotated + imgCenter
    return (int(r[0]), int(r[1]))

def getAngles(p):
    base = os.path.basename(p)
    base = os.path.splitext(base)[0]
    base = base.replace("+", "_+").replace("-", "_-")
    f = base.split("_")
    return f[-2:]

if __name__ == "__main__":
    d = getTruePosition()

    size = [384, 288]
    imgCenter = (size[0]/2, size[1]/2)

    keys = d.keys()
    keys.sort()

    deg = 90
    for k in keys:
        print k, d[k],
        point = tuple(d[k][:2])
        print getRotatedPoint(point, deg, imgCenter)
