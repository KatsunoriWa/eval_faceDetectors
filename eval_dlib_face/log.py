#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
import os
def getAngles(p):
    base = os.path.basename(p)
    base = base.replace("+", "_+").replace("-", "_-")
    f = base.split("_")
    return f[-2:]
    
import pandas as pd
df = pd.read_csv("log.csv")

pitches = []
yaws = []
angles = []
for index, rows in df.iterrows():
#    print index, rows["name"]
    pitch, yaw = getAngles(rows["name"])
    pitches.append(pitch)    
    yaws.append(yaw)    
    angles.append("%s_%s" % (pitch, yaw))

df["angles"] = angles
df["num"].hist()
print df.groupby("num").count()
print df.groupby("num").count()/float(df.shape[0])
#print df.groupby("angles", "num").count()

