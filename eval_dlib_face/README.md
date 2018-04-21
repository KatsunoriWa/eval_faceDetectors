# Evaluation for Dlib frontal face
Evaluation script for Dlib frontal face

## Requirements
OpenCV 3
Python 2.7
numpy
matplotlib
dlib
pandas
jupyter notebook

## Purpose
Evaluate the detector using some open database.

## Scripts
Python scripts are modidied from orginal sample script from Dlib to use class.

### dlibFrontal.py
a class version to use Dlib frontal face detector.

Evaluation scripts are written as jupyter notebook file (*.ipynb)

```
$ jupyter notebook
```


### evaluate detection ratio
- dlib_headPose.ipynb
- dlib_lfw.ipynb
- dlib_cnn.ipynb


### evaluate detection ratio with roll

- dlib_headPose_rotate.ipynb
- dlib_lfw_rotate.ipynb


### evaluate detection ratio with size

dlib_headPose_size.ipynb
dlib_lfw_size.ipynb


### Results

You can see some results in the jupyter notebook file.


## note:
Dlib has tow types of face_detector.
http://dlib.net/face_detector.py.html
http://dlib.net/cnn_face_detector.py.html


Some comments are written in Japanese.
