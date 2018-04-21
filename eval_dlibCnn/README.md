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
Python scripts are modidied from orginal sample script from Dlib to use similar class.
http://dlib.net/cnn_face_detector.py.html

### dlibCnnFace.py
a class wrapper version to use Dlib CNN face detector.

Evaluation scripts are written as jupyter notebook file (*.ipynb)

```
$ jupyter notebook
```

### evaluate detection ratio
- dlibCnn_headPose.ipynb
- dlibCnn_lfw.ipynb
- dlibCnn_cnn.ipynb

### evaluate detection ratio with roll
- dlibCnn_headPose_rotate.ipynb
- dlibCnn_lfw_rotate.ipynb

### evaluate detection ratio with size
- dlibCnn_headPose_size.ipynb
- dlibCnn_lfw_size.ipynb

### Results

You can see some results in the jupyter notebook file.


## note:
Dlib has tow types of face_detector.
http://dlib.net/face_detector.py.html
http://dlib.net/cnn_face_detector.py.html


Some comments are written in Japanese.
