# eval_resnet_ssd_face

Evaluation script for resnet_ssd_face_python in OpenCV

## Requirements
OpenCV 3
Python 2.7
numpy
matplotlib
pandas
jupyter notebook

## Purpose
Evaluate resnet_ssd_face_python in OpenCV using some open database.

## Scripts
Python scripts are modidied from orginal resnet_ssd_face_python in OpenCV to use class.

### original scripts in OpenCV
opencv/samples/dnn/resnet_ssd_face_python.py
https://github.com/opencv/opencv/blob/24bed38c2b2c71d35f2e92aa66648f8485a70892/samples/dnn/resnet_ssd_face_python.py

### required data
Copy opencv/samples/dnn/face_detector/ as face_detector/

### resenetFaceDetector.py
a class version to use resenet ssd face.

### resnet_ssd_face_python_file.py
a script to process image files.

Evaluation scripts are written as jupyter notebook file (*.ipynb)

```
$ jupyter notebook
```

### evalate detection ratio
- log_lfw.ipynb
- log_headPose.ipynb
- log_cnn.ipynb


### evalate detection ratio with roll

- log_headPose_rotate.ipynb
- log_lfw_rotate.ipynb

### Results

You can see some results in the jupyter notebook file.
Some plots show that resnet_ssd_face_python in OpenCV has high detection ratio.


## note:
Some comments are written in Japanese.


##　追加考察すべきこと
- 顔の大きさによる検出率の変化
- 誤検出数の評価
-
