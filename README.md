# eval_faceDetectors
evaluation of face detectors including opencv dnn face detector

- You can evaluate face detection in each face detector in Jupyter environment.

- Some modules are wrapped by class interface


## Haar Cascade Face Detector in opencv

## resnet ssd Face Detector in opencv

## dlib frontal Face detector

## dlib CNN Face detector

## tensorflow Face detector


|detector| pretrained_model | input image size|
|----|----|----|
|OpenCV Haar Cascade| haarcascade_frontalface_default.xml  | no upper limit   |
|OpenCV resnet ssd face detcator | res10_300x300_ssd_iter_140000.caffemodel |  300x300 |
|dlib FrontalFace |                                      |                  |
|dlib cnn_face_detector |  mmod_human_face_detector.dat |                  |
|tensorflow Face detector | frozen_inference_graph_face.pb |                  |




### memo in Japanese

顔検出器の評価のポイント
- 検出率
- 検出にかかる時間・計算量

顔画像の検出器は、撮影される画像の素性による。
LFWの画像の場合は、
顔照合を目的とした画像セットであるため、
顔の顔向きがほぼ正面顔である。
