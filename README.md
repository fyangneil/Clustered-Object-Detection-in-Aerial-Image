# Clustered-Object-Detection-in-Aerial-Image
The repo is about our recent work on object detection in aerial image, the paper of the work "Clustered Object Detection in Aerial Image" (ICCV2019) and its supplementatry are available [here](https://drive.google.com/drive/folders/1qnqEXIkraCbdWW-WFRcLqIcLTKdPvyPc?usp=sharing).

# Installing codebase
1. The work is implemented based on [Caffe2](https://caffe2.ai/docs/getting-started.html?platform=windows&configuration=compile) , please install it according to the corresponding instruction.
2. git clone git@github.com:fyangneil/ClusDet.git. Please follow the instruction in [Detectron](https://github.com/facebookresearch/Detectron) to install the repo.
# Generating cluster region ground truth
Here, we use VisDrone dataset as an example to demonstrate the process to generate cluster region ground truth.
1. run "./detectron/ops/add_cluster_annotation.m" to generate cluster ground truth and add it to original object annotation files.
2. run "./detectron/ops/visdrone2cocoformat.m" to convert VisDrone format annotation to COCO format.

# Train CPNet and global detector
```shell
cd $ROOT_DIR/ClusDet
python ./tools/train_net.py \
    --cfg ./configs/e2e_faster_rcnn_R-50-FPN_CPNet_1x_1GPU.yaml \
    OUTPUT_DIR ./trainedmodel/faster_rcnn_R-50-FPN_CPNet_1x_1GPU
```
# Inference CPNet to produce cluster regions on global image
```shell
python tools/test_net.py \
    --cfg ./configs/e2e_faster_rcnn_R-50-FPN_CPNet_1x_1GPU.yaml \
    TEST.WEIGHTS ./trainedmodel/faster_rcnn_R-50-FPN_CPNet_1x_1GPU/train/coco_2014_train/generalized_rcnn/model_final.pkl \
    NUM_GPUS 1
 ``` 
 crop cluster regions by running 
 ```shell
python detectron/ops/crop_cluster_proposals.py
 ``` 
 Please Change the corresponding path when used on your computer.
# Train detector on global images and cropped cluster chips
  ```shell
 python ./tools/train_net.py \
    --cfg ./configs/e2e_faster_rcnn_R-50-FPN_1x_1GPU.yaml \
    OUTPUT_DIR ./trainedmodel/faster_rcnn_R-50-FPN_1x_1GPU
```
# Inference detector on global images and cropped cluster chips
```shell
python tools/test_net.py \
    --cfg ./configs/e2e_faster_rcnn_R-50-FPN_1x_1GPU.yaml \
    TEST.WEIGHTS ./trainedmodel/faster_rcnn_R-50-FPN_1x_1GPU/train/coco_2014_train/generalized_rcnn/model_final.pkl \
    NUM_GPUS 1
 ``` 
# Fuse the detections from global images and cluster chips
 run "./detectron/ops/fuse_global_cluster_detections.m"
# Cite 
@InProceedings{Yang_2019_ICCV,
author = {Yang, Fan and Fan, Heng and Chu, Peng and Blasch, Erik and Ling, Haibin},
title = {Clustered Object Detection in Aerial Images},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}


 
