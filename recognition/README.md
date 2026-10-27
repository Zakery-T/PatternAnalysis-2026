# Recognition Tasks
Various recognition tasks solved in deep learning frameworks.

Tasks may include:
* Image Segmentation
* Object detection
* Graph node classification
* Image super resolution
* Disease classification
* Generative modelling with StyleGAN and Stable Diffusion

# Pattern Recognition 2026 Project: Lesion Detection using YOLOv8
---
Name: Zakery Tsin
Student ID: s48741510
Course: COMP3710, S2 2026

Please note that this README.md (currently) contains my decomposition of the problem as well as research conducted for understand and this problem.

## Key Project Information
Dataset: ISIC2017
Model: YOLOv8 (pre-trained models available via ultralytics)
Description: Developing a model based on YOLOv8 which detects lesions within the ISIC 2017 dataset. Analyse how this model balances precision and recall, and in turn optimise it in order to achieve the ideal balance between these factors. Subsequently, develop an operational policy which triages biopsy based on suspected maligncancy of lesion (classification).

## YOLOv8
### Algorithm Description
YOLO (You Only Look Once) models are a family of object detection models. They combine bounding box prediction and class labelling within a single pass (rather than traditional two-stage approach employed by other model types). In other words, YOLO models identify regions of interest (bounding boxes) for where an object is present, and then classify that specific region of interest (class labelling) within a single pass (https://roboflow.com/blog/guide-to-yolo-models). 

### Advantages over traditional CNNs
General YOLO models significantly improved runtimes and thus efficiency by drfting from the traditional CNN multi-stage detection methods to the aforementioned single-pass methodology. While earlier models did initially struggle with maintaing accuracy of their predictions, the various iterations the family has gone through (all the way up to YOLOv26 as of Oct 2026) has acted to rectify this issue, with models such as YOLOv8 able to maintain accuracy while also greatly improving runtimes (https://arxiv.org/pdf/2208.00773).

### Structure of YOLOv8
In general, "every Ultralytics YOLO model is built from three stages: a backbone that extracts features, a neck that fuses them across scales, and a head that predicts boxes and classes" (https://docs.ultralytics.com/guides/yolo-architecture). The fundamental unit for these models is the Conv block (2D convolution, bacth normalization, SiLU activation). YOLOv8 is simply a version of YOLO. Its core structure can be found here https://docs.ultralytics.com/guides/yolo-architecture, and a diagram is included within the repository (developed by GitHub user RangeKing).

Evolving from YOLOv5, major architectural differences between YOLOv8 and its predecessors include an achor-free detection head, which "simplifies the training process by removing the need to manually configure anchor boxes [per dataset]", a C2f module, which "improves gradient flow and allowed the model to learn richer feature representations without a significant increase in computational cost" and a decoupled head structure, which seperates classification and regression and has been shown to improve convergence speed and accuracy (https://docs.ultralytics.com/compare/yolov8-vs-yolov5). 

## Optimising the precision-recall frontier
General YOLO models significantly improved runtimes and thus efficiency by drfting from the traditional CNN multi-stage detection methods to the aforementioned single-pass methodology. The CNN network has gone through various iterations, ranging all the way up to YOLOv26 as of writing (Oct 2026). 


# My Own Research (Ignore)
Firtly, I had some questions that needed answering before beginning to develop a solution.
### What do I actually need to do?
The overarching purpose of the project revolves around detecting lesions. However, within the project description, it is stated that "an operational triage system must detect, localize with bounding boxes, and classify skin lesions simulataneously." The core Engineering Dilemmna subseqeuntly revoles around balancing clinical sensitivty vs biopsy overload, whereby over-detection of false alarms (non-skin lesions) can lead to overloading of clinics however under-detection of lesions may result in malignancies being missed. As such, the proposed model must balance this "precision-recall frontier."

Some other key notes I noticed:
- The task does not state you need to CLASSIFY lesions. Rather, it seems the lesion is simply needed to be DETECTED. HOWEVER, the recommended engineering approach states that an operating policy should be developed which triages biopsies based on whether lesion is detected as malignant or not. Clarification will need to be made in regards to this, however for now will assume classification is required.


In summary, the core task is this:
***Develop a model based on YOLOv8 which detects lesions within the provided ISIC2017/2018 dataset. Analyse how this model balances precision and recall, and in turn optimise it in order to achieve the ideal balance between these factors. Subsequently, develop an operational policy which triages biopsy based on suspected maligncancy of lesion (classification).***

## YOLOv8
### What is YOLOv8? What is it made out of?
YOLO (You Only Look Once) models are a family of object detection models. They combine bounding box prediction and class labelling within a single pass (rather than traditional two-stage approach employed by other model types). In other words, YOLO models identify regions of interest (bounding boxes) for where an object is present, and then classify that specific region of interest (class labelling) within a single pass (https://roboflow.com/blog/guide-to-yolo-models).

In general, "every Ultralytics YOLO model is built from three stages: a backbone that extracts features, a neck that fuses them across scales, and a head that predicts boxes and classes" (https://docs.ultralytics.com/guides/yolo-architecture). The fundamental unit for these models is the Conv block (2D convolution, bacth normalization, SiLU activation)

YOLOv8 is simply a version of YOLO. Its core structure can be found here https://docs.ultralytics.com/guides/yolo-architecture.



### What things are necessary for it to function?
#### Data Configuration
In terms of uploaded images/datasets, YOLOv8 requires 3 key things (https://roboflow.com/formats/yolov8-pytorch-txt).

*Directory Structure:* Dataset must contain parallel folders for images and labels, split into train, val (validation) and optionally test directories (ie. each of train, val and test are directories, and within each of these is a folder for images and another folder for associated labels (excluding test - only has images, not labels (?))). 

*Annotation File Format (.txt):* Each image must be represented in a specific form, in this case an associated txt file. The file exists whereby each line within it represents a bounding box. Each line is in the format class_id center_x center_y width height. Coordinates are normalized from zero to one. To convert to normalized xywh from pixel values, divide x (and width) by the image's width and divide y (and height) by the image's height.
--> In our case, the segmentation masks as part of ISIC2018 are binary mask images (Object = 255, Background = 0). As such we must normalize these into polygon coordinates - do this in Python using OpenCV (cv2) library to find object contours.

*Configuration File (data.yaml):* This file contains configuration values used by the model to locate images and map class names to class_ids. It exists in the form:
#Paths to dataset root or specific split text files
eg. train: datatset/train/images
#Number of classes
eg. nc: 2
#Class names list (index 0, 1, ...)
eg. names:['cat', 'dog']

NOTE: there is another .yaml file called MODELNAME.yaml, which contains the actual neural network architecture for the associated YOLO model. I believe this can be altered to try and optimise the performance for your specific applications (eg. with a custom block) - will need to investigate if this is necessary. 

#### How do you train the model?
The process to train a YOLOv8 model can be found here (https://yolov8.com/). Firstly, you pip install *ultralytics*. Once this is done, the website contains a specific command to train the model. For this, it requires a dataset which is *YOLOv8-formatted*

### What is the ISIC 2018 dataset? What does it contain?


