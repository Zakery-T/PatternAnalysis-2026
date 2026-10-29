# Pattern Recognition 2026 Project: Lesion Detection and Classification using YOLOv8
---
Name: Zakery Tsin
Student ID: s48741510

## Key Project Information
Dataset: ISIC2017
Model: YOLOv8 (pre-trained models available via ultralytics)
Description: Developing a model based on YOLOv8 which detects and classifies lesions within the ISIC 2017 dataset. Analyse how this model balances precision and recall, and in turn optimise it in order to achieve the ideal balance between these factors. Subsequently, develop an operational policy which triages biopsy based on suspected maligncancy of lesion (classification).

## YOLOv8
### Algorithm Description
YOLO (You Only Look Once) models are a family of object detection models. They combine bounding box prediction and class labelling within a single pass (rather than traditional two-stage approach employed by other model types). In other words, YOLO models identify regions of interest (bounding boxes) for where an object is present, and then classify that specific region of interest (class labelling) within a single pass (https://roboflow.com/blog/guide-to-yolo-models). 

General YOLO models significantly improved runtimes and thus efficiency by drifting from the traditional CNN multi-stage detection methods to the aforementioned single-pass methodology. While earlier models did initially struggle with maintaininng accuracy of their predictions, the various iterations the family has gone through (all the way up to YOLOv26 as of Oct 2026) has acted to rectify this issue, with models such as YOLOv8 able to maintain accuracy while also greatly improving runtimes (https://arxiv.org/pdf/2208.00773).

In general, "every Ultralytics YOLO model is built from three stages: a backbone that extracts features, a neck that fuses them across scales, and a head that predicts boxes and classes" (https://docs.ultralytics.com/guides/yolo-architecture). The fundamental unit for these models is the Conv block (2D convolution, bacth normalization, SiLU activation). YOLOv8 is simply a version of YOLO. Its core structure can be found here https://docs.ultralytics.com/guides/yolo-architecture, and a diagram is included within the repository (developed by GitHub user RangeKing)<img src="lesion-detection-48741510/SupportingImages/YOLOv8 Architecture Diagram (RangeKing).jpg" width="200">. Evolving from YOLOv5, major architectural differences between YOLOv8 and its predecessors include an achor-free detection head, which "simplifies the training process by removing the need to manually configure anchor boxes [per dataset]", a C2f module, which "improves gradient flow and allowed the model to learn richer feature representations without a significant increase in computational cost" and a decoupled head structure, which seperates classification and regression and has been shown to improve convergence speed and accuracy (https://docs.ultralytics.com/compare/yolov8-vs-yolov5). 

# Feasability Report
*Please note that major changes have occurred since this feasbility report, in particular the use of ISIC2018 to ISIC2017. I went to tutors for help with this via EdBoard, however they were unable to provide me sufficient information to get ISIC2018 up and running. As such, made the change to ISIC2017 due to its inclusion in Rangpur already.*

**Malignancy Identification in ISIC2018 using YOLOv8 Architecture**
INTENDED USER: Dermatologists, GPs conducting primary dermatological screening
MODEL PURPOSE: Classification of skin lesions for malignancy detection, balancing high clinical sensitivity versus false positive detection/s (The Engineering Dilemma).

MODEL ACCEPTANCE CRITERIA
CRITERIA	ACCEPTANCE	JUSTIFICATION
mAPval, 50-95	>40%	Due to high irregularity between lesions, and due to the implications of false negatives over false positives as spoken about within the open  ilemma, a >40% mAP (high performance) was deemed necessary. Furthermore, the COCO dataset, current YOLOv8 models achieve between 37.3% to 53.9% during validation, scaling with model size (Jocher & Derrenger, 2025). As ISIC datasets are a more difficult dataset to categorize due to greater fine detail required, and as computational limits will be highly relevant to this task, target was deemed as realistic. 
Runtime	>4ms	ISIC 2018 dataset contains 11,527 total images (ISIC, 2018). Limited computational resources available means relatively fast runtimes/image required in order to minimise overall model runtimes. Current YOLOv8 models speed on similar A100s take 0.99ms to 3.53ms/per image depending on model size. Once ISIC 2018 images rescaled to similar 640x640 size, runtime should closely mirror these. In practical applications runtime important due to likely high clinical workload. Thus, target deemed as realistic.
F1 Score	>0.80	F1 acts to balance both precision and recall (Jocher, et al., 2023), which is important due to open dilemma between managing false positives (clinical overload) vs false negatives (potentially fatal consequences). Other models trained on the ISIC 2018 dataset generally sit between 0.75 to 0.85 F1 score (Saeed, et al., 2025), and (Rana, et al., 2025) utilised YOLOv8 model to return an F1 score of 0.83, thus proposed F1 target deemed realistic.


MODEL SPECIFICATIONS
-	Model: YOLOv8 
o	Better performance/runtime over other CNNs. YOLOv8 > YOLOv7, YOLOv9 due to smaller inference, comparable mAP, CPU training (Jocher & Derrenger, 2025).
-	Dataset: ISIC 2018
o	Larger sample size, more feature types compared to past datasets (ISIC, 2018).
o	More variability and sample size = higher generalization = better performance on unseen data thus more robust for real world applications.
o	Reinforces importance of pre-processing data, efficient usage of limited GPU resources (Rangpur), associated GPU framework (eg. PyTorch).
-	Difficulty: Normal Difficulty

ABOUT ISIC2018 – Initial Data Audit
ISIC 2018 dataset consists of 11,527 total images, with a 10,015-1512 training-testing split (ISIC, 2018). It contains a total of 7 classes, of which two classes are malignant (Melanoma, Basal Cell Carcinoma). It must also be noted that one class is deemed as pre-cancerous (Actinic Keratosis), which has a small chance of progression into Squamous Cell Carcinoma. Particular images count for each class are provided, and ground truth data is provided for the training set images only (Cassidy, et al., 2022). It must be noted that ISIC 2018 images possess variable sizes (pixel x pixel) and resolutions, with corresponding ground truth segmentation masks matching the dimensions of the parent lesion image (ISIC, 2018) 

KEY TECHNICAL RISKS
-	Difficulties with pre-processing of data due to variable sizes, resolutions
-	Data leakage between splits  avoids overinflation of performance metrics
-	Data augmentation: important for classification of classes with lower prevalence within data set (ie. rarer lesion types – see source for specific image counts)  avoids overfitting
-	Limited GPU resources (Rangpur)

NEXT STEPS
-	Implement YOLOv8 for smaller, pre-standardized dataset eg. MNIST
-	See if ISIC2018 has pt. data IDs – if not, can split images however I want (no worries about data leakage(??))
-	Decide if binary (saying simply malignaant vs non-malignant) or 7 classes  using 7 classes might be better because don’t have to worry about the pre-cancerous types 10% leading to it
Bibliography
ISIC, 2018. ISIC Challenge Datasets. [Online] 
Available at: https://challenge.isic-archive.com/data/#2018
Jocher, G., 2025. YOLOv7 vs YOLOv8. [Online] 
Available at: https://docs.ultralytics.com/compare/yolov8-vs-yolov9
Jocher, G. & Derrenger, P., 2025. YOLOv8 vs YOLOv9. [Online] 
Available at: https://docs.ultralytics.com/compare/yolov8-vs-yolov9
Jocher, G. et al., 2023. Performance Metrics Deep Dive. [Online] 
Available at: https://docs.ultralytics.com/guides/yolo-performance-metrics
Rana, L., Modi, N. & Pandey, S., 2025. Explainable YOLO models for early skin cancer detection using hair artifact removal and VGG16 guided annotation. Discover Artifical Intelligence, 5 (358), pp. https://doi.org/10.1007/s44163-025-00637-7.
Saeed, M., Afify, Y., Badr, N. & Helal, N., 2025. Multimodal deep learning ensemble framework for skin cancer detection. Scientific Reports, pp. 10.1038/s41598-025-30534-z.

# Dependencies

# Example

# Data Pre-Procesing
## Pre-Processing Images for YOLO compatibility
In line with project requirements, a pretrained YOLOv8.n model was utilised via the ultralytics package. Dataset.py acts to process these images into a version compatible with YOLO models (in line with ultralytics documentation). Running this file produces three core outcomes which are necessary for YOLO models to run:
1. Sets up parallel directories for train (/imgs, /labels) and val (/imgs, /labels) splits
2. Converts masks to bounding boxes and subsequently converts images into associated annotation.txt files
3. (TO BE IMPLEMENTED) Automatically writes a configured data.yaml file**
**This was initially done manually, however changing this manually became cumbersome thus this was automated (via config.py file)


## Data Splits
Training, validation and testing splits were already provided as part of the ISIC2017 data. A link to this data and the steps required for its organisation will be linked in the reproducibility section

# Reproducibility
In order to reproduce the results obtained within this repository, follow the below framework. Note that ultralytics is natively built on top of PyTorch, and framework is based on the assumption Rangpur (UQ HPC) is being utilized.

1. Install dependencies using [pip install -r requirements.txt]
2. Download ISIC2017 dataset from https://challenge.isic-archive.com/data/#2017. Unzip folder into data/ISIC2017/
3. Run dataset.py to create YOLO compatible directories, annotation files (bounding boxes) and data.yaml files. Verify console output is [train:2000, val: 150, test: 600] pairs (ISIC2017 splits). Images stored in data/yolo/
4. Upload images (data/yolo), config.py, modules.py and train.py to Rangpur directory (within same directory). Run [python train.py] on GPU via Slurm job request (ensuring Rangpur directory has correct dependencies installed as well). 
- Outputs go to runs/detect/runs/[run name]. These include weights/best.pt (best model), args.yaml (settings utilised), results.csv, results.png and validation plots.
5. Copy weights/best.pt into project directory (local machine). 
6. Run evaluate.py to generate model performance metrics (mAP, mean IoU, Precision (P), Recall (R), etc.). 
7. Run predict.py for per-detection results and annotated images. Outputs will be saved under outputs/predict

Using the files as configured within the repository, the results obtained were Test Mean IoU approx 0.755, accuracy approx 0.757

# Investigation of the Open Research Dilemma















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


