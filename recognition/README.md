# Pattern Recognition 2026 Project: Lesion Detection and Classification using YOLOv8
---
Name: Zakery Tsin  
Student ID: s48741510

## Key Project Information
Dataset: ISIC2017 \
Model: YOLOv8 (pre-trained models available via ultralytics) \
Description: Developing a model using YOLOv8 which detects and classifies lesions within the ISIC 2017 dataset. Analyse how this model balances precision and recall, and in turn optimise it to achieve the ideal balance between these factors. Subsequently, develop an operational policy which triages biopsy based on suspected maligncancy of lesion (classification).

## YOLOv8
### Algorithm Description
YOLO (You Only Look Once) models are a family of object detection models. They combine bounding box prediction and class labelling within a single pass (rather than traditional two-stage approach employed by other model types). In other words, YOLO models identify regions of interest (bounding boxes) for where an object is present, and then classify that specific region of interest (class labelling) within a single pass (https://roboflow.com/blog/guide-to-yolo-models). General YOLO models significantly improved runtimes and thus efficiency by drifting from the traditional CNN multi-stage detection methods to the aforementioned single-pass methodology. While earlier models did initially struggle with maintaininng accuracy of their predictions, the various iterations the family has gone through (all the way up to YOLOv26 as of Oct 2026) has acted to rectify this issue, with models such as YOLOv8 able to maintain accuracy while also greatly improving runtimes (https://arxiv.org/pdf/2208.00773).

### Structure
In general, "every Ultralytics YOLO model is built from three stages: a backbone that extracts features, a neck that fuses them across scales, and a head that predicts boxes and classes" (https://docs.ultralytics.com/guides/yolo-architecture). The fundamental unit for these models is the Conv block (2D convolution, bacth normalization, SiLU activation). YOLOv8 is simply a version of YOLO. Its core structure can be found here https://docs.ultralytics.com/guides/yolo-architecture, and a diagram is included within the repository (developed by GitHub user RangeKing)<img src="lesion-detection-48741510/Supporting/YOLOv8 Architecture Diagram (RangeKing).jpg" width="600">. 

Evolving from YOLOv5, major architectural differences between YOLOv8 and its predecessors include an achor-free detection head, which "simplifies the training process by removing the need to manually configure anchor boxes [per dataset]", a C2f module, which "improves gradient flow and allowed the model to learn richer feature representations without a significant increase in computational cost" and a decoupled head structure, which seperates classification and regression and has been shown to improve convergence speed and accuracy (https://docs.ultralytics.com/compare/yolov8-vs-yolov5). 

# Feasability Report
*Please note that major changes have occurred since this feasability report, in particular the use of ISIC2018 to ISIC2017. I went to tutors for help with this via EdBoard, however they were unable to provide me sufficient information to get ISIC2018 up and running. As such, made the change to ISIC2017 due to its inclusion in Rangpur already.*

The feasability report completed for this project is attached via this [link](<lesion-detection-48741510/SupportingImages/Feasability Report.pdf>).

# Dependencies
Dependencies required for this project are included within [requirements.txt](lesion-detection-48741510/requirement.txt) Note that *ultralytics* is built on top of PyTorch, and thus installing this package will install the necessary modules for this functionality.

# Examples
Example outputs stem from the two files [predict.py](lesion-detection-48741510/predict.py) and [evaluate.py](lesion-detection-48741510/evaluate.py). Examples of these outputs are shown below.

**predict.py (run on ISIC_0015963.jpg)**
![alt text](lesion-detection-48741510/outputs/predict/lesion_predictions/ISIC_0015963.jpg)

**evaluate.py (validation batch 01)**
![alt text](lesion-detection-48741510/Supporting/val_batch1_pred.jpg)

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
## Quantitative Benchmarking
The baseline model utilised to test YOLOv8 performance against was YOLOv3, in particular "yolov5-tinyu.pt" weights as included within the Ultralytics package. For the implemented model, a pre-trained YOLOv8 model (pre-trained on COCO dataset)
The final models configuration that I used for this report can be found in config.py. Its results can also be found in results.txt, all of which are based on the ISIC2017 test split (600 images). IoU is the top-confidence predicted box against the mask-derived box. 


As outlined above, the target mean IoU of > 0.8 was unable to be met. This was despite extensive experimentation on the YOLOv8 configuration, which are as outlined below.


The baseline model implemented was YOLOv8n with weights trained on the COCO dataset. This was loaded into the codebase via the ultralytics package and *load_initial_model* as part of *modules.py*. This utilized the default ultralytics settings. In comparison, the mode utilized within this repository posseessed almost the exact same architecure apart from image size, whereby the initial mode possessed 640 and the chosen model possessed 1024.

Final performance results are outlined below.![alt text](../runs/imgsz1024_baseline/results.png)

The target IoU of 0.8 was unable to be achieved despite numerous attempts to rectify this.
| # | Modification | Type | Change tested | Test mean IoU | IoU ≥ 0.8 | Outcome |
|---|---|---|---|---|---|---|
| 0 | Baseline YOLOv8 | Training | Default config, 640 px, 50 epochs | 0.750 | 51.7% | Reference |
| 1 | Larger image size | Training | 640 → 1024 px | 0.755 | 52.5% | Marginal IoU gain (+0.005). Clear classification gain (accuracy 0.718 → 0.757, melanoma sensitivity 0.368 → 0.547) |
| 2 | Longer training | Training | Increased epochs | No gain | n/a | No improvement |
| 3 | Box merging | Post-processing | Confidence-weighted average of boxes overlapping the top box | 0.758 | 53.3% | No meaningful change (within run-to-run variation) |
| 4 | Oracle best-box diagnostic | Diagnostic | Best candidate box per image, chosen using ground truth | 0.803 | 58.3% | Upper bound only, not an achievable result. Shows ≤ 0.05 headroom from box selection |
| 5 | Box shape diagnosis | Diagnostic | Bias report (predicted minus true) | n/a | n/a | Boxes well centred, but median predicted/true area ratio 0.84. Width and height errors are large and bidirectional |
| 6 | Filtered mask labels | Dataset | Ignore mask fragments < 15% of the largest region | Unchanged | Unchanged | 58 of 2750 images had multiple regions, 177 fragments ignored. No measurable effect |
| 7 | Box scale calibration | Post-processing | Width and height ×1.02, fitted on val | 0.771 | 57.8% | +0.016 gain, but still below 0.8. Not retained in the final code |
| 8 | Consensus box selection | Post-processing | Confidence-weighted consensus box, `min_conf` = 0.1 (chosen on val) | 0.755 | 52.5% | Val improved (0.784 → 0.797) but test did not (0.755 → 0.755). Likely val noise |
| 9 | Larger model | Training | About 3.5× parameters (11.1M vs 3.1M), 1024 px, default augmentation | 0.755 | 53.0% | IoU unchanged. Classification worsened (accuracy 0.693, melanoma sensitivity 0.444, mAP50 0.505). Model size and augmentation changed together, so effects are confounded |

**Notes**
- Rows 3, 7 and 8 are inference-time post-processing, not model improvements. Their settings were chosen on val and applied once to test.
- Rows 4 and 5 are diagnostics, not attempted fixes.
- Differences of about 0.005 fall within the run-to-run variation seen across repeated evaluations of the same weights (0.755 to 0.759), so treat them as equal.



## Computational Resource Analysis
| Resource | Value |
| --- | ---|
| Peak GPU RAM | 5.11G |
| Paramater Count | 3,133,193 | 
| Inference Runtime/Latency | Speed: 2.0ms preprocess, 77.4ms inference, 0.0ms loss, 0.5ms postprocess per image |

## Failure Analysis
Failure cases were drawn from the ISIC 2017 **test split** (600 images) using the final model with single-pass, top-confidence box selection. In each overlay, the green box is the ground-truth box and the **red box** is the model's prediction. Each overlay is labelled with the true class, predicted class, confidence and IoU.

Cases were chosen to cover the main error modes visible in the quantitative results. They are grouped into classification failures and localisation failures. Explanations of cause are **hypotheses based on visual inspection**, not tested conclusions. Note that the nevus class is defined by exclusion (neither melanoma nor seborrheic keratosis), so it represents either nevus or some other benign case (not classified). Test cases were sourced from outputs/examples.
### Part A: Classification Failures
#### Failure Case 1: Non-Malignancy Identified as Malignancy (SK --> Melanoma)
![alt text](<lesion-detection-48741510/outputs/10 true 10 false/00_ISIC_0012199.jpg>)
**Observation:** The lesion shows irregular borders and pigmentation noticeably darker than that of typical SK examples in the dataset.
**Hypothesis:** These features overlap with those associated with early-stage melanoma, which may explain the prediction.
**Implication:** Under the proposed operating policy this is a false alarm that would be routed to clinical review, increasing clinic burden but posing no patient risk.

#### Failure Case 2: SK predicted as nevus / other benign
![alt text](lesion-detection-48741510/outputs/examples/07_ISIC_0012272.jpg)
**Observation:** The lesion appears to display features typical of well-developed/severe SK. In the example images inspected, similar severities of SK presentations were uncommon, with most ground-truth SK images appearing milder.
**Hypothesis:** If this holds across the dataset, the model may have seen few advanced SK presentations in training. This could explain its poor classification performance.
**Implication:** Both classes are benign, so the clinical risk is low. The error would still misdirect follow-up.

#### Case 3: Missed melanoma, predicted as nevus / other benign
![alt text](lesion-detection-48741510/outputs/examples/10_ISIC_0012395.jpg)
**Observation:** The lesion has relatively smooth, well-defined borders, a feature more typical of nevus than of melanoma.
**Hypothesis:** Border regularity is only one of several dermoscopic cues, but the model may rely heavily on it. Class imbalance in ISIC2017 means that the model may also have limited exposure to atypical-looking melanomas.
**Implication:** This is the clinically most serious error type, since an automatically cleared melanoma delays diagnosis. It motivates a lowered melanoma decision threshold in the operating policy (see the Recommendations section).

### Part B: Localisation failures
#### Failure Case 4: Low IoU with with poor bounding box 
![alt text](lesion-detection-48741510/outputs/examples/14_ISIC_0012484.jpg)

**Observation:** Poor IoU, with models bounding box dramatically larger than ground truth. The surrounding skin appears flaky with lots of hair present, which may produce a texture that resembles a potential lesion boundary.
**Hypothesis:** Surface texture may have shifted the predicted extent, and the model may also have confused the lesion type (SK predicted in place of nevus).

#### Failure Case 5 Multi-part or fragmented lesion
![alt text](lesion-detection-48741510/outputs/test_final/worst_iou/02_ISIC_0015115.jpg)
*NOTE*: This is from worst-ious study into possible ways to improve mean IoU
**Observation:** Poor IoU, with model focussed on a single fragment of a much larger lesion. For additional context, as part of testing to try and improve Mean IoU across all classes, a set of 20 of the worst-case IoU images was produced. Of these 20, 6 contained multi-part lesions.
**Hypothesis:** Multi-part lesions harder to classify, and the model instead focussed on the fragment that was more pronounced (pigmented) and with more regular borders, which it has likely seen much more similar variations of when in training.

# Limitations of the Study
ISIC2017 does not ship with patient ID labels, and as such leakage was unable to be assessed despite best attempts made to try. Unless done externally, each image is treated as independent, even if it is the same picture of a lesion. The ISIC2017 dataset also has a major class imbalance, with images weighted mainly to Nevus, with much less images attributed to SK or Melanoma types. This is a further exacerbated by the major variation between SK and Melanoma cases, and resulted in the model performing worse on these test cases, as the reduced training images available meant the model struggled to generalize. Augmentation of the dataset itself rather than further modifications of the YOLOv8 model may yield better results. 












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


