"""
File for loading and processing of ISIC2018 data. Important due to variable size and resolution of images within dataset.
"""
import os
import cv2 # industry-standard for working with visual images
import random # for shufflinf of stems
import numpy as np
import shutil

# IN LINE WITH RESEARCH, 3 CORE SECTIONS TO PREPARE DATASE
# SECTION 1: SETTING UP PARALLEL DIRECTORIES FOR TRAIN, VAL, TEST

# creating filepaths using os
dataset = "/home/groups/comp3710/ISIC2018" # filepath in Rangpur
my_dir = "/home/Student/s4874151/01final"
output = os.path.join (my_dir, "yolo") # output directory

src_imgs = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2" # training images directory
src_masks ="/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2" # corresponding segmentation masks directory

# splits - ISIC2018 does not provide their own splits thus must make our own
ratios = {"train": 0.8, "val": 0.1, "test":0.1}
random.seed (42) # for reproducibility

# pairing of masks with corresponding images
stems = []
for f in os.listdir (src_masks): # listdir returns string of image names
    if not f.endswith("_segmentation.png"): 
        continue
    stem = f.replace("_segmentation.png","")
    if os.path.exists(os.path.join(src_imgs, stem + ".jpg")): # if image for this mask exists
            stems.append(stem)
    else:
        print(f"No image found for mask {f}") # error handling

# applying the splits
stems.sort() # descending order
random.shuffle(stems) # now random
n_train = int(len(stems) * ratios["train"]) # number of training images
n_val = int(len(stems) * ratios["val"]) # number of validation images
assignment = {}

# assigning every stem to a certain folder - as stems are already randomised, can simply go up to number req.
for i, s in enumerate(stems):
    assignment[s] = "train" if i < n_train else "val" if i < n_train + n_val else "test"


MIN_AREA = 50  # ignore specks smaller than this many pixels
counts = {"train": 0, "val": 0, "test": 0}

for stem, split in assignment.items():
    images_dir = os.path.join(output, split, "images")
    labels_dir = os.path.join(output, split, "labels") # adding to relevant directories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True) 


# SECTION 2: CONVERTING TO ANNOTATION FILE (.txt)
    img_path = os.path.join(src_imgs, stem + ".jpg")
    mask = cv2.imread(os.path.join(src_masks, stem + "_segmentation.png"), cv2.IMREAD_GRAYSCALE) # grayscale
    img = cv2.imread(img_path)

    if img is None or mask is None:
        print(f"Skipped {stem}; could not read image/mask") # error handling
        continue

    # error handling - ensuring mask and image sizes agree
    h, w = img.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # force the mask to strictly 0 or 255
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # trace the outlines of the white regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # drop tiny specks
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]

    # one box around everything that's left
    annotations = []
    if contours: # if exists
        bx, by, bw, bh = cv2.boundingRect(np.vstack(contours))

        # normalise to YOLO format (class x_center y_center width height)
        x_center = (bx + bw / 2) / w
        y_center = (by + bh / 2) / h
        annotations.append(f"0 {x_center:.6f} {y_center:.6f} {bw / w:.6f} {bh / h:.6f}")

    # write the label file (empty file = no lesion, valid in YOLO)
    with open(os.path.join(labels_dir, stem + ".txt"), "w") as f:
        f.write("\n".join(annotations))

    #  copy the image across (copy, not move)
    shutil.copy2(img_path, os.path.join(images_dir, stem + ".jpg"))
    counts[split] += 1

print("Done:", counts)





















