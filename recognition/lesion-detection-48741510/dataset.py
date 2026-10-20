"""
File for loading and processing of ISIC2018 data. Important due to variable size and resolution of images within dataset.
"""
import os
import cv2 # industry-standard for working with visual images
import random # for shufflinf of stems
import numpy as np
import shutil
# import csv # for classification labels

# IN LINE WITH RESEARCH, 3 CORE SECTIONS TO PREPARE DATASET

# ================================================================
# SECTION 1: SETTING UP PARALLEL DIRECTORIES FOR TRAIN + VAL AND SORTING OF IMAGES

# creating filepaths using os
dataset = "/home/groups/comp3710/ISIC2018" # filepath in Rangpur
my_dir = "/home/Student/s4874151/01final"
output = os.path.join (my_dir, "yolo") # output directory
src_imgs = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2" # training images directory in Rangpur
src_masks ="/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2" # corresponding segmentation masks directory
# csv_path = os.path.join(my_dir, "ISIC2018_Task3_Training_GroundTruth.csv")

# other relevant configurations
CLASS_NAMES =["lesion"] # UPDATING DEPENDNING ON ED BOARD RESPONSE
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1} # splits - ISIC2018 does not provide their own splits thus must make our own
MIN_AREA = 50 # removes specks smaller than this resolution - NOTE: Claude Sonnet 5.0 (Medium)
SEED = 42 # for reproducibility


def get_stems(src_imgs, src_masks):
    """Retrieves the stem for each image and corresponding mask ie. its image ID."""
    stems = [] # the image id
    for f in os.listdir (src_masks): # listdir returns string of image names
        if not f.endswith("_segmentation.png"): # error handling
            continue
        stem = f.replace("_segmentation.png","") # removing segmentation label
        if os.path.exists(os.path.join(src_imgs, stem + ".jpg")): # if image for this mask exists - src_imgs/stem.jpg
                stems.append(stem)
        else:
            print(f"No image found for mask {f}") # error handling
    return sorted(stems) # ascending order

def split_stems(stems, ratios, seed = SEED):
    """Splits images and assigns them into train or validation sets."""
    stems = list(stems)
    random.Random(seed).shuffle(stems) # now random
    n_train = int(len(stems) * ratios["train"]) # number of training images
    n_val = int(len(stems) * ratios["val"]) # number of validation images

    assignment = {}
    # assigning every stem to a certain folder - as stems are already randomised, can simply go up to number req.
    for i, s in enumerate(stems):
        assignment[s] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return assignment

def make_dirs(output, ratios):
    """Creates the output/{split}/images and output/{split}/labels directories."""
    for split in ratios:
        os.makedirs(os.path.join(output, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output, split, "labels"), exist_ok=True)

# =================================================================
# SECTION 2: CONVERTING TO ANNOTATION FILE (.txt) - for EVERY image
def mask_to_box(mask, w, h, min_area = MIN_AREA):
    """Converts a given segmentation mask into an annotation box - following LINK"""
    if mask.shape != (h, w): # error handling - ensures dimensions now agree
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) # NOTE: Claude Sonnet 5.0 (Medium)

    # force the mask to strictly 0 or 255
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # trace the outlines of the white regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area] # dropping images with too small resolution
    if not contours: # if empty:
        return None
    bx, by, bw, bh = cv2.boundingRect(np.vstack(contours)) # one box around everything left
    return (bx + bw / 2) / w, (by + bh / 2) / h, bw / w, bh / h

def get_class(stem):
    """Class id for an image, single class for now waiting on EdBoard response"""
    return 0

def write_label(label_path, cls, box):
    """Writes one line for YOLO annotation.txt file: class x_center y_center width height."""
    xc, yc, bw, bh = box
    with open(label_path, "w") as f:
        f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def process_image(stem, split):
    """Converts one image/mask pair into a label file (annotation.txt) and copied image. Combining the above functions into one single call."""
    img_path = os.path.join(src_imgs, stem + ".jpg")
    mask_path = os.path.join(src_masks, stem + "_segmentation.png")
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None: # error handling
        print(f"Skipped {stem}; could not read image/mask")
        return False

    h, w = img.shape[:2]
    box = mask_to_box(mask, w, h) # creates annotation box using previously defined function
    if box is None:
        print(f"Skipped {stem}; empty mask")  # no label file, so YOLO never sees it as background
        return False

    write_label(os.path.join(output, split, "labels", stem + ".txt"), get_class(stem), box)
    shutil.copy2(img_path, os.path.join(output, split, "images", stem + ".jpg"))
    return True
# ================================================================
# MAIN
def build_dataset(assignment):
    """Processes every assigned image and returns per-split counts."""
    counts = {split: 0 for split in RATIOS}
    for stem, split in assignment.items():
        if process_image(stem, split):
            counts[split] += 1
    return counts

def main():
    stems = get_stems(src_imgs, src_masks) # getting the actual stems
    print(f"{len(stems)} image/mask pairs found") # NOTE: Claude Sonnet 5.0 (Medium)
    assignment = split_stems(stems, RATIOS) # producing the split
    make_dirs(output, RATIOS) # producing the directories
    counts = build_dataset(assignment) # counting 
    print("Done:", counts) # files now ready for use in YOLO


if __name__ == "__main__":
    main()


