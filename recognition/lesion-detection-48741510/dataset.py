"""
File for loading and processing of ISIC2017 data. Important due to variable size and resolution of images within dataset.

NOTE: on 22/10/2026, decided to switch from ISIC2018 --> ISIC2017 data
"""
import os
import cv2 # industry-standard for working with visual images
import numpy as np
import shutil
import csv # for classification labels

# IN LINE WITH RESEARCH, 3 CORE SECTIONS TO PREPARE DATASET

# ================================================================
# SECTION 1: SETTING UP PARALLEL DIRECTORIES FOR TRAIN + VAL AND SORTING OF IMAGES

# creating filepaths using os
my_dir = "C:/Users/zakts/CODE/comp3710_final" # my local directory
dataset = os.path.join(my_dir, "ISIC2017") # dataset location 
output = os.path.join (my_dir, "yolo") # output directory

# ISIC2017 comes with pre-determined splits (unlike ISIC2018)
SPLIT_DIRS = {
    "train": {
        "imgs":  os.path.join(dataset, "ISIC-2017_Training_Data"),
        "masks": os.path.join(dataset, "ISIC-2017_Training_Part1_GroundTruth"),
        "csv":   os.path.join(dataset, "ISIC-2017_Training_Part3_GroundTruth.csv"),
    },
    "val": {
        "imgs":  os.path.join(dataset, "ISIC-2017_Validation_Data"),
        "masks": os.path.join(dataset, "ISIC-2017_Validation_Part1_GroundTruth"),
        "csv":   os.path.join(dataset, "ISIC-2017_Validation_Part3_GroundTruth.csv"),
    },
    "test": {
        "imgs":  os.path.join(dataset, "ISIC-2017_Test_v2_Data"),
        "masks": os.path.join(dataset, "ISIC-2017_Test_v2_Part1_GroundTruth"),
        "csv":   os.path.join(dataset, "ISIC-2017_Test_v2_Part3_GroundTruth.csv"),
    },
}

# other relevant configurations
CLASS_NAMES =["melanoma", "seborrheic keratosis", "nevus"] # UPDATING DEPENDNING ON ED BOARD RESPONSE
SPLITS = list(SPLIT_DIRS.keys())
MIN_AREA = 50

def get_stems(src_imgs, src_masks):
    """Retrieves the stem for each image and corresponding mask ie. its image ID."""
    stems = [] # the image id
    for f in os.listdir (src_masks): # listdir returns string of image names
        if not f.endswith("_segmentation.png"): # error handling
            continue
        stem = f.replace("_segmentation.png","") # removing segmentation label\
        if os.path.exists(os.path.join(src_imgs, stem + ".jpg")): # if image for this mask exists - src_imgs/stem.jpg
                stems.append(stem)
        else:
            print(f"No image found for mask {f}") # error handling
    return sorted(stems) # ascending order

def make_dirs(output, splits):
    """Creates the output/{split}/images and output/{split}/labels directories."""
    for i in splits:
        os.makedirs(os.path.join(output, i, "images"), exist_ok=True)
        os.makedirs(os.path.join(output, i, "labels"), exist_ok=True)

def load_labels(csv_path):
    """Reads a Part 3 CSV and returns {image_id: class_id}, following CLASS_NAMES order."""
    labels = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if float(row["melanoma"]) == 1:
                labels[row["image_id"]] = 0  # melanoma
            elif float(row["seborrheic_keratosis"]) == 1:
                labels[row["image_id"]] = 1  # seborrheic keratosis
            else:
                labels[row["image_id"]] = 2  # nevus (both columns 0)
    return labels


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

def write_label(label_path, cls, box):
    """Writes one line for YOLO annotation.txt file: class x_center y_center width height."""
    xc, yc, bw, bh = box
    with open(label_path, "w") as f:
        f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def process_image(stem, split, labels):
    """Converts one image/mask pair into a label file (annotation.txt) and copied image. Combining the above functions into one single call."""
    src_imgs = SPLIT_DIRS[split]["imgs"]
    src_masks = SPLIT_DIRS[split]["masks"]
    img_path = os.path.join(src_imgs, stem + ".jpg")
    mask_path = os.path.join(src_masks, stem + "_segmentation.png")
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None: # error handling
        print(f"Skipped {stem}; could not read image/mask")
        return False
    if stem not in labels:
        print(f"Skipped {stem}; no class label in CSV")
        return False

    h, w = img.shape[:2]
    box = mask_to_box(mask, w, h) # creates annotation box using previously defined function
    if box is None:
        print(f"Skipped {stem}; empty mask")  # no label file, so YOLO never sees it as background
        return False

    write_label(os.path.join(output, split, "labels", stem + ".txt"), labels[stem], box)
    shutil.copy2(img_path, os.path.join(output, split, "images", stem + ".jpg"))
    return True
# ================================================================
# MAIN
def build_dataset():
    """Processes every assigned image and returns per-split counts."""
    counts = {}
    for split in SPLITS:
        paths = SPLIT_DIRS[split]
        labels = load_labels(paths["csv"])
        stems = get_stems(paths["imgs"], paths["masks"])
        print(f"{split}: {len(stems)} image/mask pairs found")
        counts[split] = sum(process_image(s, split, labels) for s in stems)
    return counts


def main():
    make_dirs(output, SPLITS)  # producing the directories
    counts = build_dataset()   # converting and counting
    print("Done:", counts)     # files now ready for use in YOLO

if __name__ == "__main__":
    main()


