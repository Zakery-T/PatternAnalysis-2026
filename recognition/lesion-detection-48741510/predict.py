"""

"""
import os
import cv2 # industry-standard for working with visual images
import numpy as np


from modules import load_trained
# constants
CLASS_NAMES =["melanoma", "seborrheic keratosis", "nevus"] # UPDATING DEPENDNING ON ED BOARD RESPONSE
CONF_THRESHOLD = 0.001
IMG_EXTS = [".jpg", ".jpeg", ".png"] # ensures get_image_paths doesn't process random image types


def get_image_paths(split_dir):
    return sorted(os.path.join(split_dir, f)
                  for f in os.listdir(split_dir)
                  if f.lower().endswith(IMG_EXTS)
                  )

def read_true_class(label_path):
    """Returns the class id (first token) from given YOLO label file."""
    with open(label_path) as f:
        return int(f.readline().split()[0])

def read_true_box(label_path):
    """Returns bounding box coordinates from a given YOLO label file - for IoU"""
    with open(label_path) as f:
        parts = f.readline().split()
    return tuple(float(x) for x in parts[1:5])

def predict_image(model_current, img_path):
    results = model_current.predict(img_path, conf = CONF_THRESHOLD, verbose = False)[0] # [0] as single image
    detections = [] # clas_id, conf score, box coordinates

    for f in results.boxes: # boxes = holes all detections for one image
        class_id = int(f.cls.item()) # predicted class id
        conf = float(f.conf.item()) # confidence score
        box = tuple(f.xywhn[0].tolist())  # normalised (cx, cy, w, h)
        detections.append((class_id, conf, box))
    return detections

def select_prediction(detections):
    """Returns the highest-confidence detection"""
    if not detections:
        return None # ie. no lesion type detected
    return max(detections, key = lambda d: d[1])
                

def main():
    load_trained(weights_path) # loading of trained model


if __name__ == "__main__":
    main()




