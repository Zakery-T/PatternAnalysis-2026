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
NO_DET = -1  # marker for "no detection"

def get_image_paths(split_dir):
    return sorted(os.path.join(split_dir, i)
                  for i in os.listdir(split_dir)
                  if i.lower().endswith(IMG_EXTS)
                  )

def read_true_class(label_path):
    """Returns the class id (first token) from given YOLO label file."""
    with open(label_path) as i:
        return int(i.readline().split()[0])

def read_true_box(label_path):
    """Returns bounding box coordinates from a given YOLO label file - for IoU"""
    with open(label_path) as i:
        parts = i.readline().split()
    return tuple(float(x) for x in parts[1:5])

def predict_image(model_current, img_path):
    results = model_current.predict(img_path, conf = CONF_THRESHOLD, verbose = False)[0] # [0] as single image
    detections = [] # clas_id, conf score, box coordinates

    for i in results.boxes: # boxes = holes all detections for one image
        class_id = int(i.cls.item()) # predicted class id
        conf = float(i.conf.item()) # confidence score
        box = tuple(i.xywhn[0].tolist())  # normalised (cx, cy, w, h)
        detections.append((class_id, conf, box))
    return detections

def select_prediction(detections):
    """Returns the highest-confidence detection"""
    if not detections:
        return None # ie. no lesion type detected
    return max(detections, key = lambda d: d[1])

def class_scores(detections):
    """Returns max confidence score per class boxes"""
    scores = [0.0] * len(CLASS_NAMES) # initalisation
    for class_id, conf, _ in detections:
        scores[class_id] = max(scores[class_id], conf)
    return scores

def run_predictions(model_current, img_dir, label_dir):
    """Running predictions for every image ie. predicting, selecting and storing true class,
    predicted class and scores --> incorporating the above-defined functions/functionality"""
    image_paths = get_image_paths(img_dir)
    results = []
    no_detections = 0 # tracking files with no lesions detected within them
    for i in image_paths:
        stem = os.path.splitext(os.path.basename(i))[0]
        label_path = os.path.join(label_dir, stem + ".txt")
        true_class = read_true_class(label_path)
        detections = predict_image(model_current, i)

        prediction = select_prediction(detections)
        if prediction is None:
            pred_class = NO_DET
            no_detections += 1
        else:
            pred_class = prediction [0]

        scores = class_scores (detections)
        results.append((true_class, pred_class, scores))
    return results, no_detections































def run_predictions(model, image_paths, label_dir):
    """Predicts every image and stores true/predicted classes, scores and IoU."""
    true, pred, scores, ious = [], [], [], []
    no_det = 0
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, stem + ".txt")

        detections = predict_image(model, img_path)
        best = select_prediction(detections)

        true.append(read_true_class(label_path))
        scores.append(class_scores(detections))
        if best is None:
            no_det += 1
            pred.append(NO_DET)
            ious.append(0.0)
        else:
            pred.append(best[0])
            ious.append(box_iou(best[2], read_true_box(label_path)))

    print(f"No detection: {no_det}/{len(image_paths)}")
    return {"true": np.array(true), "pred": np.array(pred),
            "scores": np.array(scores), "ious": np.array(ious), "no_det": no_det}


def compute_confusion_matrix(true, pred):
    """Rows = true class, columns = predicted class plus a final 'no detection' column."""
    n = len(CLASS_NAMES)
    matrix = np.zeros((n, n + 1), dtype=int)
    for t, p in zip(true, pred):
        matrix[t, n if p == NO_DET else p] += 1
    return matrix


def compute_class_metrics(matrix):
    """Per-class precision, recall and F1 from the confusion matrix."""
    n = len(CLASS_NAMES)
    metrics = {}
    for c in range(n):
        tp = matrix[c, c]
        fp = matrix[:, c].sum() - tp
        fn = matrix[c, :].sum() - tp  # includes no-detection cases
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[CLASS_NAMES[c]] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def compute_melanoma_metrics(true, pred, scores):
    """Melanoma vs rest: sensitivity, specificity and AUC."""
    y_true = (true == 0)
    y_pred = (pred == 0)
    tp = np.sum(y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    auc = roc_auc_score(y_true, scores[:, 0])
    return sensitivity, specificity, auc


def box_iou(box_a, box_b):
    """IoU of two normalised (cx, cy, w, h) boxes."""
    def corners(b):
        cx, cy, w, h = b
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    ax1, ay1, ax2, ay2 = corners(box_a)
    bx1, by1, bx2, by2 = corners(box_b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return inter / union if union > 0 else 0.0


def save_visualisations(model, image_paths, out_dir, n):
    """Saves up to n images with predicted boxes and labels drawn on."""
    os.makedirs(out_dir, exist_ok=True)
    for img_path in image_paths[:n]:
        result = model.predict(img_path, conf=0.25, verbose=False)[0]  # display threshold
        cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), result.plot())

                

def main():
    load_trained(weights_path) # loading of trained model


if __name__ == "__main__":
    main()




