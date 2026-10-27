"""
evaluate.py: evaluates the trained YOLO model on the ISIC 2017 test split.
Reports localisation IoU against the task target (>= 0.8), YOLO detection
metrics (mAP), per-class classification metrics, and melanoma-vs-rest
sensitivity/specificity/AUC.

Usage:
    python evaluate.py --weights path/to/best.pt --data path/to/yolo --out eval_out
"""
import os
import argparse
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score  # for reporting of values

from modules import load_trained

# defaults (edit these once for your machine)
DEFAULT_WEIGHTS = "C:/Users/zakts/CODE/PatternAnalysis-2026/workspace/baseline_YOLOv8n-4/weights/best.pt"
DEFAULT_DATA = "C:/Users/zakts/CODE/PatternAnalysis-2026/workspace/yolo"
DEFAULT_YAML = "C:/Users/zakts/CODE/PatternAnalysis-2026/recognition/lesion-detection-48741510/data.yaml"
DEFAULT_OUT = "eval_out"

# constants
CLASS_NAMES = ["melanoma", "seborrheic keratosis", "nevus"]
CONF_THRESHOLD = 0.001  # low, so scores are kept for AUC
IMGSZ = 640             # must match train.py
IOU_TARGET = 0.8        # task target: test IoU >= 0.8
IMG_EXTS = (".jpg", ".jpeg", ".png")  # extensions/tags
NO_DET = -1             # "no detection" case


def get_image_paths(split_dir):
    """Sorted list of image paths in a split folder."""
    return sorted(os.path.join(split_dir, i)
                  for i in os.listdir(split_dir)
                  if i.lower().endswith(IMG_EXTS))

def read_true_class(label_path):
    """Returns the class id (first token) from given YOLO label file."""
    with open(label_path) as f:
        return int(f.readline().split()[0])

def read_true_box(label_path):
    """Returns bounding box coordinates from a given YOLO label file - for IoU."""
    with open(label_path) as f:
        parts = f.readline().split()
    return tuple(float(x) for x in parts[1:5])

def predict_image(model_current, img_path, device):
    """Returns all detections for one image as (class_id, conf, (cx, cy, w, h))."""
    results = model_current.predict(img_path, imgsz=IMGSZ, conf=CONF_THRESHOLD,
                                    device=device, verbose=False)[0]  # [0] as single image
    detections = []  # class_id, conf score, box coordinates
    for b in results.boxes:  # boxes = holds all detections for one image
        class_id = int(b.cls.item())
        conf = float(b.conf.item())
        box = tuple(b.xywhn[0].tolist())  # normalised (cx, cy, w, h)
        detections.append((class_id, conf, box))
    return detections

def select_prediction(detections):
    """Returns the highest-confidence detection."""
    if not detections:
        return None  # ie. no lesion detected
    return max(detections, key=lambda d: d[1])

def class_scores(detections):
    """Returns max confidence score per class."""
    scores = [0.0] * len(CLASS_NAMES)  # initialisation
    for class_id, conf, _ in detections:
        scores[class_id] = max(scores[class_id], conf)
    return scores

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

def run_predictions(model_current, img_dir, label_dir, device):
    """Predicts every image; stores true class, predicted class, scores and IoU."""
    image_paths = get_image_paths(img_dir)
    results = []
    no_detections = 0  # tracking files with no lesions detected within them
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, stem + ".txt")
        true_class = read_true_class(label_path)
        detections = predict_image(model_current, img_path, device)

        prediction = select_prediction(detections)
        if prediction is None:
            pred_class = NO_DET
            iou = 0.0  # miss counts as zero overlap
            no_detections += 1
        else:
            pred_class = prediction[0]
            iou = box_iou(prediction[2], read_true_box(label_path))

        scores = class_scores(detections)
        results.append((true_class, pred_class, scores, iou))
    return results, no_detections


# NOTE: written with help of Claude Sonnet 5.0 (Medium)
def compute_iou_metrics(ious, pred):
    """Localisation summary against the IoU target."""
    detected = pred != NO_DET
    mean_detected = ious[detected].mean() if detected.any() else 0.0
    return {
        "mean_all": ious.mean(),            # misses count as 0
        "mean_detected": mean_detected,     # detected images only
        "share_gt_0.5": np.mean(ious > 0.5),
        "share_at_target": np.mean(ious >= IOU_TARGET),
        "n_detected": int(detected.sum()),
    }

def compute_map(model, data_yaml, device, out_dir):
    """YOLO's own detection metrics on the test split (Ultralytics validation).
    Returns overall and per-class mAP50, mAP50-95, P, R, plus AP at IoU 0.8."""
    m = model.val(data=data_yaml, split="test", imgsz=IMGSZ, device=device,
                  workers=0, project=out_dir, name="yolo_val", exist_ok=True,
                  verbose=False)
    box = m.box
    per_class = {}
    for i, c in enumerate(box.ap_class_index):
        per_class[CLASS_NAMES[c]] = {"P": box.p[i], "R": box.r[i],
                                     "mAP50": box.ap50[i], "mAP50-95": box.ap[i]}
    return {
        "map50": box.map50, "map": box.map, "p": box.mp, "r": box.mr,
        "ap_at_0.8": box.all_ap[:, 6].mean(),  # column 6 = IoU threshold 0.80
        "per_class": per_class,
    }

def compute_confusion_matrix(true, pred):
    """Rows = true class, columns = predicted class plus a final 'no detection' column."""
    n = len(CLASS_NAMES)  # number of columns
    matrix = np.zeros((n, n + 1), dtype=int)
    for t, p in zip(true, pred):
        matrix[t, n if p == NO_DET else p] += 1
    return matrix

def compute_class_metrics(matrix):
    """Per-class precision, recall and F1 from the confusion matrix."""
    metrics = {}
    for c in range(len(CLASS_NAMES)):
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

def save_visualisations(model, image_paths, out_dir, device):
    """Saves images with predicted boxes and labels drawn on."""
    os.makedirs(out_dir, exist_ok=True)
    for img_path in image_paths:
        result = model.predict(img_path, imgsz=IMGSZ, conf=0.25,  # display threshold
                               device=device, verbose=False)[0]
        cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), result.plot())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="path to best.pt")
    parser.add_argument("--data", default=DEFAULT_DATA, help="dataset folder from dataset.py")
    parser.add_argument("--yaml", default=DEFAULT_YAML, help="data.yaml path")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output folder")
    parser.add_argument("--device", default="cpu", help="'cpu' or GPU index, e.g. 0")
    parser.add_argument("--no-map", action="store_true", help="skip the YOLO mAP step")
    args = parser.parse_args()
    device = int(args.device) if args.device.isdigit() else args.device

    img_dir = os.path.join(args.data, "test", "images")  # TEST split
    label_dir = os.path.join(args.data, "test", "labels")
    data_yaml = args.yaml or os.path.join(args.data, "data.yaml")

    model = load_trained(args.weights)  # loading of trained model
    results, no_det = run_predictions(model, img_dir, label_dir, device)
    print(f"No detection: {no_det}/{len(results)}")

    true, pred, scores, ious = zip(*results)
    true, pred = np.array(true), np.array(pred)
    scores, ious = np.array(scores), np.array(ious)

    # --- localisation: task target ---
    iou = compute_iou_metrics(ious, pred)
    print("\n=== Localisation (top-confidence box vs mask-derived box) ===")
    print(f"Mean IoU, all images (misses = 0): {iou['mean_all']:.3f}")
    print(f"Mean IoU, detected only ({iou['n_detected']} images): {iou['mean_detected']:.3f}")
    print(f"Images with IoU > 0.5: {iou['share_gt_0.5']:.1%}")
    print(f"Images with IoU >= {IOU_TARGET}: {iou['share_at_target']:.1%}")
    print(f"Target (mean IoU >= {IOU_TARGET}) met: {iou['mean_all'] >= IOU_TARGET}")

    # --- detection metrics (Ultralytics) ---
    if not args.no_map:
        det = compute_map(model, data_yaml, device, args.out)
        print("\n=== Detection metrics (Ultralytics val, test split) ===")
        print(f"Overall: mAP50 {det['map50']:.3f}, mAP50-95 {det['map']:.3f}, "
              f"P {det['p']:.3f}, R {det['r']:.3f}")
        print(f"AP at IoU 0.8 (mean over classes): {det['ap_at_0.8']:.3f}")
        for name, m in det["per_class"].items():
            print(f"{name}: P {m['P']:.3f}, R {m['R']:.3f}, "
                  f"mAP50 {m['mAP50']:.3f}, mAP50-95 {m['mAP50-95']:.3f}")

    # --- classification ---
    matrix = compute_confusion_matrix(true, pred)
    print("\n=== Classification (top-confidence box per image) ===")
    print("Confusion matrix (rows = true, last column = no detection):\n", matrix)
    for name, m in compute_class_metrics(matrix).items():
        print(f"{name}: precision {m['precision']:.3f}, recall {m['recall']:.3f}, F1 {m['f1']:.3f}")
    print(f"Overall accuracy: {np.mean(true == pred):.3f}")

    sens, spec, auc = compute_melanoma_metrics(true, pred, scores)
    print(f"Melanoma vs rest: sensitivity {sens:.3f}, specificity {spec:.3f}, AUC {auc:.3f}")

    # --- example images: up to 10 failures and 10 correct ---
    image_paths = get_image_paths(img_dir)  # same order as run_predictions
    wrong = [p for p, t, q in zip(image_paths, true, pred) if t != q][:10]
    right = [p for p, t, q in zip(image_paths, true, pred) if t == q][:10]
    save_visualisations(model, wrong + right, os.path.join(args.out, "examples"), device)
    print(f"\nExamples saved to {args.out}/examples")


if __name__ == "__main__":
    main()
