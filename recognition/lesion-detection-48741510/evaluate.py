# NOTE: Helped written by Claude Sonnet 5.0 (Medium)

"""
evaluate.py: evaluates the trained YOLO model on the ISIC 2017 test split.
Reports localisation IoU against the task target (>= 0.8), YOLO detection
metrics (mAP), per-class classification metrics, and melanoma-vs-rest
sensitivity/specificity/AUC.

Usage:
    python evaluate.py --weights path/to/best.pt --data path/to/yolo --out eval_out
"""
import os # standard Python library
import argparse # standard Python library
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score  # for reporting of values
from pathlib import Path

from modules import load_trained
# configurations
from config import WEIGHTS, DATA_DIR, OUT_DIR, CLASS_NAMES, CONF_EVAL, IOU_TARGET, IMG_EXTS, NO_DET, IMGSZ, write_data_yaml


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
    results = model_current.predict(img_path, imgsz=IMGSZ, conf=CONF_EVAL,
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

def scale_box(box, sw, sh):
    """Scales a normalised (cx, cy, w, h) box about its centre, clipped to the image."""
    cx, cy, w, h = box
    w, h = w * sw, h * sh
    x1, x2 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
    y1, y2 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def fit_scale(pred_boxes, true_boxes):
    """Grid-searches width/height factors that maximise mean IoU (misses count as 0).
    Run on VAL only, then pass the result to the test run with --scale."""
    def mean_iou(sw, sh):
        return np.mean([0.0 if p is None else box_iou(scale_box(p, sw, sh), t)
                        for p, t in zip(pred_boxes, true_boxes)])
    grid = np.arange(0.90, 1.301, 0.02)
    best = max((mean_iou(sw, sh), sw, sh) for sw in grid for sh in grid)
    print("\n=== Box scale fit (use these on test via --scale) ===")
    print(f"No scaling: mean IoU {mean_iou(1.0, 1.0):.3f}")
    print(f"Best: width x{best[1]:.2f}, height x{best[2]:.2f}, mean IoU {best[0]:.3f}")
    print(f"Run on test with: --scale {best[1]:.2f} {best[2]:.2f}")

def run_predictions(model_current, img_dir, label_dir, device):
    """Predicts every image; stores true class, predicted class, scores and IoU."""
    image_paths = get_image_paths(img_dir)
    results = []
    no_detections = 0  # tracking files with no lesions detected within them
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, stem + ".txt")
        true_class = read_true_class(label_path)
        true_box = read_true_box(label_path)  # read once, reused below
        detections = predict_image(model_current, img_path, device)  # keep your current argument list

        prediction = select_prediction(detections)  # or select_prediction(detections)
        if prediction is None:
            pred_class, pred_box = NO_DET, None
            iou, best_iou = 0.0, 0.0
            no_detections += 1
        else:
            pred_class, pred_box = prediction[0], prediction[2]
            iou = box_iou(pred_box, true_box)
            best_iou = max(box_iou(d[2], true_box) for d in detections)

        scores = class_scores(detections)
        results.append((true_class, pred_class, scores, iou, best_iou, pred_box, true_box))
    return results, no_detections

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

def compute_map(model, data_yaml, device, out_dir, split):
    """YOLO's own detection metrics on the test split (Ultralytics validation).
    Returns overall and per-class mAP50, mAP50-95, P, R, plus AP at IoU 0.8."""
    m = model.val(data=data_yaml, split=split, imgsz=IMGSZ, device=device,
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

def save_worst_overlays(image_paths, ious, pred_boxes, true_boxes, out_dir, n=20):
    """Step 1: saves the n lowest-IoU test images with the true box (green) and predicted box (red)."""
    os.makedirs(out_dir, exist_ok=True)
    for rank, idx in enumerate(np.argsort(ious)[:n]):
        img = cv2.imread(image_paths[idx])
        scale = 900 / max(img.shape[:2])          # shrink very large images
        img = cv2.resize(img, None, fx=scale, fy=scale)
        h, w = img.shape[:2]

        def draw(box, colour):
            cx, cy, bw, bh = box
            p1 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h))
            p2 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
            cv2.rectangle(img, p1, p2, colour, 3)

        draw(true_boxes[idx], (0, 255, 0))        # true (mask-derived): green
        if pred_boxes[idx] is not None:
            draw(pred_boxes[idx], (0, 0, 255))    # predicted: red
        cv2.putText(img, f"IoU {ious[idx]:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        name = os.path.basename(image_paths[idx])
        cv2.imwrite(os.path.join(out_dir, f"{rank:02d}_{name}"), img)


def bias_report(pred_boxes, true_boxes):
    """Step 2: signed prediction-minus-truth errors. A mean far from 0 suggests systematic bias;
    a mean near 0 with a large spread suggests random error (a regression limit)."""
    pairs = [(p, t) for p, t in zip(pred_boxes, true_boxes) if p is not None]
    p = np.array([x[0] for x in pairs])
    t = np.array([x[1] for x in pairs])
    print("\n=== Box bias (predicted minus true, normalised units) ===")
    for i, name in enumerate(["centre x", "centre y", "width", "height"]):
        d = p[:, i] - t[:, i]
        print(f"{name}: mean {d.mean():+.4f}, std {d.std():.4f}, mean abs {np.abs(d).mean():.4f}")
    ratio = (p[:, 2] * p[:, 3]) / (t[:, 2] * t[:, 3])
    print(f"Predicted/true box area: median {np.median(ratio):.3f}, "
          f"larger than true in {np.mean(ratio > 1):.1%} of images")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS, help="path to best.pt")
    parser.add_argument("--data", default=DATA_DIR, help="dataset folder from dataset.py")
    parser.add_argument("--out", default=OUT_DIR, help="output folder")
    parser.add_argument("--device", default="cpu", help="'cpu' or GPU index, e.g. 0")
    parser.add_argument("--no-map", action="store_true", help="skip the YOLO mAP step")
    parser.add_argument("--diagnose", action="store_true", help="save worst-IoU overlays and print box bias")
    parser.add_argument("--split", default="test", help="split to evaluate: val or test")
    parser.add_argument("--scale", nargs=2, type=float, default=[1.0, 1.0],
                        metavar=("W", "H"), help="width/height factor for predicted boxes")
    parser.add_argument("--fit-scale", action="store_true", help="fit scale factors (use on val)")
    args = parser.parse_args()
    device = int(args.device) if args.device.isdigit() else args.device

    # img_dir = os.path.join(args.data, "test", "images")  # TEST split
    # label_dir = os.path.join(args.data, "test", "labels")
    img_dir = os.path.join(args.data, args.split, "images")
    label_dir = os.path.join(args.data, args.split, "labels")
    data_yaml = write_data_yaml(args.data, Path(args.out) / "data_eval.yaml")

    model = load_trained(args.weights)  # loading of trained model
    results, no_det = run_predictions(model, img_dir, label_dir, device)
    print(f"No detection: {no_det}/{len(results)}")

    true, pred, scores, ious, best_ious, pred_boxes, true_boxes = zip(*results)
    true, pred = np.array(true), np.array(pred)
    scores, ious, best_ious = np.array(scores), np.array(ious), np.array(best_ious)
    if args.fit_scale:
        fit_scale(pred_boxes, true_boxes)
    sw, sh = args.scale
    if (sw, sh) != (1.0, 1.0):
        pred_boxes = tuple(None if p is None else scale_box(p, sw, sh) for p in pred_boxes)
        ious = np.array([0.0 if p is None else box_iou(p, t)
                         for p, t in zip(pred_boxes, true_boxes)])
        print(f"\nApplied box scale: width x{sw:.2f}, height x{sh:.2f}")

    # --- localisation: task target ---
    iou = compute_iou_metrics(ious, pred)
    print("\n=== Localisation (top-confidence box vs mask-derived box) ===")
    print(f"Mean IoU, all images (misses = 0): {iou['mean_all']:.3f}")
    print(f"Mean IoU, detected only ({iou['n_detected']} images): {iou['mean_detected']:.3f}")
    print(f"Images with IoU > 0.5: {iou['share_gt_0.5']:.1%}")
    print(f"Images with IoU >= {IOU_TARGET}: {iou['share_at_target']:.1%}")
    print(f"Target (mean IoU >= {IOU_TARGET}) met: {iou['mean_all'] >= IOU_TARGET}")
    print(f"Mean IoU, best detection per image: {best_ious.mean():.3f}")
    print(f"Split evaluated: {args.split}")
    # --- detection metrics (Ultralytics) ---
    if not args.no_map:
        det = compute_map(model, data_yaml, device, args.out, args.split)
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

    image_paths = get_image_paths(img_dir)
    if args.diagnose:
        bias_report(pred_boxes, true_boxes)
        save_worst_overlays(image_paths, ious, pred_boxes, true_boxes,
                            os.path.join(args.out, "worst_iou"))
        print(f"Worst-IoU overlays saved to {os.path.join(args.out, 'worst_iou')}")
    # --- example images: up to 10 failures and 10 correct ---
    # image_paths = get_image_paths(img_dir)  # same order as run_predictions
    wrong = [p for p, t, q in zip(image_paths, true, pred) if t != q][:10]
    right = [p for p, t, q in zip(image_paths, true, pred) if t == q][:10]
    save_visualisations(model, wrong + right, os.path.join(args.out, "examples"), device)
    print(f"\nExamples saved to {args.out}/examples")


if __name__ == "__main__":
    main()
