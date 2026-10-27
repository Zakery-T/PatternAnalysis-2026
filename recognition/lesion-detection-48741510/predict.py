"""
predict.py: demo usage of the trained YOLO model on ISIC 2017 images.
Prints per-detection results and saves annotated images.
For test-set metrics (IoU, accuracy, mAP), use evaluate.py instead.

Usage:
    python predict.py
    python predict.py --source path/to/image_or_folder --weights path/to/best.pt
"""
import argparse
import os

from modules import load_trained
# from config import WEIGHTS, OUT_DIR, CLASS_NAMES, CONF_DEMO

# defaults (edit these once for your machine)
DEFAULT_WEIGHTS = "C:/Users/zakts/CODE/PatternAnalysis-2026/workspace/baseline_YOLOv8n-4/weights/best.pt"
DEFAULT_SOURCE = "C:/Users/zakts/CODE/PatternAnalysis-2026/workspace/yolo/test/images/ISIC_0015963.jpg"
DEFAULT_OUT = "runs/predict"

# constants
CLASS_NAMES = ["melanoma", "seborrheic keratosis", "nevus"]
IMGSZ = 640   # must match the imgsz used in train.py
CONF = 0.25   # display threshold


def demo_predict(model, source, device, out_dir):
    """Predicts on an image or folder, prints each detection and saves annotated images."""
    results = model.predict(source=source,
                            imgsz=IMGSZ,
                            conf=CONF,
                            device=device,
                            save=True,
                            show=False,
                            project=out_dir,
                            name="lesion_predictions",
                            exist_ok=True)

    for r in results:
        print("\nImage:", r.path)
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                class_name = CLASS_NAMES[int(box.cls.item())]
                conf = float(box.conf.item())
                xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
                print(f"  class: {class_name}")
                print(f"  confidence: {conf:.2f}")
                print(f"  box (x1, y1, x2, y2): {xyxy}")
        else:
            print("  no lesions detected")

    print(f"\nDone predicting. Annotated images saved under {os.path.join(out_dir, 'lesion_predictions')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="image file or folder of images")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="path to best.pt")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output folder")
    parser.add_argument("--device", default="cpu", help="'cpu' or GPU index, e.g. 0")
    args = parser.parse_args()
    device = int(args.device) if args.device.isdigit() else args.device

    model = load_trained(args.weights)
    demo_predict(model, args.source, device, args.out)


if __name__ == "__main__":
    main()
