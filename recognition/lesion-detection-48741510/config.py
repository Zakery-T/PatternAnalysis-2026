"""
Due to the big use of CONSTANTS and PATHS, this file acts as an anchor for all other files for this project.
It contains all of the CONSTANTS utilised throughout, ensuring the code is robust and easy to modify.

Paths are relevant to this file, and paths I personally used for data, sorted data, weights and output are provided. 
If wanting to use different paths, simply replace paths with location of relevant files.

"""
import os
from pathlib import Path # standard Python library, for building paths from a core anchor

# ---------- paths (relative to this file, so they work on any machine) ----------
PROJECT_DIR = Path(__file__).resolve().parent # parent directory for project/download

# Creation of main directories using above-defined function/functionality
RAW_DIR = Path(PROJECT_DIR / "data" / "ISIC2017")  # raw downloaded ISIC2017 folders
DATA_DIR = Path(PROJECT_DIR / "data" / "yolo")    # dataset.py output - raw ISIC data now sorted into splits
WEIGHTS = Path(PROJECT_DIR / "weights" / "best.pt")
OUT_DIR = Path(PROJECT_DIR / "outputs")

# ISIC2017 raw data split names
SPLITS = ["train", "val", "test"]
_ISIC_NAMES = {"train": "Training", "val": "Validation", "test": "Test_v2"}

SPLIT_DIRS = {
    s: {
        "imgs":  RAW_DIR / f"ISIC-2017_{n}_Data",
        "masks": RAW_DIR / f"ISIC-2017_{n}_Part1_GroundTruth",
        "csv":   RAW_DIR / f"ISIC-2017_{n}_Part3_GroundTruth.csv",
    }
    for s, n in _ISIC_NAMES.items()
}

# Classes as part of ISIC2017 dataset
CLASS_NAMES = ["melanoma", "seborrheic keratosis", "nevus"]

# For dataset.py - allows for ignoring of smaller resolution images, treated instead as simply noise
\

# For modules.py and train.py - training parameters
BASE_MODEL = "yolov8n.pt"  
IMGSZ = 640                # training size; evaluate.py reads the size from the weights by default
EPOCHS = 50
BATCH = 16
WORKERS = 1                # Rangpur recommendation
SEED = 0                   # For repeatability

# For evaluation.py and predict.py
CONF_EVAL = 0.001   # low, so scores are kept for AUC
CONF_DEMO = 0.25    # display threshold for predict.py
IOU_TARGET = 0.8    # task target: test IoU >= 0.8
IMG_EXTS = (".jpg", ".jpeg", ".png") # image extensions, ensures other non-relevant files ignored
NO_DET = -1         # no (lesion) detection

# NOTE: recommended by Claude SOnnet 5.0 (Medium)
def write_data_yaml(path=None):
    """Writes data.yaml pointing at DATA_DIR, so it can never go stale - no manual definition required"""
    path = Path(path) if path else DATA_DIR / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    path.write_text(
        f"path: {DATA_DIR.as_posix()}\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n{names}\n")
    return path
