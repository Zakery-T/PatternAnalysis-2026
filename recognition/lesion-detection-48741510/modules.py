from ultralytics import YOLO

# loading of model
def load_model(weights = "yolov8n.pt"):
    """
    Loading of YOLOv8 model from ulralytics (in partciular yolov8n)

    Args:
        Weights (simply using pre-determined model)

    Returns:
        YOLOv8 model pre-trained on COCO dataset
    """
    model = YOLO(weights)
    return model

