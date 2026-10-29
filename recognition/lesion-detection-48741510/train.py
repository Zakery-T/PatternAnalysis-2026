#  following process from https://docs.ultralytics.com/usage/python
from modules import load_initial_model, load_trained
from config import DATA_DIR, BASE_MODEL, IMGSZ, EPOCHS, BATCH, WORKERS, SEED

def main():
    model = load_initial_model() # loads pre-trainied YOLOv8

    # training of model
    results = model.train(# documentation at https://docs.ultralytics.com/modes/train
        data = str(DATA / "data.yaml"),
        epochs = EPOCHS,
        imgsz = IMGSZ,
        batch = BATCH,
        device = 0, # CUDA GPU selection - ?no multi-GPU runs so simply use 0
        workers = WORKERS, # Rangpur recommendation - avoids slowdown/freeze
        seed = SEED, #  for fair comparison
        project = "runs", # output directory
        name = "imgsz1024_baseline" # subfolder in output directory
    )

    best = results.save_dir / "weights" / "best.pt"
    
    trained = load_trained(best)  # loading the BEST trained model
    val_metrics = trained.val(data = DATA, split = "val", imgsz = IMGSZ, workers = WORKERS)
    print(f"Val: mAP50 {val_metrics.box.map50:.3f}, mAP50-95 {val_metrics.box.map:.3f}, "
          f"P {val_metrics.box.mp:.3f}, R {val_metrics.box.mr:.3f}")
    

if __name__ == "__main__":
    main()

