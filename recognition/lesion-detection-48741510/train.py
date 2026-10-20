from modules import load_initial_model, load_trained
#  following process from https://docs.ultralytics.com/usage/python
def main():
    model = load_initial_model() # loads pre-trainied YOLOv8

    # training of model
    results = model.train( # documentation at https://docs.ultralytics.com/modes/train
        data = "/home/Student/s4874151/01final/data.yaml",
        epochs = 50,
        imgsz = 640,
        batch = 16,
        device = 0, # CUDA GPU selection - ?no multi-GPU runs so simply use 0
        project = "runs", # output directory
        name = "baseline_YOLOv8n" # subfolder in output directory
        )

    best = results.save_dir / "weights" / "best.pt"
    
    trained = load_trained(best) # loading the BEST trained dataset
    val_metrics  = trained.val(split = "val")
    test_metrics = trained.val(split = "test")
    print (val_metrics, test_metrics)

if __name__ == "__main__":
    main()
