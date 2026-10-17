#  following process from https://docs.ultralytics.com/usage/python

# training of model
results = model.train( # documentation at https://docs.ultralytics.com/modes/train
    data = "data.yaml", 
    epochs = 50,
    imgsz = 640,
    batch = 16,
    device = 0, # CUDA GPU selection - ?no multi-GPU runs so simply use 0
    project = "runs", # output directory
    name = "baseline_YOLOv8n" # subfolder in output directory
    )

evaluation = model.val() # evaluating trained model on val set
print(evaluation) # printing out values
