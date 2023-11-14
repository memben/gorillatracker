import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO


model_name = "yolov8n" #@param {type:"string"}
dataset_name = "gorilla.yaml" #@param {type:"string"}

# Initialize YOLO Model
model = YOLO(f"{model_name}.pt")

# Add W&B callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Train/fine-tune your model
# At the end of each epoch, predictions on validation batches are logged
# to a W&B table with insightful and interactive overlays for
# computer vision tasks
model.train(project="ultralytics", data=dataset_name, epochs=24, imgsz=640)
model.val()

# Finish the W&B run
wandb.finish()
