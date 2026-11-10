from ultralytics import YOLO

print("Starting DeepPlastic training...")


# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")


# Train the model
results = model.train(
    data="dataset/data.yaml",
    epochs=5,
    imgsz=640,
    batch=8,
    device="mps",
    workers=2,
    project="outputs",
    name="deepplastic_model"
)


print("\nTraining completed successfully!")
print("Model saved inside:")
print("outputs/deepplastic_model/weights/")
