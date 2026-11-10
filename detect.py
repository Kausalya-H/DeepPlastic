from ultralytics import YOLO

# Load our trained model
model = YOLO(
    "/Users/kausalya/runs/detect/outputs/deepplastic_model/weights/best.pt"
)

# Test images
source = "dataset/test/images"

# Run detection
results = model.predict(
    source=source,
    conf=0.40,
    imgsz=640,
    device="mps",
    save=True
)

print("\nDetection completed successfully!")
print("Images processed:", len(results))
