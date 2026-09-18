from ultralytics import YOLO

# Load the trained model
model = YOLO(
    "/Users/kausalya/runs/detect/outputs/deepplastic_model/weights/best.pt"
)

# Evaluate the model on the test dataset
results = model.val(
    data="dataset/data.yaml",
    split="test",
    imgsz=640,
    batch=8,
    device="mps"
)

print("\nEvaluation completed!")

print("Precision:", results.box.mp)
print("mAP50:", results.box.map50)
print("mAP50-95:", results.box.map)
