from ultralytics import YOLO

# Load the YOLOv8n model pre-trained on COCO
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(
    data="crash_test_data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="crash_test_yolo",
    project="runs/train",
    pretrained=True,
    val=True,  # Validate during training
)
