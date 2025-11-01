from ultralytics import YOLO

# Load the pre-trained YOLOv10n model
model = YOLO("yolov10n.pt")
results = model("bus.jpg")
results[0].show()