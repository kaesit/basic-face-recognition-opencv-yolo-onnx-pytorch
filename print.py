from ultralytics import YOLO
m = YOLO("yolov10n.onnx")
print("names attr:", getattr(m, "names", None))
# veya
try:
    print("m.model.names:", m.model.names)
except Exception:
    pass