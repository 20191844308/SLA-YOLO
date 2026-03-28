from ultralytics import YOLO

# Load a model

model = YOLO("runs/train/Fanet/augVisdrone-s-FRM1_nosppf/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val(imgsz=640, batch=32, device=[0], project='runs/val/Fanet/', name='augVisdrone-s-FRM1_nosppf')  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category