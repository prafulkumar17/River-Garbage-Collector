import cv2
import os

image_dir = "C:/Personal_1/Praful/sem6/TARP/TARP_ml/train/images"
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Corrupt image found: {img_path}")
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

model.train(
    data="C:/Personal_1/Praful/sem6/TARP/TARP_ml/data.yaml",
    epochs=20,
    imgsz=640,  
    batch=4,  # Increase batch size (try 4, 8, or higher if GPU allows)
    device="cuda",
    name = "YOLOv8s_FSL_100_v1",
    workers=0,  # Prevents excessive RAM usage
    # half=False, #disables fb16
    # amp=False, #Removes constaint of GPU usage
    # val = False # Disable validation to check if training works
    cache=False,  # Avoid caching large datasets
    half=False,   # Disable mixed precision if GPU is unstable
    val=False     # Disable validation temporarily
)
