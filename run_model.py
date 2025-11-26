from ultralytics import YOLO
import glob

model = YOLO("yolov8m.pt")  

for img in glob.glob("*.jpg"):
    results = model.predict(img, save=True)
    results[0].show()