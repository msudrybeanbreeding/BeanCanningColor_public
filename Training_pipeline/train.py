# import packages
import os
import shutil
from ultralytics import YOLO

# delete runs directory if exists
if os.path.exists("./runs"):
    shutil.rmtree("./runs")

# initialize model
model = YOLO('yolov8n.pt')

# train model
model.train(
    data='./dataset/data.yaml',  
    epochs=100,                 
    imgsz=640,                
    batch=16,                   
    name='yolov8n-custom',
    patience=10,               
)