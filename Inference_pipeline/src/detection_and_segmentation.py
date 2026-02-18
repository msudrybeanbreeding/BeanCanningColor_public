from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import torch
import cv2
import os


def detection_and_segmentation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_path = []
    root_dirs = [
        "./input/images",
    ]
    valid_exts = ('.jpg', '.jpeg', '.png')
    for root_dir in root_dirs:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    img_path.append(os.path.join(root, file))

    yolo_model = YOLO('./models/yolov8n_best.pt')
    # sam = sam_model_registry["vit_b"](checkpoint="./models/sam_vit_b_01ec64.pth")
    # sam = sam_model_registry["vit_l"](checkpoint="./models/sam_vit_l_0b3195.pth")
    sam = sam_model_registry["vit_h"](checkpoint="./models/sam_vit_h_4b8939.pth")
    sam.to(device)
    predictor = SamPredictor(sam)

    for img in tqdm(img_path, desc="⏳ Detecting and Segmenting Beans"):
        write_path1 = img.replace("input/images", "output/images")
        write_path2 = img.replace("input/images", "output/detection")
        write_path3 = img.replace("input/images", "output/segmentation")
        image = cv2.imread(img)
        cv2.imwrite(write_path1, image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = yolo_model(image_rgb, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() 

        image_with_prompt = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_with_prompt, (x1, y1), (x2, y2), (0, 0, 255), 10)

            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2

            points = np.array([
                [(x1 + xc) / 2, (y1 + yc) / 2],  
                [(xc + x2) / 2, (y1 + yc) / 2],  
                [(x1 + xc) / 2, (yc + y2) / 2],  
                [(xc + x2) / 2, (yc + y2) / 2],  
                [xc, yc]                         
            ])

            for pt in points:
                cv2.circle(image_with_prompt, (int(pt[0]), int(pt[1])), 15, (0, 255, 0), -1)  
        cv2.imwrite(write_path2, image_with_prompt)

        predictor.set_image(image_rgb)
        for box in boxes:
            x1, y1, x2, y2 = box
            input_box = np.array(box)

            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2

            points = np.array([
                [(x1 + xc) / 2, (y1 + yc) / 2],  
                [(xc + x2) / 2, (y1 + yc) / 2],  
                [(x1 + xc) / 2, (yc + y2) / 2],  
                [(xc + x2) / 2, (yc + y2) / 2],
                [xc, yc]  
            ])

            labels = np.array([1, 1, 1, 1, 1])  

            masks, scores, logits = predictor.predict(
                box=input_box[None, :],
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )

            best_mask = masks[np.argmax(scores)]
            image[best_mask] = [0, 255, 0]
        cv2.imwrite(write_path3, image)