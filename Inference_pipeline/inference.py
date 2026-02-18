# import packages
import os
import shutil
import time

from src.detection_and_segmentation import detection_and_segmentation
from src.extract_and_calibrate_lab import extract_and_calibrate_lab
from src.calculate_d_score import calculate_d_score


def main():
    # step 1
    if os.path.exists("./output"):
        shutil.rmtree("./output")

    os.makedirs("./output/images")
    os.makedirs("./output/detection")
    os.makedirs("./output/segmentation")
    print("✅ Cleared output Directory")

    # step 2
    start = time.time()
    detection_and_segmentation()
    end = time.time()
    duration = end - start
    print(f"✅ Completed Beans Detection and Segmentation in {int(duration // 60)}m {int(duration % 60)}s")
    
    # step 3
    start = time.time()
    extract_and_calibrate_lab()
    end = time.time()
    duration = end - start
    print(f"✅ Completed LAB Extraction and Calibration in {int(duration // 60)}m {int(duration % 60)}s")

    # step 4
    calculate_d_score()
    print("✅ Completed D Score Calculation")


if __name__=="__main__":
    main()