# Canned-Beans-Training-Pipeline

## Requirements
**Create virtual environment**
```bash
python -m venv venv
```
**Activate virtual environment**
```bash
source venv/bin/activate
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

## Prerequisite: Prepare the Dataset
Before running `train.py`, make sure the dataset is properly prepared and structured.

The dataset should follow the YOLO format and include:

- **Images** for training and validation
- **Label files** in YOLO format (`.txt`)
- A **`data.yaml`** file that specifies paths, number of classes, and class names

Training will not start unless the dataset is correctly set up. Place all files in an accessible directory (e.g., `./dataset/`) and ensure paths in `data.yaml` are correct.

## Instructions for Running the Training Pipeline
**Directly run the python script**
```python
python train.py
```

<h3 align="center">OR</h3>

**Submit a job to HPCC**
```bash
sbatch train_job_script.sh
```

**Note:** Before submitting the job to HPCC, make the appropriate changes to `train_job_script.sh`.

## Training Results
After training the YOLO model using the Python script, all metrics (e.g., precision, recall, mAP) and the best model weights are saved in the `runs` directory.

**Path to Best Model Weights:** `./runs/detect/yolov8n-custom/weights/best.pt`