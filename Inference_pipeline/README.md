# Canned-Bean-Color-Detection-Pipeline

## Requirements
### 1. Install Git to Download Source Code
**(a) Windows Users:**
* Download **Git for Windows** from the [official site](https://git-scm.com/downloads/win).
* Run the installer and complete the installation.

**(a) macOS Users:**
```bash
brew install git
```
**(a) Ubuntu Users:**
```bash
sudo apt install -y git
```
**(b) Download the Source Code:**
```bash
git clone https://github.com/msudrybeanbreeding/msudrybeanbreeding.git
```

### 2. Install Python
**Windows Users:**
* Download **Python** from the [official site](https://www.python.org/downloads/windows/).
* Run the installer and complete the installation.

**macOS Users:**
```bash
brew install python
```
**Ubuntu Users:**
```bash
sudo apt install -y python3 python3-pip
```

### 3. Download the pretrained SAM models into the [`models`](./models/) directory
**Windows Users:**
- [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

**macOS / Ubuntu / Linux Users:**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 4. Set up the environment
**Windows Users:**

Create virtual environment
```bash
python -m venv venv
```
Activate virtual environment
```bash
venv\Scripts\activate.bat 
```
Install dependencies
```bash
pip install -r requirements.txt
```

**macOS / Ubuntu / Linux Users:**

Create virtual environment
```bash
python -m venv venv
```
Activate virtual environment
```bash
source venv/bin/activate
```
Install dependencies
```bash
pip install -r requirements.txt
```

## Inference
**1. Upload images folder, color checker file and class file**
* Upload all input images to the [`input/images`](./input/images) directory.
* Upload the color checker image to the [`input`](./input) folder and name it exactly **color_checker**.
* Upload the [`class.txt`](./input/class.txt) file, with the class name on the first line, to the [`input`](./input) folder.

**Important guidelines for the color checker**
* The image must be in horizontal orientation (i.e., width > height).
* Ensure the edges of the color checker are clearly visible and aligned.
* A sample color checker image is shown below for reference.
<img src="assets/color_checker.JPG" alt="Color Checker" width="375">

**2. Run code**
```python
python inference.py
```
<h3 align="center">OR</h3>

**2. Run code**
* Open the [`inference_job_script.sh`](inference_job_script.sh) file.
* Update the paths in the file accordingly.
* Replace the email address with your own (using your NetID) in the #SBATCH --mail-user field.
* The pipeline might take approximately 30 minutes to process 1000 images, so update the #SBATCH --time value accordingly to ensure the job runs long enough.

```bash
sbatch inference_job_script.sh
```

## Results
Once the pipeline completes successfully, the results will be available in the [`output`](output) folder:
- [`output/detection/`](./output/detection/): Contains images with detected beans. Detected beans is highlighted with a red bounding box.
- [`output/segmentation/`](./output/segmentation/): Contains images with segmented beans, highlighted in green to indicate the segmentation mask.
- [`output/patches/`](./output/patches/): Contains the 24 extracted patches used to calibrate the LAB values.
- [`output/results.csv`](./output/results.csv): Contains Date, Time, D scores for all the images, along with LAB values before and after color calibration, and Notes with Min Distance and Max Distance updates.

## Debugging Tips
* Check the [`job_logs/error.err`](./job_logs/error.err) file in the job_logs directory for the reason behind the job failure.
* Check the [`job_logs/output.out`](./job_logs/output.out) file in the job_logs directory to identify the step where the issue occurred.
* Review the detection, segmentation, and patches outputs to determine the root cause of the issue.