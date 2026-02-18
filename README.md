Deep learning and computer vision for image-based high-throughput phenotyping of canning quality traits in dry beans
Authors: Lovepreet Singh, Om Sai Krishna Madhav Lella, Evan M. Wright, Karen Cichy and Valerio Hoyos-Villegas*
*Corresponding Authors: hoyosval@msu.edu


Training and Inference Pipelines:

This repository contains the complete training and inference pipelines for a custom deep learning and computer vision based detection, segmentation and downstream metric computation for canning color evaluation in dry beans.

The scientific application and experimental context are described in the associated publication. This repository focuses strictly on computational implementation, structure, and reproducibility.

```
Canning_segmentation/
│
├── Training-Pipeline/
│   ├── train.py
│   ├── train_job_script.sh
│   ├── requirements.txt
│   └── README.md
│
├── Canned-Beans-Inference-Pipeline/
│   ├── inference.py
│   ├── inference_job_script.sh
│   ├── requirements.txt
│   ├── assets/
│   ├── input/
│   ├── models/
│   ├── output/
│   ├── src/
│   └── README.md
│
└── README.md
```


Components
Training Pipeline

The training pipeline is responsible for:

Dataset loading

Model initialization

Hyperparameter configuration

Training and validation

Checkpoint generation

Logging

All training-related scripts and documentation are located inside:
Training-Pipeline/

Execution instructions and environment setup details are provided in:
Training-Pipeline/README.md


Inference Pipeline

The inference pipeline is responsible for:

Loading trained model weights

Running segmentation predictions

Post-processing model outputs

Feature extraction and metric computation

Structured result generation

All inference-related scripts and documentation are located inside:
Canned-Beans-Inference-Pipeline/

Execution instructions and environment setup details are provided in:
Canned-Beans-Inference-Pipeline/README.md

Expected Workflow

Train the model using the Training Pipeline

Export or copy the final trained weights into the models/ directory of the Inference Pipeline

Run inference on new data

Retrieve results from the output/ directory

Each pipeline is modular and can be executed independently.


Reproducibility

This repository provides:

Complete training implementation

Complete inference implementation

Explicit configuration and parameter definitions

Script-based execution (local or HPC environments)

Deterministic directory organization

All preprocessing steps, model configurations, and post-processing routines are defined in code.

Data Availability

The dataset used in the associated publication is not publicly released. Few examples images are provided in the inference pipeline.

Users may adapt the pipelines to their own datasets by modifying dataset paths and configuration parameters as described in the respective subdirectory documentation.
