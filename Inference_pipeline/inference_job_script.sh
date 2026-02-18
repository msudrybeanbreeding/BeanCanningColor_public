#!/bin/bash --login
#SBATCH --job-name=inference             # Name of the job
#SBATCH --output=job_logs/output.out     # Output log file (%x=job name, %j=job ID)
#SBATCH --error=job_logs/error.err       # Error log file
#SBATCH --ntasks=1                       # Number of tasks (usually 1 for single job)
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --gpus=v100:1                    # Type and number of GPUs
#SBATCH --mem=16G                        # Total memory
#SBATCH --time=00:15:00                  # Time limit (hh:mm:ss)
#SBATCH --mail-type=ALL                  # Email notifications
#SBATCH --mail-user=lellaom@msu.edu      # Your email address

# Activate your virtual environment
source /mnt/home/lellaom/Canned-Bean-Color-Detection-Pipeline/venv/bin/activate

# Navigate to your project directory
cd /mnt/home/lellaom/Canned-Bean-Color-Detection-Pipeline

# Run your Python script
python inference.py