#!/bin/bash
#SBATCH --job-name=cbr_training
#SBATCH --output=job_outputs/job_%A_%a.out
#SBATCH --error=job_outputs/job_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --constraint=h100
#SBATCH --account=ywa@h100
#SBATCH --hint=nomultithread
#SBATCH --partition=gpu_p6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malorenaudin1@gmail.com
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --signal=SIGUSR1@90

# Create output directory
mkdir -p job_outputs

# Run the training script with the config for this array job
python train_single_job.py $SLURM_ARRAY_TASK_ID