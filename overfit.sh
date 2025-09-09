#!/bin/bash
#SBATCH --job-name=overfit_batch_all_models
#SBATCH --output=logs/overfit_batch_all_models_%j.log
#SBATCH --error=logs/overfit_batch_all_models_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --constraint=h100
#SBATCH --account=ywa@h100
#SBATCH --hint=nomultithread
#SBATCH --partition=gpu_p6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malorenaudin1@gmail.com
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --signal=SIGUSR1@90

# Jean Zay module setup (adapt as needed)
module purge
module load pytorch-gpu/py3/2.0.1

# Activate your environment (adapt to your setup)
source /path/to/your/env/bin/activate

# Move to your repo directory
cd /path/to/Hard-CBR-RNN

# Make logs directory if not exists
mkdir -p logs

echo "Running overfit_one_batch_all_models.py on Jean Zay"

python cbr_lightning/overfit_one_batch_all_models.py