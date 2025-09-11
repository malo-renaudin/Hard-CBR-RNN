#!/bin/bash
#SBATCH --job-name=test_simple_cbr_SGD
#SBATCH --output=logs/test_simple_cbr_%j.log
#SBATCH --error=logs/test_simple_cbr_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
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


# Move to your repo directory
cd /lustre/fswork/projects/rech/ywa/uds37kc/Hard-CBR-RNN

# Make logs directory if not exists
mkdir -p logs


python test_simple_cbr.py