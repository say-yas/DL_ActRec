#!/bin/bash

#SBATCH --partition=class
#SBATCH --job-name=har_withdevice 
#SBATCH --output=%x_%j.out 
#SBATCH --error=%x_%j.err 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=sharareh.sayyad@wsu.edu 
#SBATCH --time=7-00:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:tesla:1

module load anaconda3  
conda init bash 
conda activate har

time srun python /home/sharareh.sayyad/HAR/deeplearning_activity_recognition/MLP/mlp_wisdm_nodevice.py /home/sharareh.sayyad/HAR/deeplearning_activity_recognition/ 20 7 4 10 mlp2

