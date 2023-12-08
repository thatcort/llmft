#!/bin/bash
#SBATCH -JGPUExample                                # Job name
#SBATCH -N1 --gres=gpu:V100:1 --ntasks-per-node=6   # Number of nodes, GPUs, and cores required
#SBATCH --mem-per-gpu=20G                           # Memory per gpu
#SBATCH -t15                                        # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                             # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=afischer39@gatech.edu            # e-mail address for notifications

conda activate myenv

srun python teacher.py gpu 1000  # Run test example