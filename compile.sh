#!/bin/bash
#
#SBATCH --partition=students
#SBATCH --output="cuda_compile_log.txt"
#SBATCH --gpus=1

srun nvcc KMEANS_cuda_new.cu -arch=sm_75 -lm -o KMEANS_cuda_new.out