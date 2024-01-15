#!/bin/bash -l
#SBATCH -J aipaca_test_0
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --output=aipaca_test_0.txt
#SBATCH -c 7                # Cores assigned to each tasks
#SBATCH -G 1
#SBATCH --time=0-48:00:00
#SBATCH -p gpu
#SBATCH --mail-user oscar.castro@uni.lu
#SBATCH --mail-type BEGIN,END,FAIL

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"
# Python 3.X by default (also on system)
module load lib/UCX/1.9.0-GCCcore-10.2.0-CUDA-11.1.1

conda activate tcc

#python run_benchmark.py
python large_generate_data_cnn2d.py