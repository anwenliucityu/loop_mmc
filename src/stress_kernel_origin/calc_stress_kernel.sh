#!/usr/bin/env bash 
#SBATCH --job-name="stress_kernel"
#SBATCH --partition=xlong
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks=1
#SBATCH --mem=5G 
#SBATCH --exclude=gauss[1,6,16],gauss
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

cd /gauss12/home/cityu/anwenliu/loop_stress/loop_mmc/src 
python generate_stress_kernel_origin.py
