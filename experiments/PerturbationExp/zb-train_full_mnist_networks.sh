#!/bin/bash




#SBATCH -A task0_pmml
#SBATCH -t 3-10 
#SBATCH -N 1



#SBATCH -n 1
#SBATCH -J full_mnist
#SBATCH -o out2.txt
#SBATCH -e err2.txt



#SBATCH -p dl_shared
#SBATCH --gres=gpu:1




#cd to directory you want

cd ~/experiments

#load python environment
module load python/miniconda3.9
#load cuda
module load cuda/10.2.89
#to initialize conda environment: 
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh



conda activate my_env2

python train_full_MNIST_networks.py

