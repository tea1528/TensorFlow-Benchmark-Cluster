#!/bin/bash
#SBATCH --account=sc3260
#SBATCH --partition=maxwell
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=20:00:00
#SBATCH --output=mnist_1.out

# Set up environment
module load GCC Singularity git

# Clone tensorflow repo to run a tutorial
git clone https://github.com/tensorflow/models.git
singularity exec --nv docker://tensorflow/tensorflow:latest-gpu \
    python ./tutorials/image/mnist/convolutional.py