#!/bin/bash
#SBATCH --account=sc3260
#SBATCH --reservation=sc3260
#SBATCH --partition=mic
#SBATCH --mem=10G
#SBATCH --time=20:00:00
#SBATCH --output=inception4_cpu.out

# Set up environment
module load GCC Singularity git

# Clone tensorflow repo to run a tutorial
# git clone https://github.com/tensorflow/models.git
singularity exec --nv docker://tensorflow/tensorflow:latest \
    python tf_cnn_benchmarks.py --batch_size=32 --model=inception4 --variable_update=parameter_server --device=cpu --data_format=NHWC
