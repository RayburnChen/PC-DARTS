#!/bin/bash
# Usage: sbatch slurm-gpu-job-script
# Prepared By: Ruibin Chen,  Apr 2021

# NOTE: To activate a SLURM option
# remove the whitespace between the '#' and 'SBATCH'

# Set your account id
#SBATCH --account=az20

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=7-00:00:00

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=pcdartsJob

# Request CPU resource for a serial job
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

# Request for GPU, 
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g

# SBATCH --gres=gpu:P100:1
# SBATCH --partition=m3h

# Memory usage (MB)
#SBATCH --mem-per-cpu=4000

# To receive an email when job completes or fails
#SBATCH --mail-user=rche0046@student.monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the file for output (stdout)
#SBATCH --output=PC-DARTS-EVA-JOB-%j.out

# Set the file for error log (stderr)
#SBATCH --error=PC-DARTS-EVA-JOB-%j.err


# Use reserved node to run job when a node reservation is made for you already
# SBATCH --reservation=reservation_name


# Command to run a gpu job
# For example:

# module load cuda/7


module load anaconda/2019.03-Python3.7-gcc5
# module load anaconda

module load pytorch/1.6-cuda10


python -V
python ../train.py --auxiliary --cutout --arch "Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('skip_connect', 1), ('max_pool_3x3', 2), ('dil_conv_5x5', 4), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))"


nvidia-smi
# deviceQuery

