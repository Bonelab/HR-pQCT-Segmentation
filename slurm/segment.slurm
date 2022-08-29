#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=8:00:00
#SBATCH --mem=100G
#SBATCH --partition=gpu-v100 --gres=gpu:1

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate "$ENVNAME"
python -u segment.py "$IMAGEDIR" "$MODELLABEL" --cuda
