#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=13:00:00
#SBATCH --mem=0
#SBATCH --partition=gpu-v100 --gres=gpu:2

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate bl_torch
python -u train.py --cuda --device-ids 0 1 LABEL_ARG PREV_MODEL_ARG PREV_OPT_ARG START_EPOCH_ARG STOP_EPOCH_ARG --training-batch-size 4 --validation-batch-size 20 --training-data-dir ./data/training/ --validation-data-dir ./data/validation/ --image-loader-workers 5 --slice-loader-workers 5 --num-epochs-half-cycle 10 --num-epochs-convergence 5
