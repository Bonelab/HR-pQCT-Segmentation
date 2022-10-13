#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --mem=80G

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate "$ENVNAME"
python -u segment.py "$IMAGEDIR" "$MODELLABEL"
