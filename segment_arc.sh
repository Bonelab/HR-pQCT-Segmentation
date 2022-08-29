#!/bin/bash
if [ $# -lt 3 ]
then
  echo "Error: not enough arguments given."
  echo "Required arguments:"
  echo "--------------------"
  echo "Argument 1: name of the conda environment to use, e.g. \`blptl\`"
  echo "Argument 2: path to the directory that contains the images to segment, e.g. \`./data/images\`"
  echo "Argument 3: the label of the trained model to use to segment, e.g. \`radius_and_tibia_final\`"
  echo "--------------------"
  echo "Example usage: ./segment_arc.sh blptl ./data/images radius_and_tibia_final"
  echo ""
  exit
fi
sbatch --export=ENVNAME="$1",IMAGEDIR="$2",MODELLABEL="$3" slurm/segment.slurm