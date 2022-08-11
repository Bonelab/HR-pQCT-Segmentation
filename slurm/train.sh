#! /bin/bash

NUM_EPOCHS=25
LABEL="UNet_Final_v2"

SLURM_DIR="slurm"
BASE_SCRIPT="base.slurm"

# first job for epoch 1
# create the slurm file
cp $SLURM_DIR/$BASE_SCRIPT $SLURM_DIR/epoch1.slurm
sed -i "s/LABEL_ARG/--label ${LABEL}_epoch1/" $SLURM_DIR/epoch1.slurm
sed -i "s/PREV_MODEL_ARG//" $SLURM_DIR/epoch1.slurm
sed -i "s/PREV_OPT_ARG//" $SLURM_DIR/epoch1.slurm
sed -i "s/START_EPOCH_ARG/--starting-epoch 1/" $SLURM_DIR/epoch1.slurm
sed -i "s/STOP_EPOCH_ARG/--stopping-epoch 2/" $SLURM_DIR/epoch1.slurm
jid=$(sbatch $SLURM_DIR/epoch1.slurm | tr -dc "0-9")
echo "Submitted job ${jid}, epoch 1. Sleeping for 5 seconds..."
sleep 5

# now we use a for loop to create the slurm scripts and run
# the jobs for the rest of the epochs, telling them not to
# run until the previous job is done
for (( epoch=2; epoch<=$NUM_EPOCHS; epoch++ ))
do
  prev_epoch="$(($epoch-1))"
  next_epoch="$(($epoch+1))"
  cp $SLURM_DIR/$BASE_SCRIPT $SLURM_DIR/epoch${epoch}.slurm
  sed -i "s/LABEL_ARG/--label ${LABEL}_epoch${epoch}/" $SLURM_DIR/epoch${epoch}.slurm
  sed -i "s/PREV_MODEL_ARG/--prev-trained-model trained_models\/${LABEL}_epoch${prev_epoch}.pth/" $SLURM_DIR/epoch${epoch}.slurm
  sed -i "s/PREV_OPT_ARG/--prev-optimizer optimizer_state_dicts\/${LABEL}_epoch${prev_epoch}.pth/" $SLURM_DIR/epoch${epoch}.slurm
  sed -i "s/START_EPOCH_ARG/--starting-epoch ${epoch}/" $SLURM_DIR/epoch${epoch}.slurm
  sed -i "s/STOP_EPOCH_ARG/--stopping-epoch ${next_epoch}/" $SLURM_DIR/epoch${epoch}.slurm
  jid=$(sbatch --dependency=afterany:$jid $SLURM_DIR/epoch${epoch}.slurm | tr -dc "0-9")
  echo "Submitted job ${jid}, epoch ${epoch}. Sleeping for 5 seconds..."
  sleep 5
done
