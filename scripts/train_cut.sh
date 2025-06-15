#!/usr/bin/env bash
set -ex

# Print a message
echo "Starting style transfer with Contrastive Unpaired Translation (CUT)..."
DATA_ROOT=./datasets/train_ready/CUT_DSIAC_Oktal/
CHECKPOINTS_DIR=./checkpoints


# Option 1: Traning from scratch
# Run the Contrastive Unpaired Translation (CUT) program
echo "Training CUT for mwir_real2cut from scratch ..."
python ./train_cut.py \
  --dataroot $DATA_ROOT \
  --name cut_real2psim \
  --n_epochs 2 \
  --n_epochs_decay 2 \
  --gpu_ids 0 \
  --batch_size 2 \
  --num_threads 8 \
  --checkpoints_dir $CHECKPOINTS_DIR 

# Indicate training completion.
echo "Finished training CUT for cut_real2psim."
