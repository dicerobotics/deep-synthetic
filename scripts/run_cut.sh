#!/usr/bin/env bash

# Print a message
echo "Starting style transfer with Contrastive Unpaired Translation (CUT)..."

# Change directory to the CUT source
cd ./../third_party/CUT/  || { echo "Failed to change directory"; exit 1; }

# Run the Contrastive Unpaired Translation (CUT) program
echo "Start running Contrastive Unpaired Translation (CUT)..."
echo "Training CUT..."
python train.py \
  --dataroot ./../../datasets/mwir2sim \
  --name mwir2sim_cut \
  --CUT_mode CUT \
  --n_epochs 1 \
  --n_epochs_decay 1 \
  --gpu_ids 0

# Indicate training completion.
echo "Finished training CUT."


echo "Testing/infering CUT..."
python test.py \
  --dataroot ./../../datasets/mwir2sim \
  --name mwir2sim_cut \
  --CUT_mode CUT \
  --phase test \
  --no_dropout \
  --results_dir ./../../results/ \
  --num_test 5  # Or a large number greater than your dataset size, default 50
  
echo "Finished testing/infering CUT."

# The cd only affects the script's process, not your terminal. When the script finishes, you're still in the same directory you started in — unless you source the script.

