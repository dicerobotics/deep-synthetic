#!/usr/bin/env bash

# Print a message
echo "Starting style transfer with Contrastive Unpaired Translation (CUT)..."

# Change directory to the CUT source
cd ./../third_party/CUT/  || { echo "Failed to change directory"; exit 1; }

# Run the Contrastive Unpaired Translation (CUT) program
echo "Training CUT for mwir_sim2cut### ..."
python train.py \
  --dataroot ./../../datasets/mwir_sim2cut \
  --name mwir_sim2cut \
  --CUT_mode CUT \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --gpu_ids 0

# Indicate training completion.
echo "Finished training CUT for mwir_sim2cut."


echo "Testing/infering CUT for mwir_sim2cut..."
python test.py \
  --dataroot ./../../datasets/mwir_sim2cut \
  --name mwir_sim2cut \
  --CUT_mode CUT \
  --phase test \
  --no_dropout \
  --results_dir ./../../results/ \
  --num_test 50  # Default 50, Or a large number greater than your dataset size i.e. 9200 for mwir_sim2cut

echo "Finished testing/infering CUT for mwir_sim2cut."

# The cd only affects the script's process, not your terminal. When the script finishes, you're still in the same directory you started in — unless you source the script.

