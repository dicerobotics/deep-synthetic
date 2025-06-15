#!/usr/bin/env bash
set -ex

# Print a message
echo "Testing Contrastive Unpaired Translation (CUT)..."

DATA_ROOT=./datasets/train_ready/CUT_DSIAC_Oktal/
RESULTS_DIR=./results


# echo "Testing/infering CUT for cut_mwir_real2sym..."
python ./test_cut.py \
  --dataroot $DATA_ROOT \
  --name cut_real2psim \
  --CUT_mode CUT \
  --phase test \
  --no_dropout \
  --results_dir $RESULTS_DIR \
  --num_test 9999  # Default 50, Or a large number greater than your test data size 

echo "Finished testing/infering CUT for cut_mwir_real2sym."

# The cd only affects the script's process, not your terminal. When the script finishes, you're still in the same directory you started in — unless you source the script.

