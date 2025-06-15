#!/bin/bash

# Set shared parameters
DATA_ROOT=./datasets/mwir_pseudopaired_cut20/mwir_cat_AB
# DATA_ROOT=./datasets/mwir_unpaired_oktal/cat_AB # For oktal start
CLASSIFIER_ROOT=./../../checkpoints/classifiers/resnet18_mwir_weights.pth
DIRECTION=BtoA
EPOCH=latest
CHECKPOINT_DIR=./checkpoints
RESULTS_DIR=./results/ # For pseudopair start
# RESULTS_DIR=./results/oktal # For oktal start
PHASE=test
NUM_TEST=920  # adjust as needed
GPU_ID=0

# # -------- Pix2Pix Baseline --------
# Set --lambda_gradcam to 0.0 (no heatmaps output) or --lambda_gradcam to 0.00001 (for heatmaps output)
echo "Running Pix2Pix baseline test..."
python test.py \
  --dataroot $DATA_ROOT \
  --name mwir_pix2pix \
  --model pix2pix_gradcam \
  --netG unet_256 \
  --direction $DIRECTION \
  --dataset_mode aligned \
  --norm batch \
  --results_dir $RESULTS_DIR \
  --num_test $NUM_TEST \
  --phase $PHASE \
  --epoch $EPOCH \
  --gpu_ids $GPU_ID \
  --resnet18_path $CLASSIFIER_ROOT \
  --lambda_gradcam 0.00001

# -------- Pix2Pix + GradCAM --------
echo "Running Pix2Pix + GradCAM test..."
python test.py \
  --dataroot $DATA_ROOT \
  --name mwir_pix2pix_gradcam \
  --model pix2pix_gradcam \
  --netG unet_256 \
  --direction $DIRECTION \
  --dataset_mode aligned \
  --norm batch \
  --results_dir $RESULTS_DIR \
  --num_test $NUM_TEST \
  --phase $PHASE \
  --epoch $EPOCH \
  --gpu_ids $GPU_ID \
  --resnet18_path $CLASSIFIER_ROOT \
  --lambda_gradcam 10.0

echo "Tests script complete."
