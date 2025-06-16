#!/bin/bash

# We need to generate four kinds of results to calculate full evaluate_matrics set
# ---------psim start----------------------------------
DATA_ROOT=./datasets/train_ready/Pix2Pix_GradCAM/
RESULTS_DIR=./results/psim_start # For pseudopair start
DIRECTION=AtoB


# ----------oktal start----------------------------------------------
# DATA_ROOT=./datasets/train_ready/cat_oktal_dsiac/ # For oktal start
# RESULTS_DIR=./results/oktal_start # For oktal start
# DIRECTION=AtoB


# --------- psim target (Use relevant heatmaps only)-----------------------------------------------------------
# DATA_ROOT=./datasets/train_ready/Pix2Pix_GradCAM/
# RESULTS_DIR=./results/psim_target # For psim as target domain (Use gradcam heatmaps for real_B (psim) only )
# DIRECTION=BtoA


# --------- oktal target (Use relevant heatmaps only)-----------------------------------------------------------
# DATA_ROOT=./datasets/train_ready/cat_oktal_dsiac/
# RESULTS_DIR=./results/oktal_target # For oktal as target domain (Use gradcam heatmaps for real_B (oktal) only)
# DIRECTION=BtoA



# Set shared parameters
CLASSIFIER_ROOT=./checkpoints/classifiers/resnet18_mwir.pth
EPOCH=latest
CHECKPOINT_DIR=./checkpoints
PHASE=test
NUM_TEST=9999  # adjust as needed
GPU_ID=0

# # -------- Pix2Pix Baseline --------
# Set --lambda_gradcam to 0.0 (no heatmaps output) or --lambda_gradcam to 0.0001 (for heatmaps output)
# echo "Running Pix2Pix baseline test..."
python test.py \
  --dataroot $DATA_ROOT \
  --name pix2pix_gradcam_psim2preal_1_10_0.0001 \
  --model pix2pix_gradcam \
  --netG unet_256 \
  --direction $DIRECTION \
  --dataset_mode aligned \
  --results_dir $RESULTS_DIR \
  --num_test $NUM_TEST \
  --phase $PHASE \
  --epoch $EPOCH \
  --gpu_ids $GPU_ID \
  --resnet18_path $CLASSIFIER_ROOT \
  --lambda_gradcam 0.0001

# # -------- Pix2Pix + GradCAM --------
echo "Running Pix2Pix + GradCAM test..."
python test.py \
  --dataroot $DATA_ROOT \
  --name pix2pix_gradcam_psim2preal_1_10_50 \
  --model pix2pix_gradcam \
  --netG unet_256 \
  --direction $DIRECTION \
  --dataset_mode aligned \
  --results_dir $RESULTS_DIR \
  --num_test $NUM_TEST \
  --phase $PHASE \
  --epoch $EPOCH \
  --gpu_ids $GPU_ID \
  --resnet18_path $CLASSIFIER_ROOT \
  --lambda_gradcam 50

# echo "Tests script complete."
