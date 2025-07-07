#!/bin/bash
set -ex

# Set dataset path
DATA_ROOT=./datasets/train_ready/Pix2Pix_GradCAM/
CLASSIFIER_ROOT=./checkpoints/classifiers/resnet18_mwir.pth
CHECKPOINTS_DIR=./checkpoints


# Train a pix2pix_gradcam model (clean start from scratch)
python ./train.py \
  --dataroot $DATA_ROOT \
  --name pix2pix_gradcam_feat_psim2preal_1_10_50_50 \
  --model pix2pix_gradcam_feat \
  --direction AtoB \
  --resnet18_path $CLASSIFIER_ROOT \
  --checkpoints_dir $CHECKPOINTS_DIR \
  --lambda_L1 10 \
  --lambda_gradcam 50 \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --gpu_ids 0 \
  --batch_size 4

# # (Optional) Finetune a pix2pix_gradcam model (starting from latest pretrained pix2pix):
# # Step 1: Copy pretrained pix2pix weights to new model directory (Total checkpoints = 200)
# if [ ! -d checkpoints/mwir_pix2pix_gradcam ]; then
#   cp -r checkpoints/mwir_pix2pix checkpoints/mwir_pix2pix_gradcam
# fi
# # Step 2: Start training pix2pix_gradcam from epoch 201
# python train.py \
#   --dataroot $DATA_ROOT \
#   --name mwir_pix2pix_gradcam \
#   --model pix2pix_gradcam \
#   --direction AtoB \
#   --resnet18_path $CLASSIFIER_ROOT \
#   --lambda_gradcam 50 \
#   --continue_train \
#   --epoch 200 \
#   --epoch_count 201 \
#   --n_epochs 200 \
#   --n_epochs_decay 200 \
#   --gpu_ids 0

