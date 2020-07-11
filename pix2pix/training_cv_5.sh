#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0
patch_root_dir=/projects/shart/digital_pathology/data/PenMarking/WSIs_Img_pairs_256   # directory where saves image patch pairs for training

# train with 50 cases
training_case_txt_50=/projects/shart/digital_pathology/data/PenMarking/WSIs/training_cases_50_cv_5.txt # directory where saves WSI training cases
Train_output=/projects/shart/digital_pathology/data/PenMarking/model/pixel2pixel_256_50cases_cv5  # Output directory to save trained models.

python pix2pix.py \
  --mode train \
  --output_dir ${Train_output} \
  --max_epochs 10 \
  --input_patch_root_dir ${patch_root_dir} \
  --input_case_list_txt ${training_case_txt_50} \
  --which_direction BtoA

