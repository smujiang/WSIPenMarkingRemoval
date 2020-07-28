#!/usr/bin/bash

# export CUDA_VISIBLE_DEVICES=2
patch_root_dir=/projects/shart/digital_pathology/data/PenMarking/WSIs_Img_pairs_256   # patches root directory need to be restored
test_output=/projects/shart/digital_pathology/data/PenMarking/WSIs_Img_pairs_256_restored    # evaluation output directory
testing_case_txt_50=/projects/shart/digital_pathology/data/PenMarking/WSIs/testing_cases_50_cv_1.txt # directory where saves WSI training cases

Train_output=/projects/shart/digital_pathology/data/PenMarking/model/pixel2pixel_256_50cases_cv1   # directory where saves the pre-trained model

python pix2pix.py \
  --mode test \
  --output_dir ${test_output} \
  --input_patch_root_dir ${patch_root_dir} \
  --checkpoint ${Train_output}\
  --input_case_list_txt ${testing_case_txt_50} \


