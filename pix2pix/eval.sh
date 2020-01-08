#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=2
patch_root_dir=/data/PenMarking/WSIs_Img_pairs_256    # patchs need to be restored
testing_case_txt=/data/PenMarking/WSIs/testing_cases_50.txt  # WSI case list
test_output=/data/PenMarking/eval/pixel2pixel_256_50cases    # evaluation output directory
mkdir ${test_output}
Train_output=/data/PenMarking/model/pixel2pixel_256_50cases   # directory where saves the pre-trained model

python pix2pix.py \
  --mode test \
  --output_dir ${test_output} \
  --input_patch_root_dir ${patch_root_dir} \
  --input_case_list_txt ${testing_case_txt}  \
  --checkpoint ${Train_output}

