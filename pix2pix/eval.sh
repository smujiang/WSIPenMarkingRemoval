#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=2
patch_root_dir=../img_samples    # patchs need to be restored
test_output=../img_restored    # evaluation output directory
mkdir ${test_output}
Train_output=../model   # directory where saves the pre-trained model

python pix2pix.py \
  --mode test \
  --output_dir ${test_output} \
  --input_patch_root_dir ${patch_root_dir} \
  --checkpoint ${Train_output}

