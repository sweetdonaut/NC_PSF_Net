#!/usr/bin/env bash
# NC_PSF_Net inference (MVP)
#
# Pair test images by basename across --prev_test_path and --next_test_path.
# Default checkpoint matches train.sh's vector-PSF default output.

python inference.py \
    --model_path ../checkpoints/mvp_vector/NCPSF_lr0.001_ep20_bs16_128x128.pth \
    --prev_test_path ../data/test/prev/ \
    --next_test_path ../data/test/next/ \
    --output_dir ../output/mvp_vector \
    --img_format tiff \
    --gpu_id 0

# Scalar PSF example (matches the scalar example in train.sh):
# python inference.py \
#     --model_path ../checkpoints/mvp_scalar/NCPSF_lr0.001_ep20_bs16_128x128.pth \
#     --prev_test_path ../data/test/prev/ --next_test_path ../data/test/next/ \
#     --output_dir ../output/mvp_scalar --img_format tiff --gpu_id 0
