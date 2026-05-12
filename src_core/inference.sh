#!/usr/bin/env bash
# NC_PSF_Net inference (MVP)
#
# Pair test images by basename across --prev_test_path and --next_test_path.

python inference.py \
    --model_path ../checkpoints/mvp/NCPSF_lr0.001_ep20_bs16_128x128.pth \
    --prev_test_path ../data/test/prev/ \
    --next_test_path ../data/test/next/ \
    --output_dir ../output/mvp \
    --img_format tiff \
    --gpu_id 0
