#!/usr/bin/env bash
# NC_PSF_Net training script (MVP)
#
# Default config: PSF defect mode, paired prev/next inputs.
# Each prev/next dir contains same-named 2-channel grayscale images
# (target + 1 reference per station; model input = 4 channels total).

python trainer.py \
    --bs 16 \
    --lr 0.001 \
    --epochs 20 \
    --gpu_id 0 \
    --checkpoint_path ../checkpoints/mvp \
    --patch_size 128 \
    --num_defects_range 4 10 \
    --prev_path ../data/train/prev/ \
    --next_path ../data/train/next/ \
    --valid_prev_path ../data/val/prev/ \
    --valid_next_path ../data/val/next/ \
    --img_format tiff \
    --cache_size 100 \
    --defect_mode psf \
    --psf_type type1 type2 type3 \
    --psf_pool_size 1000 \
    --psf_pool_workers 6 \
    --gamma_start 2.0 \
    --gamma_end 2.0 \
    --num_workers 4 \
    --prefetch_factor 2 \
    --seed 42

# Gaussian-mode fallback example:
# python trainer.py \
#     --bs 16 --lr 0.001 --epochs 100 --gpu_id 0 \
#     --checkpoint_path ../checkpoints/mvp_gaussian \
#     --patch_size 128 --num_defects_range 4 10 \
#     --prev_path ../data/train/prev/ --next_path ../data/train/next/ \
#     --img_format tiff --defect_mode gaussian

# Vector PSF example (uses type4_vector_strong.yaml, intensity 60-80
# instead of the original [[8, 12]] which is calibrated for real optical
# amplitude, not pixel-perturbation magnitude on synthetic baseline):
# python trainer.py \
#     --bs 16 --lr 0.001 --epochs 20 --gpu_id 0 \
#     --checkpoint_path ../checkpoints/mvp_vector \
#     --patch_size 128 --num_defects_range 4 10 \
#     --prev_path ../data/train/prev/ --next_path ../data/train/next/ \
#     --valid_prev_path ../data/val/prev/ --valid_next_path ../data/val/next/ \
#     --img_format tiff --defect_mode psf --psf_type type4_vector_strong \
#     --psf_pool_size 1000 --psf_pool_workers 6 --seed 42
