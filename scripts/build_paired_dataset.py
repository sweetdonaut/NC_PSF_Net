"""
Build a paired prev/next dataset from Background_Removal_Net grayscale tiffs.

Strategy (matches docs/design.md strategy B — fully synthetic):
  - For each source grayscale image `gray`:
    - prev: stack 3 copies + small per-die noise (sigma_die)
    - next: apply systematic process shift (brightness * k_b + offset_b +
      Gaussian sigma_proc), then stack 3 copies + per-die noise

Output: tiffs saved CHW (3, H, W) float32 to mirror BRN's convention.
"""

import argparse
import json
import os
from glob import glob

import numpy as np
import tifffile
from tqdm import tqdm


def to_grayscale(img):
    """Reduce arbitrary tiff to single-channel grayscale (H, W) float32.

    Handles BRN format (4, H, W) where ch0 carries the actual data and ch1/2
    are duplicates, ch3 is zero. Also handles plain (H, W) and HWC inputs.
    """
    if img.ndim == 2:
        return img.astype(np.float32)
    if img.ndim == 3:
        if img.shape[0] in (3, 4) and img.shape[1] > 4 and img.shape[2] > 4:
            return img[0].astype(np.float32)
        if img.shape[2] in (3, 4):
            return img[:, :, 0].astype(np.float32)
    raise ValueError(f"Unexpected image shape: {img.shape}")


def synth_pair(gray, rng,
               brightness_range=(0.85, 1.15),
               offset_range=(-15.0, 15.0),
               sigma_proc_range=(0.3, 1.0),
               sigma_die=0.3):
    """Produce one (prev_chw, next_chw) pair from a single grayscale image."""
    h, w = gray.shape
    k_b = float(rng.uniform(*brightness_range))
    off = float(rng.uniform(*offset_range))
    sigma_proc = float(rng.uniform(*sigma_proc_range))

    def add_die_noise(base):
        # 2 dies per station: target (T) + 1 reference (R)
        return np.stack([
            base + rng.normal(0, sigma_die, base.shape).astype(np.float32),
            base + rng.normal(0, sigma_die, base.shape).astype(np.float32),
        ], axis=0)

    prev = np.clip(add_die_noise(gray), 0, 255).astype(np.float32)

    gray_next = gray * k_b + off
    gray_next = gray_next + rng.normal(0, sigma_proc, gray.shape).astype(np.float32)
    gray_next = np.clip(gray_next, 0, 255)
    nxt = np.clip(add_die_noise(gray_next), 0, 255).astype(np.float32)

    meta = {"brightness": k_b, "offset": off, "sigma_proc": sigma_proc,
            "sigma_die": sigma_die}
    return prev, nxt, meta


def build_dataset(src_glob, out_root, val_count, test_count, seed):
    paths = sorted(glob(src_glob))
    if not paths:
        raise ValueError(f"No images matched: {src_glob}")
    print(f"Found {len(paths)} source images")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(paths))

    n = len(paths)
    n_test = test_count
    n_val = val_count
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Not enough images: {n} - val {n_val} - test {n_test}")
    splits = {
        "train": [paths[i] for i in perm[:n_train]],
        "val":   [paths[i] for i in perm[n_train:n_train + n_val]],
        "test":  [paths[i] for i in perm[n_train + n_val:]],
    }
    print(f"Split: train={n_train}, val={n_val}, test={n_test}")

    manifest = {"seed": seed, "source_glob": src_glob, "splits": {}}
    for split, srcs in splits.items():
        prev_dir = os.path.join(out_root, split, "prev")
        next_dir = os.path.join(out_root, split, "next")
        os.makedirs(prev_dir, exist_ok=True)
        os.makedirs(next_dir, exist_ok=True)

        records = []
        for src in tqdm(srcs, desc=split):
            img = tifffile.imread(src)
            gray = to_grayscale(img)
            prev, nxt, meta = synth_pair(gray, rng)
            name = os.path.basename(src)
            tifffile.imwrite(os.path.join(prev_dir, name), prev)
            tifffile.imwrite(os.path.join(next_dir, name), nxt)
            records.append({"name": name, "src": src, **meta})
        manifest["splits"][split] = records

    manifest_path = os.path.join(out_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        default="/home/yclaizzs/ML_exploration/Background_Removal_Net/"
                "data/grid_stripe_4channel/train/good/*.tiff",
        help="Glob pattern of source grayscale tiffs (BRN format).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Output dataset root.",
    )
    parser.add_argument("--val", type=int, default=25)
    parser.add_argument("--test", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_root = os.path.abspath(args.out)
    build_dataset(args.src, out_root, args.val, args.test, args.seed)
    print(f"\nDone. Output at: {out_root}")


if __name__ == "__main__":
    main()
