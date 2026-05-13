"""Evaluate a trained model on test pairs with synthesized 9-case defects.

For each test pair, runs the same 9-case synthesis the dataloader does, then
visualizes: 4 input channels, GT mask, predicted heatmap, overlay diff.
"""

import argparse
import os

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from dataloader import (
    CHANNEL_ORDER,
    PairedDataset,
    calculate_positions,
    ensure_2ch,
    ensure_hwc,
)
from model import SegmentationNetwork


def _load_image(path, img_format):
    if img_format == "tiff":
        img = tifffile.imread(path)
    else:
        img = cv2.imread(path)
    img = ensure_2ch(ensure_hwc(img)).astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_min < 0 or img_max > 255:
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255.0
        else:
            img = np.zeros_like(img)
    return img


def visualize_eval(channels, gt_mask, heatmap, output_path):
    fig = plt.figure(figsize=(14, 7), dpi=200)
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1.2])

    panels = [
        ("prev_T", channels["prev_T"]),
        ("prev_R", channels["prev_R"]),
        ("next_T", channels["next_T"]),
        ("next_R", channels["next_R"]),
    ]
    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (label, img), (r, c) in zip(panels, coords):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    ax_gt = fig.add_subplot(gs[0, 2])
    ax_gt.imshow(gt_mask, cmap="hot", vmin=0, vmax=1)
    ax_gt.set_title("GT (00→10 and 01→10)", fontsize=9)
    ax_gt.axis("off")

    ax_pred = fig.add_subplot(gs[1, 2])
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min < 1e-8:
        h_min, h_max = 0, 1
    ax_pred.imshow(heatmap, cmap="hot", vmin=h_min, vmax=h_max)
    ax_pred.set_title("Predicted heatmap", fontsize=9)
    ax_pred.axis("off")

    ax_overlay = fig.add_subplot(gs[:, 3])
    rgb = np.zeros((*heatmap.shape, 3), dtype=np.float32)
    rgb[..., 0] = heatmap  # red = prediction
    rgb[..., 1] = gt_mask  # green = GT (overlap appears yellow)
    rgb = np.clip(rgb, 0, 1)
    ax_overlay.imshow(rgb)
    ax_overlay.set_title("R=pred  G=GT  Y=overlap", fontsize=9)
    ax_overlay.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def sliding_window_inference(channels_full, model, patch_size, device):
    """Run sliding-window inference on a full 4-channel image (CHW dict).

    channels_full: dict mapping CHANNEL_ORDER name -> (H, W) float32 (0..255).
    Returns: heatmap (H, W) in [0, 1].
    """
    h, w = channels_full["prev_T"].shape
    patch_h, patch_w = patch_size
    y_positions = calculate_positions(h, patch_h)
    x_positions = calculate_positions(w, patch_w)
    if y_positions is None or x_positions is None:
        raise ValueError(f"Image too small for patch {patch_size}")

    out_map = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    for y_idx, y in enumerate(y_positions):
        for x_idx, x in enumerate(x_positions):
            stacked = np.stack([
                channels_full[k][y:y + patch_h, x:x + patch_w]
                for k in CHANNEL_ORDER
            ], axis=0)
            t = torch.from_numpy(stacked).float() / 255.0
            t = t.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(t)
                sm = F.softmax(logits, dim=1)
                p = sm[:, 1, :, :].squeeze().cpu().numpy()

            if len(y_positions) > 1 or len(x_positions) > 1:
                y_stride = y_positions[1] - y_positions[0] if len(y_positions) > 1 else patch_h
                y_margin = (patch_h - y_stride) // 2
                if y_idx == 0:
                    ys, ye = 0, patch_h - y_margin
                elif y_idx == len(y_positions) - 1:
                    ys, ye = y_margin, patch_h
                else:
                    ys, ye = y_margin, patch_h - y_margin
                if len(x_positions) > 1:
                    x_stride = x_positions[1] - x_positions[0]
                    x_margin = (patch_w - x_stride) // 2
                    if x_idx == 0:
                        xs, xe = 0, patch_w - x_margin
                    elif x_idx == len(x_positions) - 1:
                        xs, xe = x_margin, patch_w
                    else:
                        xs, xe = x_margin, patch_w - x_margin
                else:
                    xs, xe = 0, patch_w

                out_map[y + ys:y + ye, x + xs:x + xe] = p[ys:ye, xs:xe]
                weight[y + ys:y + ye, x + xs:x + xe] = 1
            else:
                out_map[y:y + patch_h, x:x + patch_w] = p
                weight[y:y + patch_h, x:x + patch_w] = 1

    return out_map / np.maximum(weight, 1)


def synthesize_full_image_defects(dataset, channels_full, rng_seed):
    """Apply 9-case defects across the entire (not patched) image."""
    np.random.seed(rng_seed)
    h, w = channels_full["prev_T"].shape
    # Run dataloader's defect generator on the full-size image as if it were
    # one big patch. dataloader.generate_paired_defects handles arbitrary shape.
    channels_inp = {k: v.copy() for k, v in channels_full.items()}
    return dataset.generate_paired_defects(channels_inp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prev_test_path", type=str, required=True)
    parser.add_argument("--next_test_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--img_format", type=str, choices=["png_jpg", "tiff"], default="tiff")
    parser.add_argument("--num_defects_range", type=int, nargs=2, default=[5, 10])
    parser.add_argument("--defect_mode", type=str, choices=["gaussian", "psf"], default="psf")
    parser.add_argument("--psf_type", type=str, nargs="+", default=["type1", "type2", "type3"])
    parser.add_argument("--psf_pool_size", type=int, default=200)
    parser.add_argument("--psf_pool_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n_images", type=int, default=10,
                        help="Number of test pairs to visualize")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = (torch.device(f"cuda:{args.gpu_id}")
              if torch.cuda.is_available() and args.gpu_id >= 0
              else torch.device("cpu"))
    print(f"Device: {device}")

    checkpoint = torch.load(args.model_path, map_location=device)
    patch_size = (checkpoint["img_height"], checkpoint["img_width"])
    model = SegmentationNetwork(in_channels=4, out_channels=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.model_path}; patch_size={patch_size}")

    defects_dir = os.path.join(os.path.dirname(__file__), "defects")
    psf_config_paths = (
        [os.path.join(defects_dir, f"{t}.yaml") for t in args.psf_type]
        if args.defect_mode == "psf" else None
    )

    # We instantiate PairedDataset just to reuse its defect-generation logic.
    # The path args are required but we won't iterate via __getitem__.
    dataset = PairedDataset(
        prev_path=args.prev_test_path,
        next_path=args.next_test_path,
        patch_size=patch_size,
        num_defects_range=tuple(args.num_defects_range),
        img_format=args.img_format,
        defect_mode=args.defect_mode,
        psf_config_paths=psf_config_paths,
        psf_pool_size=args.psf_pool_size,
        psf_pool_workers=args.psf_pool_workers,
        no_defect_prob=0.0,  # always synthesize defects for eval
    )

    pairs = dataset.training_paths[: args.n_images]
    print(f"Evaluating {len(pairs)} test pairs")

    all_scores, all_labels = [], []
    for i, (prev_path, next_path) in enumerate(pairs):
        prev_img = _load_image(prev_path, args.img_format)
        next_img = _load_image(next_path, args.img_format)
        channels_full = {
            "prev_T": prev_img[:, :, 0].copy(),
            "prev_R": prev_img[:, :, 1].copy(),
            "next_T": next_img[:, :, 0].copy(),
            "next_R": next_img[:, :, 1].copy(),
        }
        channels_with_defects, gt_mask = synthesize_full_image_defects(
            dataset, channels_full, rng_seed=args.seed + i)
        heatmap = sliding_window_inference(
            channels_with_defects, model, patch_size, device)

        name = os.path.splitext(os.path.basename(prev_path))[0]
        out_path = os.path.join(args.output_dir, f"{name}_eval.png")
        visualize_eval(channels_with_defects, gt_mask, heatmap, out_path)
        print(f"  [{i+1}/{len(pairs)}] {name}: GT positives={gt_mask.sum():.0f}, "
              f"heatmap range=[{heatmap.min():.3f}, {heatmap.max():.3f}]  -> {out_path}")

        all_scores.append(heatmap.reshape(-1))
        all_labels.append(gt_mask.reshape(-1))

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    if len(np.unique(labels)) > 1:
        auroc = roc_auc_score(labels, scores)
        print(f"\nPixel AUROC over {len(pairs)} test pairs: {auroc:.4f}")
    else:
        print("\nPixel AUROC: cannot compute (only one class present)")


if __name__ == "__main__":
    main()
