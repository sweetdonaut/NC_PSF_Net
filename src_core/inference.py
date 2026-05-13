import argparse
import glob
import os

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from dataloader import calculate_positions, ensure_2ch, ensure_hwc
from model import SegmentationNetwork


def _list_local(root, img_format):
    if img_format == "png_jpg":
        files = glob.glob(os.path.join(root, "*.png"))
        files.extend(glob.glob(os.path.join(root, "*.jpg")))
    else:
        files = glob.glob(os.path.join(root, "*.tiff"))
        files.extend(glob.glob(os.path.join(root, "*.tif")))
    return sorted(files)


def _pair_paths(prev_paths, next_paths):
    def key(p):
        return os.path.splitext(os.path.basename(p))[0]
    next_by_key = {key(p): p for p in next_paths}
    pairs = []
    for p in prev_paths:
        k = key(p)
        if k in next_by_key:
            pairs.append((p, next_by_key[k]))
    return pairs


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


def sliding_window_inference(prev_image, next_image, model, patch_size, device):
    """Run sliding-window inference on a paired prev/next image and stitch heatmap.

    Uses center-crop stitching: each patch contributes only its non-overlapping
    center region to the output (same scheme as Background_Removal_Net).
    """
    h, w = prev_image.shape[:2]
    patch_h, patch_w = patch_size
    if h < patch_h or w < patch_w:
        raise ValueError(f"Image size ({h}x{w}) < patch size ({patch_h}x{patch_w})")

    y_positions = calculate_positions(h, patch_h)
    x_positions = calculate_positions(w, patch_w)
    if y_positions is None or x_positions is None:
        raise ValueError(f"Image ({h}x{w}) too small for patch ({patch_h}x{patch_w})")

    output_heatmap = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    for y_idx, y in enumerate(y_positions):
        for x_idx, x in enumerate(x_positions):
            prev_patch = prev_image[y:y + patch_h, x:x + patch_w]
            next_patch = next_image[y:y + patch_h, x:x + patch_w]
            four = np.stack([
                prev_patch[:, :, 0], prev_patch[:, :, 1],
                next_patch[:, :, 0], next_patch[:, :, 1],
            ], axis=0)
            tensor = torch.from_numpy(four).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                sm = F.softmax(output, dim=1)
                patch_heatmap = sm[:, 1, :, :].squeeze().cpu().numpy()

            if len(y_positions) > 1 or len(x_positions) > 1:
                y_stride = y_positions[1] - y_positions[0] if len(y_positions) > 1 else patch_h
                y_margin = (patch_h - y_stride) // 2

                if y_idx == 0:
                    y_start_crop, y_end_crop = 0, patch_h - y_margin
                elif y_idx == len(y_positions) - 1:
                    y_start_crop, y_end_crop = y_margin, patch_h
                else:
                    y_start_crop, y_end_crop = y_margin, patch_h - y_margin

                if len(x_positions) > 1:
                    x_stride = x_positions[1] - x_positions[0]
                    x_margin = (patch_w - x_stride) // 2
                    if x_idx == 0:
                        x_start_crop, x_end_crop = 0, patch_w - x_margin
                    elif x_idx == len(x_positions) - 1:
                        x_start_crop, x_end_crop = x_margin, patch_w
                    else:
                        x_start_crop, x_end_crop = x_margin, patch_w - x_margin
                else:
                    x_start_crop, x_end_crop = 0, patch_w

                patch_region = patch_heatmap[y_start_crop:y_end_crop, x_start_crop:x_end_crop]
                oy_s, oy_e = y + y_start_crop, y + y_end_crop
                ox_s, ox_e = x + x_start_crop, x + x_end_crop
                output_heatmap[oy_s:oy_e, ox_s:ox_e] = patch_region
                weight_map[oy_s:oy_e, ox_s:ox_e] = 1
            else:
                output_heatmap[y:y + patch_h, x:x + patch_w] = patch_heatmap
                weight_map[y:y + patch_h, x:x + patch_w] = 1

    return output_heatmap / np.maximum(weight_map, 1)


def visualize_results(prev_image, next_image, heatmap, output_path):
    fig = plt.figure(figsize=(12, 6), dpi=200)
    gs = gridspec.GridSpec(2, 3, figure=fig)

    labels = [
        ("prev_T", prev_image[:, :, 0]),
        ("prev_R", prev_image[:, :, 1]),
        ("next_T", next_image[:, :, 0]),
        ("next_R", next_image[:, :, 1]),
    ]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (label, img), (r, c) in zip(labels, positions):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    ax_hm = fig.add_subplot(gs[:, 2])
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min < 1e-8:
        h_min, h_max = 0, 1
    im = ax_hm.imshow(heatmap, cmap="hot", vmin=h_min, vmax=h_max)
    ax_hm.set_title("New defect heatmap", fontsize=10)
    ax_hm.axis("off")
    plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def inference(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {args.gpu_id}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    patch_size = (checkpoint["img_height"], checkpoint["img_width"])
    print(f"Model patch size: {patch_size}")

    model = SegmentationNetwork(in_channels=4, out_channels=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    prev_paths = _list_local(args.prev_test_path, args.img_format)
    next_paths = _list_local(args.next_test_path, args.img_format)
    pairs = _pair_paths(prev_paths, next_paths)
    if not pairs:
        raise ValueError("No prev/next pairs matched by basename")
    print(f"Paired {len(pairs)} prev/next test images")

    for prev_path, next_path in pairs:
        prev_img = _load_image(prev_path, args.img_format)
        next_img = _load_image(next_path, args.img_format)
        if prev_img.shape != next_img.shape:
            print(f"Skipping pair (shape mismatch): {prev_path} / {next_path}")
            continue

        heatmap = sliding_window_inference(prev_img, next_img, model, patch_size, device)

        filename = os.path.splitext(os.path.basename(prev_path))[0]
        output_path = os.path.join(args.output_dir, f"{filename}_result.png")
        visualize_results(prev_img, next_img, heatmap, output_path)
        print(f"Saved result: {output_path}")

    print(f"\nInference completed. Results saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Inference for NC_PSF_Net (paired prev/next)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prev_test_path", type=str, required=True)
    parser.add_argument("--next_test_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id (-1 for CPU)")
    parser.add_argument("--img_format", type=str, choices=["png_jpg", "tiff"], default="tiff")
    args = parser.parse_args()
    inference(args)


if __name__ == "__main__":
    main()
