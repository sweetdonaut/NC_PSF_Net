import argparse
import math
import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader

from dataloader import PairedDataset, sample_magnitude
from loss import FocalLoss
from model import SegmentationNetwork


def _render_defect_grid(defects, out_path, suptitle=None):
    """Render a 3x5 grid of defect patches (centered + padded) with colorbars."""
    if not defects:
        print(f"Warning: no defects to render, skipping {out_path}")
        return

    max_h = max(d[0].shape[0] for d in defects)
    max_w = max(d[0].shape[1] for d in defects)
    abs_max = max(abs(d[1]) for d in defects)

    rows, cols = 3, 5
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    for ax_idx, ax in enumerate(axes.flat):
        if ax_idx >= len(defects):
            ax.axis("off")
            continue
        local_defect, intensity = defects[ax_idx]
        h, w = local_defect.shape
        padded = np.zeros((max_h, max_w), dtype=np.float32)
        py, px = (max_h - h) // 2, (max_w - w) // 2
        padded[py:py + h, px:px + w] = local_defect
        signed = padded * intensity
        im = ax.imshow(signed, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"patch {ax_idx + 1} ({h}x{w}, x{intensity:+.1f})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved defect examples: {out_path}")


def save_training_artifacts(checkpoint_path, dataset, psf_config_paths, n_patches=15):
    """Copy PSF configs and render defect example grids for reproducibility."""
    if psf_config_paths:
        for src in psf_config_paths:
            shutil.copy(src, os.path.join(checkpoint_path, os.path.basename(src)))

    state = np.random.get_state()
    try:
        if dataset.defect_mode == "psf":
            for type_idx, cfg_path in enumerate(psf_config_paths):
                type_name = os.path.splitext(os.path.basename(cfg_path))[0]
                pool = dataset.defect_pool.pools[type_idx]
                cfg = dataset.defect_pool.cfgs[type_idx]
                intensity_spec = cfg.get("intensity_abs", dataset.intensity_abs)

                defects = []
                for _ in range(n_patches):
                    d = pool[np.random.randint(len(pool))]
                    magnitude = sample_magnitude(intensity_spec)
                    intensity = magnitude if np.random.rand() < 0.5 else -magnitude
                    defects.append((d, intensity))

                out = os.path.join(checkpoint_path, f"defect_examples_{type_name}.png")
                _render_defect_grid(defects, out, suptitle=type_name)
        else:
            H = W = 200
            defects = []
            attempts = 0
            while len(defects) < n_patches and attempts < n_patches * 30:
                local_defect, _, intensity = dataset._create_one_defect(H, W)
                if local_defect is not None:
                    defects.append((local_defect, intensity))
                attempts += 1
            out = os.path.join(checkpoint_path, "defect_examples.png")
            _render_defect_grid(defects, out)
    finally:
        np.random.set_state(state)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_focal_gamma(epoch, total_epochs, gamma_start, gamma_end, schedule="cosine"):
    if schedule == "linear":
        return gamma_start + (gamma_end - gamma_start) * (epoch / total_epochs)
    if schedule == "cosine":
        progress = epoch / total_epochs
        return gamma_start + (gamma_end - gamma_start) * (1 - math.cos(progress * math.pi)) / 2
    raise ValueError(f"Unknown schedule: {schedule}")


def evaluate_synthetic(model, val_loader, criterion, device, max_pixels=2_000_000, val_seed=12345):
    """Compute val loss + pixel-level AUROC on the synthetic validation set.

    The validation set still uses random defect synthesis, but we seed numpy
    inside the loop so the defect pattern is consistent across epochs (only
    model output differs).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    rng_state_np = np.random.get_state()
    rng_state_torch = torch.get_rng_state()
    np.random.seed(val_seed)
    torch.manual_seed(val_seed)

    pixel_scores = []
    pixel_labels = []

    with torch.no_grad():
        for sample in val_loader:
            x = sample["paired_input"].to(device)
            y = sample["target_mask"].to(device)
            logits = model(x)
            sm = torch.softmax(logits, dim=1)
            loss = criterion(sm, y)
            total_loss += loss.item()
            n_batches += 1
            score = sm[:, 1, :, :].cpu().numpy().reshape(-1)
            label = y.cpu().numpy().reshape(-1)
            pixel_scores.append(score)
            pixel_labels.append(label)

    np.random.set_state(rng_state_np)
    torch.set_rng_state(rng_state_torch)
    model.train()

    avg_loss = total_loss / max(n_batches, 1)
    if not pixel_scores:
        return avg_loss, 0.0

    scores = np.concatenate(pixel_scores)
    labels = np.concatenate(pixel_labels)
    if len(scores) > max_pixels:
        idx = np.random.choice(len(scores), max_pixels, replace=False)
        scores = scores[idx]
        labels = labels[idx]
    if len(np.unique(labels)) > 1:
        auroc = roc_auc_score(labels, scores)
    else:
        auroc = 0.0
    return avg_loss, auroc


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_on_device(args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {args.seed}")

    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {args.gpu_id}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    patch_size = (args.patch_size, args.patch_size)
    run_name = f"NCPSF_lr{args.lr}_ep{args.epochs}_bs{args.bs}_{patch_size[0]}x{patch_size[1]}"

    model_seg = SegmentationNetwork(in_channels=6, out_channels=2)
    model_seg.to(device)
    model_seg.apply(weights_init)

    optimizer = torch.optim.Adam(model_seg.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        [int(args.epochs * 0.8), int(args.epochs * 0.9)],
        gamma=0.2,
        last_epoch=-1,
    )

    criterion = FocalLoss(alpha=0.75, gamma=args.gamma_start)
    print(f"Using Focal Loss with alpha=0.75, "
          f"gamma schedule: [{args.gamma_start}, {args.gamma_end}] (cosine)")

    psf_config_paths = None
    if args.defect_mode == "psf":
        defects_dir = os.path.join(os.path.dirname(__file__), "defects")
        psf_config_paths = [os.path.join(defects_dir, f"{t}.yaml") for t in args.psf_type]

    train_dataset = PairedDataset(
        prev_path=args.prev_path,
        next_path=args.next_path,
        patch_size=patch_size,
        num_defects_range=tuple(args.num_defects_range),
        img_format=args.img_format,
        cache_size=args.cache_size,
        defect_mode=args.defect_mode,
        psf_config_paths=psf_config_paths,
        psf_pool_size=args.psf_pool_size,
        psf_pool_workers=args.psf_pool_workers,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    print(f"Train dataset size: {len(train_dataset)} samples per epoch")

    val_loader = None
    if args.valid_prev_path and args.valid_next_path:
        val_dataset = PairedDataset(
            prev_path=args.valid_prev_path,
            next_path=args.valid_next_path,
            patch_size=patch_size,
            num_defects_range=tuple(args.num_defects_range),
            img_format=args.img_format,
            cache_size=args.cache_size,
            defect_mode=args.defect_mode,
            psf_config_paths=psf_config_paths,
            psf_pool_size=max(args.psf_pool_size // 4, 64),
            psf_pool_workers=args.psf_pool_workers,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.bs, shuffle=False,
            num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
        print(f"Val dataset size: {len(val_dataset)} samples")

    save_training_artifacts(args.checkpoint_path, train_dataset, psf_config_paths)

    num_batches = len(train_loader)

    for epoch in range(args.epochs):
        current_gamma = get_focal_gamma(epoch, args.epochs, args.gamma_start, args.gamma_end, "cosine")
        criterion.update_params(gamma=current_gamma)

        epoch_loss = 0.0
        for i_batch, sample in enumerate(train_loader):
            x = sample["paired_input"].to(device)
            y = sample["target_mask"].to(device)

            logits = model_seg(x)
            sm = torch.softmax(logits, dim=1)
            loss = criterion(sm, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i_batch % 10 == 0 or i_batch == num_batches - 1:
                current_lr = get_lr(optimizer)
                progress = (i_batch + 1) / num_batches * 100
                print(f"\rEpoch [{epoch+1}/{args.epochs}] - "
                      f"Batch [{i_batch+1}/{num_batches}] ({progress:.1f}%) - "
                      f"Loss: {loss.item():.4e} - LR: {current_lr:.6f}",
                      end="", flush=True)

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch [{epoch+1}/{args.epochs}] Summary - "
              f"Avg Loss: {avg_loss:.4e} - Gamma: {current_gamma:.3f}", end="")

        if val_loader is not None:
            val_loss, val_auroc = evaluate_synthetic(model_seg, val_loader, criterion, device)
            print(f" - Val Loss: {val_loss:.4e} - Val Pixel AUROC: {val_auroc:.4f}")
        else:
            print()

        checkpoint = {
            "model_state_dict": model_seg.state_dict(),
            "img_height": patch_size[0],
            "img_width": patch_size[1],
            "epoch": epoch,
            "seed": args.seed,
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, f"{run_name}.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU id (-1 for CPU)")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--num_defects_range", type=int, nargs=2, default=[3, 8],
                        help="[min, max] defects per patch when defects are added")

    parser.add_argument("--prev_path", type=str, required=True,
                        help="Directory of prev-station training images")
    parser.add_argument("--next_path", type=str, required=True,
                        help="Directory of next-station training images")
    parser.add_argument("--valid_prev_path", type=str, default=None)
    parser.add_argument("--valid_next_path", type=str, default=None)

    parser.add_argument("--img_format", type=str, choices=["png_jpg", "tiff"], default="tiff")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cache_size", type=int, default=0)

    parser.add_argument("--gamma_start", type=float, default=2.0)
    parser.add_argument("--gamma_end", type=float, default=2.0)

    parser.add_argument("--defect_mode", type=str, choices=["gaussian", "psf"], default="psf")
    parser.add_argument("--psf_type", type=str, nargs="+", default=None,
                        help="PSF config names in defects/ (e.g., type1 type2)")
    parser.add_argument("--psf_pool_size", type=int, default=1000)
    parser.add_argument("--psf_pool_workers", type=int, default=4)

    parser.add_argument("--num_workers", type=int, default=7)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    args = parser.parse_args()
    train_on_device(args)


if __name__ == "__main__":
    main()
