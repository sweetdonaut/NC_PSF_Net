import torch
from torch.utils.data import Dataset as TorchDataset
import numpy as np
import cv2
import os
import io
import multiprocessing as mp
from glob import glob
import tifffile
from gaussian import (
    create_binary_mask,
    create_local_gaussian_defect,
    apply_local_defect_to_background,
)
from generate_psf import load_config as load_psf_config, generate_one, clean_connected_peak
from functools import lru_cache
from tqdm import tqdm


# 9 defect cases for paired prev/next die-to-die training (2 channels/station).
# Key format "<prev>→<next>" where each pattern is 2 bits (T, R).
# Value: (prev_pattern, next_pattern, is_gt). is_gt=True only for the two
# positive-sample cases that satisfy next_T=1 AND next_R=0 AND prev_T=0.
CASES = {
    "10→10": ((1, 0), (1, 0), False),  # prev T-only passthrough
    "01→01": ((0, 1), (0, 1), False),  # prev R-only passthrough
    "11→11": ((1, 1), (1, 1), False),  # prev T+R passthrough
    "00→10": ((0, 0), (1, 0), True),   # next-station-new T-only (the target!)
    "01→10": ((0, 1), (1, 0), True),   # next-station-new T-only with prev_R distractor
    "00→01": ((0, 0), (0, 1), False),  # next-station-new R-only
    "00→11": ((0, 0), (1, 1), False),  # next-station-new T+R
    "11→10": ((1, 1), (1, 0), False),  # process selectively removed R from prev T+R
    "11→01": ((1, 1), (0, 1), False),  # process selectively removed T from prev T+R
}
CASE_NAMES = list(CASES.keys())
CHANNEL_ORDER = ["prev_T", "prev_R", "next_T", "next_R"]

# The four "next-station T-only family" anchors. Each patch is forced to
# contain at least one of each, because the four are visually identical in
# the next station (all T-only) but differ in GT and in prev_T, forcing the
# model to learn "look at prev_T, ignore prev_R".
ANCHORS = ["00→10", "01→10", "10→10", "11→10"]


def parse_s3_path(s3_path):
    path = s3_path.replace("s3://", "", 1)
    parts = path.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def list_s3_objects(bucket, prefix, img_format):
    import boto3
    endpoint_url = os.environ.get("S3_ENDPOINT_URL")
    client = boto3.client("s3", endpoint_url=endpoint_url)
    extensions = (".tiff", ".tif") if img_format == "tiff" else (".png", ".jpg")
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(extensions):
                keys.append(key)
    return [f"s3://{bucket}/{k}" for k in sorted(keys)]


def ensure_hwc(image):
    """Convert CHW to HWC if shape signals it. Accepts 2/3/4 leading-dim CHW."""
    if (len(image.shape) == 3
            and image.shape[0] in (2, 3, 4)
            and image.shape[1] > 4
            and image.shape[2] > 4):
        return np.transpose(image, (1, 2, 0))
    return image


def ensure_2ch(image):
    """Keep only first 2 channels (drop extras if present)."""
    if len(image.shape) == 3 and image.shape[2] > 2:
        return image[:, :, :2]
    return image


def calculate_positions(img_size, patch_size, min_patches=2):
    """Sliding window positions: minimum overlap, maximum coverage."""
    max_start = img_size - patch_size
    if max_start < 0:
        return None
    if max_start == 0:
        return [0]
    num_patches = max(min_patches, int(np.ceil(img_size / patch_size)))
    return np.linspace(0, max_start, num_patches).astype(int).tolist()


def sample_magnitude(spec):
    """Sample defect intensity magnitude from a yaml spec.

    Accepts scalar, [a, b], or [[a1, b1], [a2, b2], ...] (equal-prob mixture).
    """
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError("intensity_abs spec is empty")
        if len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
            return float(np.random.uniform(spec[0], spec[1]))
        if all(isinstance(r, (list, tuple)) and len(r) == 2
               and all(isinstance(v, (int, float)) for v in r) for r in spec):
            chosen = spec[np.random.randint(len(spec))]
            return float(np.random.uniform(chosen[0], chosen[1]))
    raise ValueError(f"Invalid intensity_abs spec: {spec!r}")


def _psf_pool_worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _psf_pool_worker_make_one(args):
    cfg, child_seed = args
    rng = np.random.default_rng(child_seed)
    raw, _ = generate_one(cfg, rng)
    cleaned = clean_connected_peak(raw, cfg.get("threshold_multiplier", 1.0))
    nz = np.argwhere(cleaned > 0)
    if len(nz) == 0:
        return None
    y0, x0 = nz.min(axis=0)
    y1, x1 = nz.max(axis=0)
    cropped = cleaned[y0:y1 + 1, x0:x1 + 1]
    if cropped.max() > 0:
        cropped = cropped / cropped.max()
    return cropped.astype(np.float32)


class PsfDefectPool:
    """Pre-generated pool of PSF defects, mirroring Background_Removal_Net."""

    def __init__(self, psf_cfgs, pool_size=1000, n_workers=4, master_seed=None):
        self.pools = []
        self.cfgs = list(psf_cfgs)
        n_workers = max(1, int(n_workers))
        print(f"Pre-generating PSF defect pool ({pool_size} per type, "
              f"n_workers={n_workers})...")
        for i, cfg in enumerate(psf_cfgs):
            self.pools.append(self._build_one_pool(cfg, i, pool_size, n_workers, master_seed))
        self.num_types = len(self.pools)

    def _build_one_pool(self, cfg, type_idx, pool_size, n_workers, master_seed):
        cfg_master = np.random.SeedSequence(
            master_seed,
            spawn_key=(type_idx,) if master_seed is not None else (),
        )
        batch = max(pool_size + pool_size // 5, pool_size + 16)
        seed_pool = list(cfg_master.spawn(batch))
        seed_used = batch
        max_attempts = pool_size * 10
        pool = []
        nonlocal_failures = [0]

        def _drain(results_iter, pbar):
            for defect in results_iter:
                if defect is not None:
                    pool.append(defect)
                    pbar.update(1)
                    if len(pool) >= pool_size:
                        return True
                else:
                    nonlocal_failures[0] += 1
            return False

        with tqdm(total=pool_size, desc=f"  Type {type_idx}", unit="psf") as pbar:
            if n_workers == 1:
                for seed in seed_pool:
                    defect = _psf_pool_worker_make_one((cfg, seed))
                    if defect is not None:
                        pool.append(defect)
                        pbar.update(1)
                        if len(pool) >= pool_size:
                            break
                    else:
                        nonlocal_failures[0] += 1
                while len(pool) < pool_size:
                    if seed_used >= max_attempts:
                        raise RuntimeError(
                            f"PSF config {type_idx}: too many failures "
                            f"({nonlocal_failures[0]} fail, {len(pool)} ok)")
                    extra = cfg_master.spawn(pool_size - len(pool))
                    seed_used += len(extra)
                    for seed in extra:
                        defect = _psf_pool_worker_make_one((cfg, seed))
                        if defect is not None:
                            pool.append(defect)
                            pbar.update(1)
                            if len(pool) >= pool_size:
                                break
                        else:
                            nonlocal_failures[0] += 1
            else:
                with mp.Pool(n_workers, initializer=_psf_pool_worker_init) as workers:
                    args = [(cfg, s) for s in seed_pool]
                    done = _drain(workers.imap_unordered(_psf_pool_worker_make_one, args), pbar)
                    while not done:
                        if seed_used >= max_attempts:
                            raise RuntimeError(
                                f"PSF config {type_idx}: too many failures "
                                f"({nonlocal_failures[0]} fail, {len(pool)} ok)")
                        deficit = pool_size - len(pool)
                        extra_count = max(deficit + deficit // 5, deficit + 8)
                        extra = cfg_master.spawn(extra_count)
                        seed_used += extra_count
                        args = [(cfg, s) for s in extra]
                        done = _drain(workers.imap_unordered(_psf_pool_worker_make_one, args), pbar)
        return pool

    def sample(self):
        cfg_idx = np.random.randint(self.num_types)
        defect_idx = np.random.randint(len(self.pools[cfg_idx]))
        return self.pools[cfg_idx][defect_idx], self.cfgs[cfg_idx]


def _list_local_images(root, img_format):
    if img_format == "png_jpg":
        files = glob(os.path.join(root, "*.png"))
        files.extend(glob(os.path.join(root, "*.jpg")))
    elif img_format == "tiff":
        files = glob(os.path.join(root, "*.tiff"))
        files.extend(glob(os.path.join(root, "*.tif")))
    else:
        files = []
    return sorted(files)


def _pair_paths(prev_paths, next_paths):
    """Pair prev/next by file basename (without extension)."""
    def key(p):
        return os.path.splitext(os.path.basename(p))[0]
    next_by_key = {key(p): p for p in next_paths}
    pairs = []
    missing = 0
    for p in prev_paths:
        k = key(p)
        if k in next_by_key:
            pairs.append((p, next_by_key[k]))
        else:
            missing += 1
    if missing:
        print(f"Warning: {missing} prev images had no matching next image (by basename)")
    return pairs


class PairedDataset(TorchDataset):
    """Paired prev/next station dataset for die-to-die anomaly training.

    Each image is 2-channel grayscale (target, ref). The model input
    concatenates both stations into 4 channels. GT mask labels the two
    positive-sample cases 00→10 and 01→10 (both satisfy next_T=1 AND
    next_R=0 AND prev_T=0).
    """

    def __init__(self, prev_path, next_path,
                 patch_size=(128, 128),
                 num_defects_range=(4, 10),
                 img_format="tiff",
                 cache_size=0,
                 defect_mode="gaussian",
                 psf_config_paths=None,
                 psf_pool_size=1000,
                 psf_pool_workers=4,
                 intensity_abs=(60, 80),
                 no_defect_prob=0.5):
        self.patch_size = patch_size
        self.num_defects_range = num_defects_range
        self.img_format = img_format
        self.cache_size = cache_size
        self.defect_mode = defect_mode
        self.intensity_abs = intensity_abs
        self.no_defect_prob = no_defect_prob

        if defect_mode == "psf":
            if not psf_config_paths:
                raise ValueError("psf_config_paths required for psf defect mode")
            psf_cfgs = [load_psf_config(p) for p in psf_config_paths]
            self.defect_pool = PsfDefectPool(
                psf_cfgs, pool_size=psf_pool_size, n_workers=psf_pool_workers)
            print(f"Defect mode: PSF ({len(psf_cfgs)} types: {psf_config_paths})")
        else:
            print("Defect mode: Gaussian")

        if num_defects_range[0] < len(ANCHORS):
            print(f"Warning: num_defects_range minimum {num_defects_range[0]} < "
                  f"{len(ANCHORS)} anchors. The forced four-anchor mechanism "
                  "will degrade when N < 4.")

        self.prev_is_s3 = prev_path.startswith("s3://")
        self.next_is_s3 = next_path.startswith("s3://")
        prev_paths = self._list_paths(prev_path, self.prev_is_s3)
        next_paths = self._list_paths(next_path, self.next_is_s3)
        if not prev_paths:
            raise ValueError(f"No prev images found in {prev_path}")
        if not next_paths:
            raise ValueError(f"No next images found in {next_path}")
        self.training_paths = _pair_paths(prev_paths, next_paths)
        if not self.training_paths:
            raise ValueError("No prev/next pairs matched by basename")
        print(f"Paired {len(self.training_paths)} prev/next images "
              f"(prev: {prev_path}, next: {next_path})")

        self._setup_cache()
        self._detect_and_display_image_info()
        self._setup_patch_positions()

    def _list_paths(self, root, is_s3):
        if is_s3:
            bucket, prefix = parse_s3_path(root)
            return list_s3_objects(bucket, prefix, self.img_format)
        if not os.path.exists(root):
            raise ValueError(f"Path does not exist: {root}")
        return _list_local_images(root, self.img_format)

    def _setup_cache(self):
        if self.cache_size > 0:
            self._load_image = lru_cache(maxsize=self.cache_size)(self._load_image_uncached)
        else:
            self._load_image = self._load_image_uncached

    def _get_s3_client(self):
        if not hasattr(self, "_s3_client"):
            import boto3
            endpoint_url = os.environ.get("S3_ENDPOINT_URL")
            self._s3_client = boto3.client("s3", endpoint_url=endpoint_url)
        return self._s3_client

    def _load_image_uncached(self, img_path):
        if img_path.startswith("s3://"):
            bucket, key = parse_s3_path(img_path)
            client = self._get_s3_client()
            response = client.get_object(Bucket=bucket, Key=key)
            data = response["Body"].read()
            if self.img_format == "tiff":
                return tifffile.imread(io.BytesIO(data))
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if self.img_format == "tiff":
            return tifffile.imread(img_path)
        return cv2.imread(img_path)

    def _detect_and_display_image_info(self):
        print("\n" + "=" * 60)
        print("Dataset Image Information")
        print("=" * 60)
        prev_path, next_path = self.training_paths[0]
        prev_img = ensure_2ch(ensure_hwc(self._load_image(prev_path)))
        next_img = ensure_2ch(ensure_hwc(self._load_image(next_path)))
        if prev_img is None or next_img is None:
            raise ValueError(f"Failed to load sample pair: {prev_path} / {next_path}")
        if prev_img.shape != next_img.shape:
            raise ValueError(
                f"Prev/next size mismatch: prev {prev_img.shape} vs next {next_img.shape}. "
                "MVP assumes pixel-aligned prev/next pairs.")
        if prev_img.ndim != 3 or prev_img.shape[2] != 2:
            raise ValueError(
                f"Expected 2-channel images (H, W, 2), got {prev_img.shape}. "
                "Hybrid pipeline requires 2 channels per station (target + 1 reference).")
        h, w = prev_img.shape[:2]
        self.detected_img_h = h
        self.detected_img_w = w
        print(f"Sample pair: {os.path.basename(prev_path)}")
        print(f"Shape: {prev_img.shape}, dtype: {prev_img.dtype}")
        print(f"Value range: prev [{prev_img.min():.2f}, {prev_img.max():.2f}], "
              f"next [{next_img.min():.2f}, {next_img.max():.2f}]")
        print(f"Image size: {h} x {w}")
        print("=" * 60 + "\n")

    def _setup_patch_positions(self):
        h, w = self.detected_img_h, self.detected_img_w
        self.y_positions = calculate_positions(h, self.patch_size[0])
        self.x_positions = calculate_positions(w, self.patch_size[1])
        if self.y_positions is None or self.x_positions is None:
            raise ValueError(f"Image size {h}x{w} smaller than patch size {self.patch_size}")
        self.patches_per_image = len(self.y_positions) * len(self.x_positions)
        self.total_patches = len(self.training_paths) * self.patches_per_image
        print(f"Patch positions - Y: {len(self.y_positions)} positions {self.y_positions}")
        print(f"Patch positions - X: {len(self.x_positions)} positions {self.x_positions}")
        print(f"Total patches: {self.total_patches} "
              f"({self.patches_per_image}/image x {len(self.training_paths)} images)")

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image
        y_idx = patch_idx // len(self.x_positions)
        x_idx = patch_idx % len(self.x_positions)

        prev_path, next_path = self.training_paths[img_idx]
        start_y = self.y_positions[y_idx]
        start_x = self.x_positions[x_idx]

        prev_image = self._load_image(prev_path)
        next_image = self._load_image(next_path)
        if prev_image is None or next_image is None:
            raise ValueError(f"Failed to load pair: {prev_path} / {next_path}")

        prev_image = ensure_2ch(ensure_hwc(prev_image)).astype(np.float32)
        next_image = ensure_2ch(ensure_hwc(next_image)).astype(np.float32)
        prev_image = self._normalize_to_0_255(prev_image)
        next_image = self._normalize_to_0_255(next_image)

        end_y = start_y + self.patch_size[0]
        end_x = start_x + self.patch_size[1]
        prev_patch = prev_image[start_y:end_y, start_x:end_x]
        next_patch = next_image[start_y:end_y, start_x:end_x]

        channels = {
            "prev_T": prev_patch[:, :, 0].copy(),
            "prev_R": prev_patch[:, :, 1].copy(),
            "next_T": next_patch[:, :, 0].copy(),
            "next_R": next_patch[:, :, 1].copy(),
        }
        channels, gt_mask = self.generate_paired_defects(channels)

        four_channel = np.stack([channels[k] for k in CHANNEL_ORDER], axis=0)
        four_channel_tensor = torch.from_numpy(four_channel).float() / 255.0
        gt_mask_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0)
        return {
            "paired_input": four_channel_tensor,
            "target_mask": gt_mask_tensor,
        }

    @staticmethod
    def _normalize_to_0_255(image):
        img_min = image.min()
        img_max = image.max()
        if img_min < 0 or img_max > 255:
            if img_max > img_min:
                return (image - img_min) / (img_max - img_min) * 255.0
            return np.zeros_like(image)
        return image

    def _create_one_defect(self, h, w):
        """Returns (local_defect_0to1, bounds, intensity) or (None, None, None)."""
        if self.defect_mode == "gaussian":
            magnitude = sample_magnitude(self.intensity_abs)
            intensity = magnitude if np.random.rand() < 0.5 else -magnitude
            margin = 5
            cx = np.random.randint(margin, w - margin)
            cy = np.random.randint(margin, h - margin)
            if np.random.rand() > 0.5:
                size, sigma = (3, 3), 1.3
            else:
                size, sigma = (3, 5), (1.0, 1.5)
            local_defect, bounds = create_local_gaussian_defect(
                center=(cx, cy), size=size, sigma=sigma,
                patch_shape=(h, w), patch_offset=(0, 0))
            return local_defect, bounds, intensity

        if self.defect_mode == "psf":
            cropped, cfg = self.defect_pool.sample()
            magnitude = sample_magnitude(cfg.get("intensity_abs", self.intensity_abs))
            intensity = magnitude if np.random.rand() < 0.5 else -magnitude
            dh, dw = cropped.shape
            margin = 2
            max_y = h - dh - margin
            max_x = w - dw - margin
            if max_y < margin or max_x < margin:
                return None, None, None
            y = np.random.randint(margin, max_y + 1)
            x = np.random.randint(margin, max_x + 1)
            bounds = (y, y + dh, x, x + dw)
            return cropped, bounds, intensity

        return None, None, None

    def _assign_cases(self, n_defects):
        """Force the four 'next-station T-only family' anchors, fill rest uniformly.

        The four anchors (00→10, 01→10, 10→10, 11→10) all look like T-only in
        the next station but have different GT and different prev_T values.
        Forcing all four into every patch prevents the model from learning a
        "prev has anything → output 0" shortcut, which would systematically
        misclassify 01→10 (prev_R has a defect but prev_T is clean → still 1).
        """
        if n_defects <= 0:
            return []
        if n_defects <= len(ANCHORS):
            return list(ANCHORS[:n_defects])
        cases = list(ANCHORS)
        extra = np.random.choice(CASE_NAMES, size=n_defects - len(ANCHORS),
                                 replace=True).tolist()
        cases.extend(extra)
        np.random.shuffle(cases)
        return cases

    def generate_paired_defects(self, channels):
        h, w = channels["prev_T"].shape

        if np.random.rand() < self.no_defect_prob:
            return channels, np.zeros((h, w), dtype=np.float32)

        n = np.random.randint(self.num_defects_range[0], self.num_defects_range[1] + 1)
        defects = []
        for _ in range(n):
            local_defect, bounds, intensity = self._create_one_defect(h, w)
            if local_defect is not None:
                defects.append((local_defect, bounds, intensity))

        if not defects:
            return channels, np.zeros((h, w), dtype=np.float32)

        cases = self._assign_cases(len(defects))
        gt_mask = np.zeros((h, w), dtype=np.float32)

        for (local_defect, bounds, intensity), case_name in zip(defects, cases):
            prev_pat, next_pat, is_gt = CASES[case_name]
            flags = list(prev_pat) + list(next_pat)
            for key, flag in zip(CHANNEL_ORDER, flags):
                if flag:
                    channels[key] = apply_local_defect_to_background(
                        channels[key], local_defect, bounds, intensity)
            if is_gt:
                local_mask = create_binary_mask(local_defect, threshold=0.1)
                ys, ye, xs, xe = bounds
                gt_mask[ys:ye, xs:xe] = np.maximum(gt_mask[ys:ye, xs:xe], local_mask)

        return channels, gt_mask
