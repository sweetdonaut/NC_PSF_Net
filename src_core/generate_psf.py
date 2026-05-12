"""
PSF Defect Generator
====================
Annular aperture + point source -> FFT -> noise -> PSF -> connected peak cleaning

Usage:
  python generate_psf.py --config defects/type1.yaml
"""

import argparse
import numpy as np
import os
import yaml
import scipy.fft as sfft
from scipy.ndimage import label


PARAM_NAMES = [
    "outer_r", "epsilon", "ellipticity", "ellip_angle",
    "square_eps", "h_stripe_w", "v_stripe_w", "h_outer_crop", "v_outer_crop",
    "defocus", "astig_x", "astig_y", "coma_x", "coma_y",
    "spherical", "trefoil_x", "trefoil_y",
    "brightness", "background", "gaussian_sigma",
]

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in cfg:
        val = cfg[key]
        if isinstance(val, list) and len(val) == 2 and all(isinstance(v, (int, float)) for v in val):
            cfg[key] = tuple(val)
    return cfg


def sample(rng, r):
    if isinstance(r, (list, tuple)):
        return r[0] if r[0] == r[1] else rng.uniform(r[0], r[1])
    return r


def _bin_down(arr, factor, offset_y, offset_x):
    """Sum factor×factor fine cells per sensor pixel, with sub-pixel offset.

    Models sensor pixel area integration: each sensor pixel reads the integral
    of the continuous light field over its area. Random offset ∈ [0, factor)²
    controls where the PSF center lands relative to the sensor pixel grid.

    Energy is conserved (np.roll wraps; PSF tail values at boundaries are
    negligible compared to peak).
    """
    if factor == 1:
        return arr
    NB = arr.shape[0] // factor
    oy = int(offset_y) % factor
    ox = int(offset_x) % factor
    rolled = np.roll(arr, shift=(-oy, -ox), axis=(0, 1))
    return rolled.reshape(NB, factor, NB, factor).sum(axis=(1, 3))


def _build_vector_pupil(mask, phase, x, y, outer_r, na, pol_type):
    """Build Richards-Wolf vector pupil components (Ux, Uy, Uz).

    Mirrors the vector-mode logic in psf-explorer-app/src/App.jsx. Pixels with
    sin(theta) >= 1 (outside the valid angular range for the given NA) are
    zeroed in the returned mask.
    """
    rho_pixel = np.sqrt(x ** 2 + y ** 2)
    sin_theta = (rho_pixel / outer_r) * na

    valid = (sin_theta < 1) & (mask > 0)
    mask = valid.astype(np.float64)

    sin_t = np.where(valid, sin_theta, 0.0)
    cos_t = np.where(valid, np.sqrt(np.maximum(0.0, 1.0 - sin_t ** 2)), 0.0)
    phi = np.arctan2(y, x)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    apod = np.sqrt(np.maximum(0.0, cos_t))

    inv2 = 1.0 / np.sqrt(2.0)
    zeros = np.zeros_like(phi)
    ones = np.ones_like(phi)
    if pol_type == "linX":
        px_re, px_im, py_re, py_im = ones, zeros, zeros, zeros
    elif pol_type == "linY":
        px_re, px_im, py_re, py_im = zeros, zeros, ones, zeros
    elif pol_type == "lin45":
        px_re, px_im, py_re, py_im = np.full_like(phi, inv2), zeros, np.full_like(phi, inv2), zeros
    elif pol_type == "circR":
        px_re, px_im, py_re, py_im = np.full_like(phi, inv2), zeros, zeros, np.full_like(phi, inv2)
    elif pol_type == "circL":
        px_re, px_im, py_re, py_im = np.full_like(phi, inv2), zeros, zeros, np.full_like(phi, -inv2)
    elif pol_type == "radial":
        px_re, px_im, py_re, py_im = cos_p, zeros, sin_p, zeros
    elif pol_type == "azimuthal":
        # WARNING: azimuthal polarization yields a doughnut-shaped PSF (zero
        # at center). The cleanup pipeline still runs, but the resulting
        # defect is a ring — needs further validation before training use.
        # Avoid this mode for now unless intentional.
        px_re, px_im, py_re, py_im = -sin_p, zeros, cos_p, zeros
    else:
        raise ValueError(
            f"Unknown pol_type: {pol_type!r}. Allowed: "
            "linX, linY, lin45, circR, circL, radial, azimuthal."
        )

    # Richards-Wolf rotation matrix (aplanatic lens). Ayx = Axy by symmetry.
    cos2p, sin2p, csp = cos_p * cos_p, sin_p * sin_p, cos_p * sin_p
    Axx = cos_t * cos2p + sin2p
    Axy = (cos_t - 1.0) * csp
    Ayy = cos_t * sin2p + cos2p
    Azx = -sin_t * cos_p
    Azy = -sin_t * sin_p

    ox_re = Axx * px_re + Axy * py_re
    ox_im = Axx * px_im + Axy * py_im
    oy_re = Axy * px_re + Ayy * py_re
    oy_im = Axy * px_im + Ayy * py_im
    oz_re = Azx * px_re + Azy * py_re
    oz_im = Azx * px_im + Azy * py_im

    e_re, e_im = np.cos(phase), np.sin(phase)
    a = apod * mask
    ux = a * (ox_re * e_re - ox_im * e_im) + 1j * a * (ox_re * e_im + ox_im * e_re)
    uy = a * (oy_re * e_re - oy_im * e_im) + 1j * a * (oy_re * e_im + oy_im * e_re)
    uz = a * (oz_re * e_re - oz_im * e_im) + 1j * a * (oz_re * e_im + oz_im * e_re)
    return ux, uy, uz, mask


def generate_one(cfg, rng):
    # pixel_oversample controls sensor sampling. 1 = original behavior (FFT
    # on psf_size grid). >1 runs FFT on (psf_size × oversample) fine grid
    # then sums oversample² fine cells per sensor pixel with a random
    # sub-pixel offset, modeling finite-pixel-grid integration.
    oversample = int(cfg.get("pixel_oversample", 1))
    N_sensor = cfg["psf_size"]
    N = N_sensor * oversample
    # float32 grid + complex64 FFT downstream: ~2x faster, max relative
    # error ~2e-7, fully absorbed by Poisson rounding + [0,1] normalization
    # (verified bit-identical training output).
    y, x = np.mgrid[-N//2:N//2, -N//2:N//2].astype(np.float32)

    outer_r = sample(rng, cfg["outer_r"])
    eps = sample(rng, cfg["epsilon"])
    ellip = sample(rng, cfg["ellipticity"])
    ellip_ang = sample(rng, cfg["ellip_angle"])
    square_eps = sample(rng, cfg.get("square_eps", 0))
    h_stripe_w = sample(rng, cfg.get("h_stripe_w", 0))
    v_stripe_w = sample(rng, cfg.get("v_stripe_w", 0))
    h_outer_crop = sample(rng, cfg.get("h_outer_crop", 0))
    v_outer_crop = sample(rng, cfg.get("v_outer_crop", 0))
    defocus = sample(rng, cfg["defocus"])
    astig_x = sample(rng, cfg["astig_x"])
    astig_y = sample(rng, cfg["astig_y"])
    coma_x = sample(rng, cfg["coma_x"])
    coma_y = sample(rng, cfg["coma_y"])
    sph = sample(rng, cfg["spherical"])
    tri_x = sample(rng, cfg["trefoil_x"])
    tri_y = sample(rng, cfg["trefoil_y"])
    brightness = sample(rng, cfg["brightness"])
    bg = sample(rng, cfg["background"])
    g_sig = sample(rng, cfg["gaussian_sigma"])

    # Annular mask
    cos_a, sin_a = np.cos(np.radians(ellip_ang)), np.sin(np.radians(ellip_ang))
    rx = (x * cos_a + y * sin_a) / (1 + ellip)
    ry = (-x * sin_a + y * cos_a) / (1 - ellip)
    r = np.sqrt(rx**2 + ry**2)
    mask = ((r <= outer_r) & (r >= outer_r * eps)).astype(np.float64)

    # Optional pupil obstructions (matches PSF Explorer web UI)
    if square_eps > 0:
        mask[(np.abs(x) <= outer_r * square_eps) & (np.abs(y) <= outer_r * square_eps)] = 0
    if h_stripe_w > 0:
        mask[np.abs(y) <= outer_r * h_stripe_w] = 0
    if v_stripe_w > 0:
        mask[np.abs(x) <= outer_r * v_stripe_w] = 0
    if h_outer_crop > 0:
        mask[np.abs(y) > outer_r * (1 - h_outer_crop)] = 0
    if v_outer_crop > 0:
        mask[np.abs(x) > outer_r * (1 - v_outer_crop)] = 0

    # Phase (Zernike)
    dx, dy = x / outer_r, y / outer_r
    rho2 = dx**2 + dy**2
    rho = np.sqrt(rho2)
    theta = np.arctan2(dy, dx)
    phase = (defocus * (2*rho2 - 1)
             + astig_x * rho2 * np.cos(2*theta)
             + astig_y * rho2 * np.sin(2*theta)
             + coma_x * (3*rho2 - 2) * rho * np.cos(theta)
             + coma_y * (3*rho2 - 2) * rho * np.sin(theta)
             + sph * (6*rho2**2 - 6*rho2 + 1)
             + tri_x * rho2 * rho * np.cos(3*theta)
             + tri_y * rho2 * rho * np.sin(3*theta))

    # PSF: scalar (single FFT) or Richards-Wolf vector mode (3 FFTs).
    # scipy.fft (workers=1) inside the worker is intentional — outer
    # multiprocess parallelism (PsfDefectPool) handles concurrency; opening
    # BLAS threads here would oversubscribe and slow things down.
    if cfg.get("vector_mode", False):
        na = sample(rng, cfg.get("na", 0.95))
        pol_type = cfg.get("pol_type", "linX")
        ux, uy, uz, mask = _build_vector_pupil(mask, phase, x, y, outer_r, na, pol_type)
        ux = ux.astype(np.complex64); uy = uy.astype(np.complex64); uz = uz.astype(np.complex64)
        Ix = np.abs(np.fft.fftshift(sfft.fft2(ux, workers=1))) ** 2
        Iy = np.abs(np.fft.fftshift(sfft.fft2(uy, workers=1))) ** 2
        Iz = np.abs(np.fft.fftshift(sfft.fft2(uz, workers=1))) ** 2
        psf = Ix + Iy + Iz
    else:
        pupil = (mask * np.exp(1j * phase)).astype(np.complex64)
        psf = np.abs(np.fft.fftshift(sfft.fft2(pupil, workers=1))) ** 2

    # Sensor sampling: bin fine grid → sensor grid with random sub-pixel offset.
    # Skipped when oversample=1 (psf is already on sensor grid).
    if oversample > 1:
        oy = int(rng.integers(0, oversample))
        ox = int(rng.integers(0, oversample))
        psf = _bin_down(psf, oversample, oy, ox)

    # Brightness + background + noise — on sensor grid (post-bin).
    # Order matters: bg is per-sensor-pixel dark current, Gaussian σ is
    # per-sensor-pixel read noise. Doing these on a fine grid then summing
    # would scale σ by factor (physically wrong).
    psf = psf / psf.sum() * brightness + bg
    if cfg.get("poisson_noise", True):
        psf = rng.poisson(np.maximum(0, psf)).astype(np.float64)
    if g_sig > 0:
        psf += rng.normal(0, g_sig, psf.shape)
    psf = np.maximum(0, psf)

    # Center crop on sensor grid
    c = cfg["crop_size"]
    s = N_sensor // 2 - c // 2
    cropped = psf[s:s+c, s:s+c]

    params = [outer_r, eps, ellip, ellip_ang,
              square_eps, h_stripe_w, v_stripe_w, h_outer_crop, v_outer_crop,
              defocus, astig_x, astig_y, coma_x, coma_y, sph, tri_x, tri_y,
              brightness, bg, g_sig]
    return cropped, params


def clean_connected_peak(img, threshold_multiplier=1.0):
    """Keep only the connected region around the brightest pixel."""
    mu, sigma = img.mean(), img.std()
    bg = np.median(img)
    peak_y, peak_x = np.unravel_index(img.argmax(), img.shape)

    binary = img > (mu + threshold_multiplier * sigma)
    labeled, _ = label(binary)
    peak_label = labeled[peak_y, peak_x]

    if peak_label == 0:
        return np.zeros_like(img)

    core_mask = (labeled == peak_label)
    cleaned = np.where(core_mask, img - bg, 0)
    return np.maximum(cleaned, 0).astype(np.float32)


def create_psf_defect(cfg):
    """Generate one PSF defect on the fly, cleaned and cropped to bounding box.
    Returns normalized 0-1 float32 array, or None if generation fails.
    """
    rng = np.random.default_rng()
    raw, _ = generate_one(cfg, rng)
    thr = cfg.get("threshold_multiplier", 1.0)
    cleaned = clean_connected_peak(raw, thr)

    nonzero = np.argwhere(cleaned > 0)
    if len(nonzero) == 0:
        return None

    y_min, x_min = nonzero.min(axis=0)
    y_max, x_max = nonzero.max(axis=0)
    cropped = cleaned[y_min:y_max+1, x_min:x_max+1]

    if cropped.max() > 0:
        cropped = cropped / cropped.max()

    return cropped.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    n = cfg["n_samples"]
    c = cfg["crop_size"]
    thr_mult = cfg.get("threshold_multiplier", 1.0)
    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)

    rng = np.random.default_rng(42)
    images = np.zeros((n, c, c), dtype=np.float32)
    params = np.zeros((n, len(PARAM_NAMES)), dtype=np.float32)

    print(f"Generating {n} PSFs (threshold_multiplier={thr_mult}) ...")
    for i in range(n):
        raw, params[i] = generate_one(cfg, rng)
        images[i] = clean_connected_peak(raw, thr_mult)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n}")

    np.save(f"{out}/psf_images.npy", images)
    np.save(f"{out}/psf_params.npy", params)
    with open(f"{out}/params_names.txt", "w") as f:
        f.write("\n".join(PARAM_NAMES))

    nonzero_counts = [(img > 0).sum() for img in images]
    print(f"\nDone!")
    print(f"  images: {out}/psf_images.npy  {images.shape}")
    print(f"  params: {out}/psf_params.npy  {params.shape}")
    print(f"  core size: mean={np.mean(nonzero_counts):.1f}px, range=[{np.min(nonzero_counts)}, {np.max(nonzero_counts)}]px")


if __name__ == "__main__":
    main()
