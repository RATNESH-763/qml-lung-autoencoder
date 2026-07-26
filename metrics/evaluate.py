"""
Image and latent-space comparison metrics.
"""
import numpy as np
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def mse(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def compute_psnr(orig_img_uint8, recon_img_uint8):
    return float(psnr(orig_img_uint8, recon_img_uint8, data_range=255))


def compute_ssim(orig_img_uint8, recon_img_uint8):
    return float(ssim(orig_img_uint8, recon_img_uint8, data_range=255))


def compute_wasserstein(orig_img_uint8, recon_img_uint8):
    """1D Wasserstein distance between the flattened pixel-intensity
    distributions of the original and reconstructed images.
    NOTE: this compares intensity *distributions*, not pixel positions --
    it will not penalize a reconstruction that has the right histogram of
    gray levels but the wrong spatial arrangement. Report this explicitly."""
    a = orig_img_uint8.flatten().astype(np.float64)
    b = recon_img_uint8.flatten().astype(np.float64)
    return float(wasserstein_distance(a, b))


def latent_mse(orig_latent, recon_latent):
    return mse(np.asarray(orig_latent), np.asarray(recon_latent))


def evaluate_all(orig_img_uint8, recon_img_uint8, orig_latent=None, recon_latent=None):
    results = {
        "MSE": mse(orig_img_uint8, recon_img_uint8),
        "PSNR": compute_psnr(orig_img_uint8, recon_img_uint8),
        "SSIM": compute_ssim(orig_img_uint8, recon_img_uint8),
        "Wasserstein": compute_wasserstein(orig_img_uint8, recon_img_uint8),
    }
    if orig_latent is not None and recon_latent is not None:
        results["Latent_MSE"] = latent_mse(orig_latent, recon_latent)
    return results
