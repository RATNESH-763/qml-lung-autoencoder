"""
Full pipeline for a single image and a single encoding method:

Image -> Preprocessing -> Autoencoder Encoder -> 8-D Classical Latent
      -> Quantum Encoding Circuit -> Quantum Latent Vector
      -> Reverse Quantum Circuit -> Recovered Classical Latent
      -> Autoencoder Decoder -> Reconstructed Image
      -> Metrics (MSE, PSNR, SSIM, Wasserstein, Latent MSE)
"""
import numpy as np
import torch
import cv2

from autoencoder.model import Encoder, Decoder
from quantum.decode import reconstruct_latent
from metrics.evaluate import evaluate_all

IMG_SIZE = 16
LATENT_DIM = 8


def load_trained_autoencoder(weights_path, input_dim=IMG_SIZE * IMG_SIZE, latent_dim=LATENT_DIM):
    from autoencoder.model import Autoencoder
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model.encoder, model.decoder


def preprocess_image(path, size=IMG_SIZE):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img_norm = img.astype(np.float32) / 255.0
    return img, img_norm


def run_single(image_path, encoder, decoder, method):
    """Returns dict with original image, reconstructed image, latents, and metrics."""
    orig_img_uint8, img_norm = preprocess_image(image_path)
    flat = torch.tensor(img_norm.flatten(), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        latent = encoder(flat).squeeze(0).numpy()          # (8,) classical latent

    recon_latent, diagnostics = reconstruct_latent(latent, method, latent_dim=LATENT_DIM)

    with torch.no_grad():
        recon_flat = decoder(torch.tensor(recon_latent, dtype=torch.float32).unsqueeze(0))
        recon_flat = recon_flat.squeeze(0).numpy()

    recon_img_uint8 = np.clip(recon_flat.reshape(IMG_SIZE, IMG_SIZE) * 255, 0, 255).astype(np.uint8)

    metrics = evaluate_all(orig_img_uint8, recon_img_uint8, latent, recon_latent)
    metrics.update({f"diag_{k}": v for k, v in diagnostics.items()})

    return {
        "method": method,
        "orig_img": orig_img_uint8,
        "recon_img": recon_img_uint8,
        "orig_latent": latent,
        "recon_latent": recon_latent,
        "metrics": metrics,
    }


if __name__ == "__main__":
    import sys
    weights_path = "autoencoder_weights.pt"
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.png"
    method = sys.argv[2] if len(sys.argv) > 2 else "angle"

    encoder, decoder = load_trained_autoencoder(weights_path)
    result = run_single(image_path, encoder, decoder, method)
    print(f"Method: {method}")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.6f}")
