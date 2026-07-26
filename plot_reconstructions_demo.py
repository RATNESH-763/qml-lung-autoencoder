"""
Shows the original image side-by-side with its reconstruction under
all 5 encoding methods, saved as a PNG.

Usage:
    python plot_reconstructions_demo.py path/to/image.jpeg
"""
import sys
from pipeline.run_pipeline import load_trained_autoencoder, run_single
from utils.viz import plot_reconstructions

METHODS = ["basis", "angle", "dense_angle", "iqp", "amplitude"]

image_path = sys.argv[1] if len(sys.argv) > 1 else "data/test/NORMAL/IM-0001-0001.jpeg"

encoder, decoder = load_trained_autoencoder("autoencoder_weights.pt")

orig_img = None
recon_dict = {}

for method in METHODS:
    result = run_single(image_path, encoder, decoder, method)
    orig_img = result["orig_img"]
    recon_dict[method] = result["recon_img"]

plot_reconstructions(orig_img, recon_dict, save_path="results/reconstruction_grid.png")
print("Saved results/reconstruction_grid.png")