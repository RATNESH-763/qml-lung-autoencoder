"""
Runs the full pipeline across all 5 encoding methods and a set of sample
images, and produces:
  - results/comparison_raw.csv       (one row per image x method)
  - results/comparison_summary.csv   (mean +/- std per method)

Usage:
    python -m experiments.compare_encodings --image_dir data/raw --n_images 15
"""
import os
import argparse
import pandas as pd

from pipeline.run_pipeline import load_trained_autoencoder, run_single

METHODS = ["basis", "angle", "dense_angle", "iqp", "amplitude"]


def list_images(image_dir, n_images):
    paths = []
    for root, _dirs, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, f))
    paths.sort()
    return paths[:n_images]


def run_comparison(image_dir, weights_path, n_images=15, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    encoder, decoder = load_trained_autoencoder(weights_path)
    image_paths = list_images(image_dir, n_images)

    rows = []
    for path in image_paths:
        for method in METHODS:
            result = run_single(path, encoder, decoder, method)
            row = {"image": os.path.basename(path), "method": method}
            row.update(result["metrics"])
            rows.append(row)
            print(f"{os.path.basename(path):30s} | {method:12s} | "
                  f"MSE={row['MSE']:.2f} PSNR={row['PSNR']:.2f} "
                  f"SSIM={row['SSIM']:.3f} Wasserstein={row['Wasserstein']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "comparison_raw.csv"), index=False)

    numeric_cols = [c for c in df.columns if c not in ("image", "method")]
    summary = df.groupby("method")[numeric_cols].agg(["mean", "std"])
    summary.to_csv(os.path.join(out_dir, "comparison_summary.csv"))

    print("\n=== Summary (mean across images) ===")
    print(df.groupby("method")[["MSE", "PSNR", "SSIM", "Wasserstein", "Latent_MSE"]].mean())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data/raw")
    parser.add_argument("--weights_path", type=str, default="autoencoder_weights.pt")
    parser.add_argument("--n_images", type=int, default=15)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    run_comparison(args.image_dir, args.weights_path, args.n_images, args.out_dir)
