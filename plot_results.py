"""
Generates bar chart comparisons across the 5 encoding methods,
saved as PNG files in results/.

Usage:
    python plot_results.py
"""
from utils.viz import plot_metric_bars

metrics_to_plot = ["MSE", "PSNR", "SSIM", "Wasserstein", "Latent_MSE"]

for metric in metrics_to_plot:
    save_path = f"results/{metric.lower()}_bar.png"
    plot_metric_bars("results/comparison_raw.csv", metric=metric, save_path=save_path)
    print(f"Saved {save_path}")