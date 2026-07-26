"""
Splits comparison_raw.csv results by image class (NORMAL vs PNEUMONIA)
based on filename, and prints/saves per-class summaries.

Usage:
    python class_split_analysis.py
"""
import pandas as pd

df = pd.read_csv("results/comparison_raw.csv")

def get_class(filename):
    name = filename.lower()
    if "bacteria" in name or "virus" in name or "pneumonia" in name:
        return "PNEUMONIA"
    else:
        return "NORMAL"

df["class"] = df["image"].apply(get_class)

numeric_cols = ["MSE", "PSNR", "SSIM", "Wasserstein", "Latent_MSE"]

summary = df.groupby(["method", "class"])[numeric_cols].mean()
print(summary)

summary.to_csv("results/comparison_by_class.csv")
print("\nSaved to results/comparison_by_class.csv")