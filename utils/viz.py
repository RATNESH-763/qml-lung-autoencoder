"""
Plotting helpers for the final report / notebook.
"""
import matplotlib.pyplot as plt
import pandas as pd


def plot_reconstructions(orig_img, recon_dict, save_path=None):
    """recon_dict: {method_name: reconstructed_image_uint8}"""
    methods = list(recon_dict.keys())
    n = len(methods) + 1
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))

    axes[0].imshow(orig_img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, method in zip(axes[1:], methods):
        ax.imshow(recon_dict[method], cmap="gray")
        ax.set_title(method)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_metric_bars(csv_path, metric="Wasserstein", save_path=None):
    df = pd.read_csv(csv_path)
    summary = df.groupby("method")[metric].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["method"], summary["mean"], yerr=summary["std"], capsize=4)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by Encoding Method")
    plt.xticks(rotation=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
