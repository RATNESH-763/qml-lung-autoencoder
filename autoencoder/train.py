"""
Trains the classical autoencoder. Run this once before the quantum pipeline.

Usage:
    python -m autoencoder.train --image_dir data/raw --epochs 30
"""
import argparse
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from autoencoder.model import Autoencoder
from autoencoder.dataset import LungXrayDataset


def train(image_dir, epochs=30, batch_size=16, lr=1e-3,
          save_path="autoencoder_weights.pt", size=16, latent_dim=8):

    dataset = LungXrayDataset(image_dir, size=size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(input_dim=size * size, latent_dim=latent_dim)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for flat, _ in loader:
            opt.zero_grad()
            recon, z = model(flat)
            loss = criterion(recon, flat)
            loss.backward()
            opt.step()
            total_loss += loss.item() * flat.size(0)
        avg = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} - MSE Loss: {avg:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved trained autoencoder weights to {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data/raw")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="autoencoder_weights.pt")
    args = parser.parse_args()

    train(args.image_dir, args.epochs, args.batch_size, args.lr,
          args.save_path, args.size, args.latent_dim)
