"""
Classical autoencoder that compresses a flattened grayscale image into an
8-dimensional latent vector (matching the qubit budget used downstream).

Latent is bounded to [0, 1] via a final Sigmoid so it can be mapped directly
to rotation angles (theta = latent * pi) or normalized amplitudes without
any extra rescaling step in the quantum module.
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim=256, latent_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, latent_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=8, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, output_dim), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    """Full model used only for TRAINING. Encoder and Decoder are used
    separately at inference time, with the quantum pipeline sitting between
    them (Encoder -> quantum encode/decode -> Decoder)."""

    def __init__(self, input_dim=256, latent_dim=8):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
