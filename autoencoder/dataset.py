"""
Loads images from a folder, resizes to (size x size), normalizes to [0, 1],
and returns both the flattened vector (model input) and the 2D array
(for visualization / reconstruction comparisons).
"""
import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class LungXrayDataset(Dataset):
    def __init__(self, image_dir, size=16, limit=None):
        self.paths = []
        for root, _dirs, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.paths.append(os.path.join(root, f))
        self.paths.sort()
        if limit:
            self.paths = self.paths[:limit]
        if len(self.paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        flat = img.flatten()
        return flat, img
