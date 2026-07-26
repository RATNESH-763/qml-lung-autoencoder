# Quantum-Latent Image Reconstruction — Encoding Comparison

Pipeline:
```
Image -> Preprocessing -> Autoencoder Encoder -> 8-D Classical Latent
      -> Quantum Encoding Circuit (Basis / Angle / Dense Angle / IQP / Amplitude)
      -> Quantum Latent Vector
      -> Reverse Quantum Circuit -> Recovered Classical Latent
      -> Autoencoder Decoder -> Reconstructed Image
      -> Wasserstein Distance (+ MSE, PSNR, SSIM, Latent MSE)
```

## Setup
```bash
pip install -r requirements.txt
```
Place lung X-ray images (e.g. Kaggle "Chest X-Ray Images (Pneumonia)") in `data/raw/`.

## 1. Train the classical autoencoder
```bash
python -m autoencoder.train --image_dir data/raw --epochs 30 --save_path autoencoder_weights.pt
```
This produces `autoencoder_weights.pt`, containing both Encoder and Decoder
weights. The Encoder maps a 16x16 image to an 8-D latent vector in [0,1]
(Sigmoid output). The Decoder maps an 8-D latent vector back to a 16x16
image.

## 2. Run the pipeline on one image with one encoding
```bash
python -m pipeline.run_pipeline data/raw/some_image.png angle
```
Valid method names: `basis`, `angle`, `dense_angle`, `iqp`, `amplitude`.

## 3. Compare all 5 encodings across many images
```bash
python -m experiments.compare_encodings --image_dir data/raw --n_images 15
```
Produces `results/comparison_raw.csv` and `results/comparison_summary.csv`.

## Why qubit counts differ per method
| Method | Qubits used | Why |
|---|---|---|
| basis | 8 | 1 latent value -> 1 qubit (thresholded bit) |
| angle | 8 | 1 latent value -> 1 qubit (RY rotation) |
| dense_angle | 4 | 2 latent values -> 1 qubit (RY + RZ) |
| iqp | 8 | 1 latent value -> 1 qubit + pairwise ZZ entangling terms |
| amplitude | 3 | 2^3 = 8, entire latent fits directly in the amplitudes |

Keeping each method at its *natural* qubit requirement (rather than forcing
everything onto 8 qubits) is itself part of the comparison: it shows
amplitude encoding is drastically more qubit-efficient for a fixed-size
latent, at the cost of a harder-to-prepare state in general.

## What "Quantum Latent Vector" means here
- **basis / amplitude**: the full statevector (exact, ideal-simulator access).
- **angle**: per-qubit marginal probability of measuring |1>.
- **dense_angle / iqp**: per-qubit reduced density matrix (Bloch vector +
  purity), obtained via partial trace over the rest of the register. Purity
  < 1 means information has leaked into inter-qubit correlations and is
  *not* recoverable from that qubit's marginal alone — this is the expected,
  and reportable, behavior for IQP's entangling layers.

## Notes for the write-up
- The IQP prep circuit is diagonal (phase-only) after the initial Hadamards,
  so the computational-basis (Z) marginal probabilities carry **zero**
  information about the encoded values — only X/Y (phase-sensitive)
  observables do. This is a real, citable property of IQP-style circuits,
  not an implementation bug; it directly motivates their use in quantum
  kernel methods rather than for reconstruction.
- Wasserstein distance here is the 1D distance between flattened pixel-
  intensity distributions (`scipy.stats.wasserstein_distance`) — it ignores
  spatial arrangement. State this explicitly when reporting results.
