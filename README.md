# Quantum-Latent Image Reconstruction — Encoding Comparison

A proof-of-concept pipeline comparing 5 quantum data-encoding schemes on chest X-ray images (Kaggle Chest X-Ray Pneumonia dataset), evaluating how well each preserves information through an encode → quantum circuit → decode round trip.

## Pipeline

Image → Preprocessing (16x16 grayscale)
→ Autoencoder Encoder → 8-D Classical Latent
→ Quantum Encoding Circuit (Basis / Angle / Dense Angle / IQP / Amplitude)
→ Quantum Latent Vector
→ Reverse Quantum Circuit → Recovered Classical Latent
→ Autoencoder Decoder → Reconstructed Image
→ Metrics: Wasserstein Distance, MSE, PSNR, SSIM, Latent MSE


## Key Result

Evaluated on the full Kaggle test set (~624 images):

| Method | MSE | PSNR | SSIM | Wasserstein | Latent MSE |
|---|---|---|---|---|---|
| Amplitude | 338.8 | 23.44 | 0.896 | 5.43 | ~0 |
| Angle | 338.8 | 23.44 | 0.896 | 5.43 | ~0 |
| Dense Angle | 338.8 | 23.44 | 0.896 | 5.43 | ~0 |
| Basis | 6233.6 | 10.68 | 0.618 | 64.65 | 0.126 |
| IQP | 6146.8 | 10.88 | 0.502 | 60.77 | 0.216 |

**Finding:** Encodings that map each latent feature to an independent, unentangled qubit (Angle, Dense Angle, Amplitude) achieve near-lossless reconstruction. Basis Encoding loses information through crude quantization (rounding to a single bit). IQP Encoding loses information because its entangling gates scatter data across qubits — confirmed by measuring purity ≈ 0.52 on the reduced single-qubit states (1.0 = fully recoverable, lower = information leaked into inter-qubit correlations inaccessible to single-qubit read-out).

## Repository Structure
autoencoder/ - classical encoder/decoder (PyTorch) and training script
quantum/ - 5 quantum encoding circuits + reverse/decode logic (PennyLane)
metrics/ - MSE, PSNR, SSIM, Wasserstein distance, latent MSE
pipeline/ - single-image end-to-end runner
experiments/ - full-dataset comparison across all 5 encodings
utils/ - plotting (bar charts, reconstruction grids)
results/ - saved CSVs and plots from the full test-set run
autoencoder_weights.pt - trained model weights (no retraining needed)

## Why qubit counts differ per method

| Method | MSE | PSNR | SSIM | Wasserstein | Latent MSE |
|---|---|---|---|---|---|
| Amplitude | 338.8 | 23.44 | 0.896 | 5.43 | ~0 |
| Angle | 338.8 | 23.44 | 0.896 | 5.43 | ~0 |
| Dense Angle | 338.8 | 23.44 | 0.896 | 5.43 | ~0 |
| Basis | 6233.6 | 10.68 | 0.618 | 64.65 | 0.126 |
| IQP | 6146.8 | 10.88 | 0.502 | 60.77 | 0.216 |

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

## How to Reproduce

```bash
pip install -r requirements.txt
python -m autoencoder.train --image_dir data/raw --epochs 30
python -m experiments.compare_encodings --image_dir data/test --n_images 1000
```

Dataset images are not included in this repo — download the Kaggle "Chest X-Ray Images (Pneumonia)" dataset and place train/test folders under `data/raw/` and `data/test/` respectively.

## Limitations

- Runs on an ideal, noiseless simulator (PennyLane default.qubit) — real quantum hardware would require measurement/tomography, which is left as future work.
- Metrics compare against a 16x16 downsampled version of the original image, not the full-resolution clinical X-ray — this isolates the encoding/decoding pipeline's fidelity from the separate information loss incurred by initial resizing.
- Image-level reconstruction quality (MSE/PSNR/SSIM) is influenced by both the classical autoencoder's decoder and the quantum encoding step; Latent MSE is used to isolate the quantum contribution specifically.
