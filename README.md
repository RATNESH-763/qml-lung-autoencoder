# Quantum-Latent Image Reconstruction — Encoding Comparison

A proof-of-concept pipeline comparing 6 quantum data-encoding schemes on chest X-ray images (Kaggle Chest X-Ray Pneumonia dataset), evaluating how well each preserves information through an encode → quantum circuit → decode round trip.

## Pipeline
Image → Preprocessing (16x16 grayscale)
→ Autoencoder Encoder → 8-D Classical Latent
→ Quantum Encoding Circuit (Basis / Angle / Dense Angle / IQP / Amplitude / Entangled Angle)
→ Quantum Latent Vector
→ Reverse Quantum Circuit → Recovered Classical Latent
→ Autoencoder Decoder → Reconstructed Image
→ Metrics: Wasserstein Distance, MSE, PSNR, SSIM, Latent MSE

## Key Result

Evaluated on the full Kaggle test set (~624 images):

| Method | MSE | PSNR | SSIM | Wasserstein | Latent MSE | Mean Purity |
|---|---|---|---|---|---|---|
| Amplitude | 338.8 | 23.44 | 0.896 | 5.43 | ~0 | 0.78–0.94 (entangled, but read out globally) |
| Angle | 338.8 | 23.44 | 0.896 | 5.43 | ~0 | 1.0 (no entanglement) |
| Dense Angle | 338.8 | 23.44 | 0.896 | 5.43 | ~0 | 1.0 (no entanglement) |
| Entangled Angle | 5060.5 | 11.56 | 0.583 | 60.73 | 0.035 | ~0.59 |
| IQP | 6146.8 | 10.88 | 0.502 | 60.77 | 0.216 | ~0.52 |
| Basis | 6233.6 | 10.68 | 0.618 | 64.65 | 0.126 | 1.0 (cannot entangle — see below) |

**Finding:** Encodings that map each latent feature to an independent, unentangled qubit (Angle, Dense Angle) achieve near-lossless reconstruction. Basis Encoding loses information through crude quantization (rounding to a single bit), not entanglement. IQP Encoding loses information because its entangling gates scatter data across qubits, confirmed by measuring purity ≈ 0.52 on reduced single-qubit states (1.0 = fully recoverable, lower = information leaked into inter-qubit correlations inaccessible to single-qubit read-out).

**Controlled experiment (Entangled Angle):** taking plain Angle Encoding and adding only a ring of CNOT gates afterward reproduced IQP-level reconstruction loss (Wasserstein 60.73 vs 60.77, purity ~0.59 vs ~0.52) — directly demonstrating that entanglement itself, not IQP's specific gate structure, is the mechanism responsible for information loss under single-qubit read-out.

**Why Amplitude Encoding is entangled yet still lossless:** its state-prep circuit uses multi-qubit controlled rotations that generically entangle qubits (purity 0.78–0.94, not 1.0) — yet it reconstructs perfectly because it is read out *globally* (the full statevector at once, not per-qubit). This isolates the real variable: entanglement alone doesn't cause reconstruction loss — combining entanglement with a *local, per-qubit* read-out does.

**Why "entangled Basis Encoding" isn't possible:** adding a CNOT ring to Basis Encoding was tested and produces zero entanglement (purity stays exactly 1.0), because CNOT only entangles a qubit already in superposition, and Basis Encoding (built from plain X gates) never creates superposition in the first place. Superposition is a necessary precondition for entanglement.

## Repository Structure
autoencoder/ - classical encoder/decoder (PyTorch) and training script
quantum/ - 6 quantum encoding circuits + reverse/decode logic (PennyLane)
metrics/ - MSE, PSNR, SSIM, Wasserstein distance, latent MSE
pipeline/ - single-image end-to-end runner
experiments/ - full-dataset comparison across all 6 encodings
utils/ - plotting (bar charts, reconstruction grids)
results/ - saved CSVs and plots from the full test-set run
autoencoder_weights.pt - trained model weights (no retraining needed)

## Why qubit counts differ per method
| Method | Qubits used | Why |
|---|---|---|
| basis | 8 | 1 latent value -> 1 qubit (thresholded bit) |
| angle | 8 | 1 latent value -> 1 qubit (RY rotation) |
| dense_angle | 4 | 2 latent values -> 1 qubit (RY + RZ) |
| iqp | 8 | 1 latent value -> 1 qubit + pairwise ZZ entangling terms |
| amplitude | 3 | 2^3 = 8, entire latent fits directly in the amplitudes |
| entangled_angle | 8 | same as angle, plus a ring of CNOT gates (controlled entanglement experiment) |

Keeping each method at its *natural* qubit requirement (rather than forcing everything onto 8 qubits) is itself part of the comparison: it shows amplitude encoding is drastically more qubit-efficient for a fixed-size latent, at the cost of a harder-to-prepare state in general.

## What "Quantum Latent Vector" means here
- **basis / amplitude**: the full statevector (exact, ideal-simulator access).
- **angle**: per-qubit marginal probability of measuring |1>.
- **dense_angle / iqp / entangled_angle**: per-qubit reduced density matrix (Bloch vector + purity), obtained via partial trace over the rest of the register. Purity < 1 means information has leaked into inter-qubit correlations and is *not* recoverable from that qubit's marginal alone.

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
- Non-entangling encodings (Basis, Angle, Dense Angle) achieve the best reconstruction fidelity, but are classically simulable product-state preparations that do not exploit any genuinely quantum computational resource. This suggests reconstruction fidelity and "quantumness" are, in this context, opposing objectives rather than complementary ones — entangling encodings like IQP are more relevant to tasks such as classification, where the goal is a rich feature space rather than exact recoverability.