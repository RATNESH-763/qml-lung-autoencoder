"""
Implements the "Reverse Quantum Circuit -> Recovered Classical Latent" step
of the pipeline, for each encoding scheme.

Every function here is the analytic inverse of the corresponding prep
function in circuits.py. This is the mathematical equivalent of physically
uncomputing the circuit and reading out the register -- in an ideal,
noiseless simulator, applying the exact analytic inverse and applying the
adjoint circuit give identical results.
"""
import numpy as np
from quantum.circuits import get_statevector, get_reduced_bloch_vectors, ENCODERS

LATENT_DIM = 8


def _decode_basis(state, n, latent_dim):
    probs = np.abs(state) ** 2
    idx = int(np.argmax(probs))               # deterministic: prob=1 at the encoded bitstring
    bits = [(idx >> (n - 1 - i)) & 1 for i in range(n)]
    return np.array(bits[:latent_dim], dtype=float)


def _decode_angle(state, n, latent_dim):
    """Per qubit, recover theta from the marginal probability of measuring |1>,
    then invert theta = latent * pi. NOTE: RY encoding is not injective over
    [0, 2*pi) -- arcsin only returns theta in [0, pi], so this recovery is
    exact for the [0, pi] range we encoded into, but would be ambiguous
    outside it. This is intentional and worth stating in the report."""
    probs = np.abs(state) ** 2
    recon = []
    for i in range(n):
        p1 = sum(p for idx, p in enumerate(probs) if (idx >> (n - 1 - i)) & 1 == 1)
        theta = 2 * np.arcsin(np.sqrt(np.clip(p1, 0, 1)))
        recon.append(theta / np.pi)
    return np.array(recon[:latent_dim])


def _decode_dense_angle(latent_in, method, latent_dim):
    """Uses reduced-qubit Bloch vectors. Since dense_angle has NO entangling
    gates, each qubit's reduced state is pure (purity == 1) and (theta, phi)
    are recovered exactly via spherical coordinates: rz=cos(theta),
    rx=sin(theta)cos(phi), ry=sin(theta)sin(phi)."""
    bloch = get_reduced_bloch_vectors(latent_in, method)
    recon = []
    for (rx, ry, rz, purity) in bloch:
        theta = np.arccos(np.clip(rz, -1, 1))
        phi = np.arctan2(ry, rx)
        phi = phi if phi >= 0 else phi + np.pi
        recon.append(theta / np.pi)
        recon.append(phi / np.pi)
    return np.array(recon[:latent_dim])


def _decode_iqp(latent_in, method, latent_dim):
    """KEY RESEARCH FINDING: the IQP prep circuit (H, then only diagonal
    RZ / ZZ phase gates) never changes computational-basis probabilities,
    so measuring <Z_i> carries *zero* information about the encoded value.
    All information sits in the phase, visible only in <X_i>/<Y_i> -- and
    even there, the entangling ZZ terms mix each qubit's phase with its
    neighbours', so the reduced single-qubit state becomes MIXED
    (purity < 1) and only partial information is recoverable from that
    qubit alone. We report both the (approximate) angle estimate and the
    purity, since purity is itself evidence of how much information became
    inaccessible without a joint (multi-qubit) measurement."""
    bloch = get_reduced_bloch_vectors(latent_in, method)
    recon, purities = [], []
    for (rx, ry, rz, purity) in bloch:
        angle_est = np.arctan2(ry, rx)
        angle_est = angle_est if angle_est >= 0 else angle_est + 2 * np.pi
        recon.append(angle_est / np.pi)
        purities.append(purity)
    return np.array(recon[:latent_dim]), np.array(purities)


def _decode_amplitude(state, latent_dim, norm):
    """Amplitude encoding of an 8-D vector needs only 3 qubits (2**3 = 8),
    so no padding is needed and recovery is exact up to floating point
    error: divide out the L2 norm that was applied during normalization."""
    amps = np.real(state[:latent_dim])
    return amps * norm


def reconstruct_latent(latent, method, latent_dim=LATENT_DIM):
    """Full round trip: classical latent -> quantum encode -> quantum
    latent vector -> reverse -> recovered classical latent.
    Returns (recovered_latent, diagnostics_dict)."""
    latent = np.asarray(latent, dtype=float)
    diagnostics = {}

    if method == "basis":
        state, _, n = get_statevector(latent, method)
        recon = _decode_basis(state, n, latent_dim)

    elif method == "angle":
        state, _, n = get_statevector(latent, method)
        recon = _decode_angle(state, n, latent_dim)

    elif method == "dense_angle":
        recon = _decode_dense_angle(latent, method, latent_dim)

    elif method == "iqp":
        recon, purities = _decode_iqp(latent, method, latent_dim)
        diagnostics["mean_purity"] = float(np.mean(purities))

    elif method == "amplitude":
        state, extra, n = get_statevector(latent, method)
        recon = _decode_amplitude(state, latent_dim, extra["norm"])

    else:
        raise ValueError(f"Unknown encoding method: {method}")

    recon = np.clip(recon, 0.0, 1.0)
    return recon, diagnostics
