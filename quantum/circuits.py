"""
Quantum encoding circuits. Each takes the 8-D classical latent vector
(values in [0, 1], produced by the autoencoder's Sigmoid output) and
prepares a quantum state.

ENCODERS registry stores, per method:
  - n_qubits : how many qubits this scheme needs for an 8-D latent
  - prep     : function(latent, wires) that applies the state-prep gates

Design notes (read before editing):
  * basis        -> 8 qubits, 1 latent value -> 1 qubit (bit, via threshold)
  * angle        -> 8 qubits, 1 latent value -> 1 qubit (RY rotation)
  * dense_angle  -> 4 qubits, 2 latent values -> 1 qubit (RY then RZ)
  * iqp          -> 8 qubits, 1 latent value -> 1 qubit + pairwise ZZ terms
  * amplitude    -> 3 qubits (2**3 = 8), entire latent -> statevector amplitudes
Qubit counts intentionally differ per scheme -- comparing them side by side
on their *natural* qubit requirement (rather than forcing all to 8 qubits)
is itself part of the comparison story: amplitude encoding is drastically
more qubit-efficient for a fixed-size latent vector.
"""
import numpy as np
import pennylane as qml

LATENT_DIM = 8


def _basis_prep(latent, wires):
    bits = (np.asarray(latent) > 0.5).astype(int)
    qml.BasisEmbedding(bits, wires=wires)


def _angle_prep(latent, wires):
    angles = np.asarray(latent) * np.pi
    qml.AngleEmbedding(angles, wires=wires, rotation="Y")


def _dense_angle_prep(latent, wires):
    """Pack 2 latent features per qubit: RY(theta) sets polar angle,
    RZ(phi) sets azimuthal angle on the Bloch sphere."""
    latent = np.asarray(latent)
    for i, w in enumerate(wires):
        theta = latent[2 * i] * np.pi
        phi = latent[2 * i + 1] * np.pi
        qml.RY(theta, wires=w)
        qml.RZ(phi, wires=w)

def _entangled_angle_prep(latent, wires):
    """Same as Angle Encoding, but with a ring of CNOT gates added afterward
    to entangle neighboring qubits. Used as a controlled comparison against
    plain Angle Encoding, to test whether entanglement itself (not IQP's
    specific structure) is what causes reconstruction loss."""
    angles = np.asarray(latent) * np.pi
    qml.AngleEmbedding(angles, wires=wires, rotation="Y")
    n = len(wires)
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])



def _iqp_prep(latent, wires, n_repeats=2):
    angles = np.asarray(latent) * np.pi
    qml.IQPEmbedding(angles, wires=wires, n_repeats=n_repeats)


def _amplitude_prep(latent, wires):
    latent = np.asarray(latent, dtype=float)
    norm = np.linalg.norm(latent)
    vec = latent / (norm if norm > 0 else 1e-12)
    qml.AmplitudeEmbedding(vec, wires=wires, normalize=True)
    return norm


ENCODERS = {
    "basis":            {"n_qubits": 8, "prep": _basis_prep},
    "angle":            {"n_qubits": 8, "prep": _angle_prep},
    "dense_angle":      {"n_qubits": 4, "prep": _dense_angle_prep},
    "iqp":              {"n_qubits": 8, "prep": _iqp_prep},
    "amplitude":         {"n_qubits": 3, "prep": _amplitude_prep},
    "entangled_angle":  {"n_qubits": 8, "prep": _entangled_angle_prep},
}


def get_statevector(latent, method):
    """Return the full complex statevector after state prep.
    Only meaningful to decode directly for 'basis' and 'amplitude'
    (see quantum/decode.py for why angle/dense_angle/iqp instead use
    reduced density matrices)."""
    cfg = ENCODERS[method]
    n = cfg["n_qubits"]
    dev = qml.device("default.qubit", wires=n)
    extra = {}

    if method == "amplitude":
        norm = np.linalg.norm(np.asarray(latent, dtype=float))
        extra["norm"] = norm if norm > 0 else 1e-12

    @qml.qnode(dev)
    def circuit():
        cfg["prep"](latent, wires=range(n))
        return qml.state()

    state = np.array(circuit())
    return state, extra, n


def get_reduced_bloch_vectors(latent, method):
    """For each qubit, return (rx, ry, rz, purity) computed from the
    reduced single-qubit density matrix (partial trace over the rest
    of the register). This is the generic 'quantum latent vector'
    read-out used for angle / dense_angle / iqp, since it is well
    defined even when a qubit is entangled with the rest of the circuit
    (purity < 1 signals information has leaked into inter-qubit
    correlations and cannot be recovered from this qubit alone --
    this is exactly what happens for IQP)."""
    cfg = ENCODERS[method]
    n = cfg["n_qubits"]
    dev = qml.device("default.qubit", wires=n)

    results = []
    for w in range(n):
        @qml.qnode(dev)
        def circuit(w=w):
            cfg["prep"](latent, wires=range(n))
            return qml.density_matrix(wires=[w])

        rho = np.array(circuit())
        rx = 2 * np.real(rho[0, 1])
        ry = 2 * np.imag(rho[1, 0])
        rz = np.real(rho[0, 0] - rho[1, 1])
        purity = float(np.real(np.trace(rho @ rho)))
        results.append((rx, ry, rz, purity))
    return results
