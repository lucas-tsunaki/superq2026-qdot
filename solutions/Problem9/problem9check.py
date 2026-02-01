import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

def global_phase_align(A, B):
    """
    Find phase e^{iφ} that aligns B to A
    Uses phase = Tr(A B†) / |Tr(A B†)|
    """
    inner = np.trace(A @ B.conj().T)
    if abs(inner) < 1e-15:
        # fallback: use max-magnitude entry
        idx = np.unravel_index(np.argmax(np.abs(B)), B.shape)
        inner = A[idx] * np.conj(B[idx])
        if abs(inner) < 1e-15:
            return 1.0 + 0.0j
    return inner / abs(inner)

def allclose_up_to_global_phase(A, B, atol=1e-6):
    phase = global_phase_align(A, B)
    return np.allclose(A, phase * B, atol=atol, rtol=0.0)

# Load QASM unitary
qc = QuantumCircuit.from_qasm_file("problem9.qasm")
U_qasm = Operator(qc).data

# Target unitary
U_target = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, -0.5 + 0.5j, 0.5 + 0.5j],
        [0, 1j, 0, 0],
        [0, 0, -0.5 + 0.5j, -0.5 - 0.5j],
    ],
    dtype=complex,
)

phase = global_phase_align(U_qasm, U_target)
err_max = np.max(np.abs(U_qasm - phase * U_target))

print("Gate counts:", qc.count_ops())
print("Allclose up to global phase?", allclose_up_to_global_phase(U_qasm, U_target, atol=1e-4))
print("Max abs diff after global-phase fix:", err_max)
