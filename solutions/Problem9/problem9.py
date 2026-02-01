import numpy as np
import pennylane as qml

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import CXGate
from qiskit.synthesis import TwoQubitBasisDecomposer


def isTdg(op):
    if op.name in ("Adjoint(T)", "T†", "Tdg"):
        return True
    base = getattr(op, "base", None)
    return base is not None and getattr(base, "name", "") == "T"


def appendAllowed(oneQOps, op):
    name = op.name

    if len(op.wires) == 0:
        if name in ("GlobalPhase", "Identity", "I"):
            return
        if name.startswith("Adjoint(") and getattr(op, "base", None) is not None and len(op.base.wires) == 0:
            return
        raise ValueError(f"Got a wire-less op we don't handle: {name} ({op})")

    w = int(op.wires[0])

    if name == "Hadamard":
        oneQOps.append(("h", w))
        return
    if name == "T":
        oneQOps.append(("t", w))
        return
    if isTdg(op):
        oneQOps.append(("tdg", w))
        return

    if name in ("Identity", "I"):
        return

    # Z = T^4
    if name in ("PauliZ", "Z"):
        oneQOps.extend([("t", w), ("t", w), ("t", w), ("t", w)])
        return

    # X = H Z H = H T^4 H
    if name in ("PauliX", "X"):
        oneQOps.append(("h", w))
        oneQOps.extend([("t", w), ("t", w), ("t", w), ("t", w)])
        oneQOps.append(("h", w))
        return

    # Y = (global phase) X Z = (H Z H) Z
    if name in ("PauliY", "Y"):
        oneQOps.append(("h", w))
        oneQOps.extend([("t", w), ("t", w), ("t", w), ("t", w)])
        oneQOps.append(("h", w))
        oneQOps.extend([("t", w), ("t", w), ("t", w), ("t", w)])
        return

    # S = T^2 ; Sdg = (Tdg)^2
    if name == "S":
        oneQOps.extend([("t", w), ("t", w)])
        return
    if name in ("Adjoint(S)", "Sdg"):
        oneQOps.extend([("tdg", w), ("tdg", w)])
        return

    if name == "GlobalPhase":
        return

    raise ValueError(f"Unhandled 1-qubit op from gridsynth: {name} ({op})")


def oneQToHT(unitary1q, wire, eps):
    tape = qml.tape.QuantumScript([qml.QubitUnitary(unitary1q, wires=wire)], measurements=[])
    newTapes, _ = qml.clifford_t_decomposition(tape, epsilon=eps, method="gridsynth")
    decomp = newTapes[0]

    opsOut = []
    for op in decomp.operations:
        appendAllowed(opsOut, op)
    return opsOut


def circuitUnitaryFromHTCNOT(opList, nQubits=2):
    qc = QuantumCircuit(nQubits)
    for name, a, b in opList:
        if name == "h":
            qc.h(a)
        elif name == "t":
            qc.t(a)
        elif name == "tdg":
            qc.tdg(a)
        elif name == "cx":
            qc.cx(a, b)
        else:
            raise ValueError(f"Unknown gate tag: {name}")
    return Operator(qc).data, qc


def equalUpToGlobalPhase(A, B, atol=1e-6):
    # pick stable entry to estimate phase
    idx = np.unravel_index(np.argmax(np.abs(B)), B.shape)
    if np.abs(B[idx]) < 1e-14:
        return np.allclose(A, B, atol=atol)

    phase = A[idx] / B[idx]
    if np.abs(phase) < 1e-14:
        return False
    phase /= np.abs(phase)  # normalize to unit magnitude
    return np.allclose(A, phase * B, atol=atol)


def main(eps=1e-6, outQasm="structured2_htcx.qasm"):
    # Target 4x4 unitary
    U = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, -0.5 + 0.5j, 0.5 + 0.5j],
            [0, 1j, 0, 0],
            [0, 0, -0.5 + 0.5j, -0.5 - 0.5j],
        ],
        dtype=complex,
    )

    # Unitarity check
    eye = np.eye(4, dtype=complex)
    unitaryOk = np.allclose(U.conj().T @ U, eye, atol=1e-12)
    print("Unitary?", unitaryOk)
    if not unitaryOk:
        print("Max deviation from I:", np.max(np.abs(U.conj().T @ U - eye)))
        return

    # Exact 2-qubit decomposition into CX + 1q gates via Qiskit
    decomposer = TwoQubitBasisDecomposer(CXGate())
    qc2 = decomposer(Operator(U))

    # Convert to H/T/Tdg/CNOT gates
    finalOps = []  # ("h"/"t"/"tdg"/"cx", a, b)

    for ci in qc2.data:
        inst = ci.operation
        qargs = ci.qubits

        wires = [qc2.find_bit(q).index for q in qargs]

        name = inst.name.lower()

        if name in ("cx", "cnot"):
            finalOps.append(("cx", wires[0], wires[1]))
            continue

        if len(wires) == 1:
            mat = inst.to_matrix()
            oneQOps = oneQToHT(mat, wires[0], eps)
            for g, w in oneQOps:
                finalOps.append((g, w, -1))
            continue

        raise ValueError(f"Unexpected multi-qubit gate in intermediate circuit: {inst.name}")

    # Verify
    Uapprox, qcFinal = circuitUnitaryFromHTCNOT(finalOps, nQubits=2)
    ok = equalUpToGlobalPhase(Uapprox, U, atol=max(1e-12, eps * 10))
    maxdiff = np.max(np.abs(Uapprox - U))
    print("Matches up to global phase?", ok)
    print("Max abs entrywise diff (no phase fix):", maxdiff)
    print("Gate counts:", qcFinal.count_ops())

    # Export OpenQASM 2.0
    qasm = []
    qasm.append("OPENQASM 2.0;")
    qasm.append('include "qelib1.inc";')
    qasm.append("")
    qasm.append("qreg q[2];")
    qasm.append("")

    for name, a, b in finalOps:
        if name == "h":
            qasm.append(f"h q[{a}];")
        elif name == "t":
            qasm.append(f"t q[{a}];")
        elif name == "tdg":
            qasm.append(f"tdg q[{a}];")
        elif name == "cx":
            qasm.append(f"cx q[{a}],q[{b}];")
        else:
            raise ValueError("Internal error: unexpected gate tag")

    with open(outQasm, "w", encoding="utf-8") as f:
        f.write("\n".join(qasm) + "\n")
    print(f"Wrote QASM to: {outQasm}")


if __name__ == "__main__":
    main(eps=1e-4, outQasm="problem9.qasm")
