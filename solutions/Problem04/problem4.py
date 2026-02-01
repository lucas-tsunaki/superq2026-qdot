import math
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript

pi = math.pi

# Detect adjoints in PennyLane across versions
def isTdg(op):
    if op.name in ("Adjoint(T)", "T†", "Tdg"):
        return True
    base = getattr(op, "base", None)
    return base is not None and getattr(base, "name", "") == "T"

def isSdg(op):
    if op.name in ("Adjoint(S)", "S†", "Sdg"):
        return True
    base = getattr(op, "base", None)
    return base is not None and getattr(base, "name", "") == "S"

# Only h/t/tdg/cx allowed
# op tuple formats: ("h", q) ("t", q) ("tdg", q) ("cx", c, t)
def addGate(ops, name, q):
    ops.append((name, int(q)))

def addCx(ops, c, t):
    ops.append(("cx", int(c), int(t)))

def addS(ops, q):
    # S = T^2
    addGate(ops, "t", q)
    addGate(ops, "t", q)

def addSdg(ops, q):
    # S† = (T†)^2
    addGate(ops, "tdg", q)
    addGate(ops, "tdg", q)

def addZ(ops, q):
    # Z = T^4
    addGate(ops, "t", q)
    addGate(ops, "t", q)
    addGate(ops, "t", q)
    addGate(ops, "t", q)

def addX(ops, q):
    # X = H Z H
    addGate(ops, "h", q)
    addZ(ops, q)
    addGate(ops, "h", q)

def addY(ops, q):
    # Y = S X S†
    addS(ops, q)
    addX(ops, q)
    addSdg(ops, q)

# Create RZ via gridsynth and rewrite with only H/T/Tdg
def rzToAllowedOps(angle, wire, eps, debugSkip=False):
    tape = QuantumScript([qml.RZ(angle, wires=wire)], measurements=[])
    newTapes, _ = qml.clifford_t_decomposition(tape, epsilon=eps, method="gridsynth")
    decompOps = newTapes[0].operations

    out = []
    for op in decompOps:
        if len(op.wires) == 0:
            if debugSkip:
                print("Skipping wire-less op:", op.name)
            continue

        q = int(op.wires[0])
        name = op.name

        if name in ("Identity", "I", "PauliI"):
            if debugSkip:
                print(f"Skipping identity on wire {q}: {op}")
            continue

        if name in ("Hadamard", "H"):
            addGate(out, "h", q)
        elif name == "T":
            addGate(out, "t", q)
        elif isTdg(op):
            addGate(out, "tdg", q)
        elif name in ("PauliZ", "Z"):
            addZ(out, q)
        elif name in ("S",):
            addS(out, q)
        elif isSdg(op):
            addSdg(out, q)
        elif name in ("PauliX", "X"):
            addX(out, q)
        elif name in ("PauliY", "Y"):
            addY(out, q)
        else:
            raise ValueError(f"Unhandled op in RZ synthesis output: name={name}, op={op}")

    return out

#   RZ(theta) = exp(-i theta Z/2); exp(i t ZZ) = CNOT · RZ(-2t) · CNOT  (RZ on target qubit 1)
#   XX term: exp(i t XX) = (H⊗H) exp(i t ZZ) (H⊗H)
#   YY term: exp(i t YY) = (U⊗U) exp(i t ZZ) (U⊗U)†
#   where (U⊗U) = (S H) on each qubit, and (U⊗U)† = (H S†) on each qubit.

def compileH1(t=pi / 7, eps=1e-5, debugSkip=False):
    # Cache RZ needed for ZZ blocks
    rzCache = {}
    angleKey = float(np.round(-2.0 * t, 15))
    rzCache[angleKey] = rzToAllowedOps(-2.0 * t, wire=1, eps=eps, debugSkip=debugSkip)

    def addCachedRz(stepOps):
        stepOps.extend(rzCache[angleKey])

    ops = []

    # exp(i t XX)
    addGate(ops, "h", 0)
    addGate(ops, "h", 1)

    addCx(ops, 0, 1)
    addCachedRz(ops) # RZ(-2t) on qubit 1
    addCx(ops, 0, 1)

    addGate(ops, "h", 0)
    addGate(ops, "h", 1)

    # exp(i t YY)
    # Pre: (S H) on each qubit
    addS(ops, 0); addGate(ops, "h", 0)
    addS(ops, 1); addGate(ops, "h", 1)

    addCx(ops, 0, 1)
    addCachedRz(ops) # RZ(-2t) on qubit 1
    addCx(ops, 0, 1)

    # Post: (S H)† = (H S†) on each qubit
    addGate(ops, "h", 0); addSdg(ops, 0)
    addGate(ops, "h", 1); addSdg(ops, 1)

    return ops

# Write to QASM
def emitQasm(ops, nQubits=2):
    lines = []
    lines.append("OPENQASM 2.0;")
    lines.append('include "qelib1.inc";')
    lines.append(f"qreg q[{nQubits}];")

    for op in ops:
        if op[0] == "cx":
            _, c, t = op
            lines.append(f"cx q[{c}],q[{t}];")
        else:
            name, q = op
            lines.append(f"{name} q[{q}];")

    return "\n".join(lines)

# Compare result to original
def unitaryFromOps(ops):
    I = np.eye(2, dtype=complex)
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    Tdg = np.conjugate(T).T

    cx01 = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]], dtype=complex
    )

    def kron2(a, b):
        return np.kron(a, b)

    def gate2(name, q):
        if name == "h":
            g = H
        elif name == "t":
            g = T
        elif name == "tdg":
            g = Tdg
        else:
            raise ValueError(f"Unknown single-qubit gate {name}")

        if q == 0:
            return kron2(g, I)
        if q == 1:
            return kron2(I, g)
        raise ValueError("Only 2 qubits supported in verifier")

    U = np.eye(4, dtype=complex)
    for op in ops:
        if op[0] == "cx":
            U = cx01 @ U
        else:
            U = gate2(op[0], op[1]) @ U
    return U

def targetUnitary(t=pi / 7):
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)

    H1 = np.kron(X, X) + np.kron(Y, Y)

    evals, evecs = np.linalg.eigh(H1)
    phases = np.exp(1j * t * evals)
    return (evecs * phases) @ np.conjugate(evecs).T

def maxDiffUpToGlobalPhase(U, V):
    tr = np.trace(V @ np.conjugate(U).T)
    phase = (1.0 + 0j) if abs(tr) < 1e-16 else (tr / abs(tr))
    diff = V - phase * U
    return np.max(np.abs(diff))

if __name__ == "__main__":
    t = pi / 7
    eps = 1e-6

    ops = compileH1(t=t, eps=eps, debugSkip=False)
    qasmText = emitQasm(ops, nQubits=2)

    outPath = "problem4.qasm"
    with open(outPath, "w", encoding="utf-8") as f:
        f.write(qasmText)

    print(f"Wrote {outPath}")
    print(f"Total ops: {len(ops)}")

    Uapprox = unitaryFromOps(ops)
    Utarget = targetUnitary(t=t)
    err = maxDiffUpToGlobalPhase(Uapprox, Utarget)
    print(f"Max |Δ| up to global phase: {err:.6e}")
