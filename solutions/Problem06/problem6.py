import math
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript

pi = math.pi

# Detect adjoints in PL
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

# Synthesize RZ via gridsynth
def rzToAllowedOps(angle, wire, eps, debug_skip=False):
    """
    Decompose a single RZ(angle) on "wire" into only:
      ("h", q), ("t", q), ("tdg", q)
    plus exact rewrites of Z/S/Sdg/X/Y into the same set.

    Robustness:
      - Skip wire-less ops like GlobalPhase (no wires)
      - Skip Identity / I / PauliI (no-op)
    """
    tape = QuantumScript([qml.RZ(angle, wires=wire)], measurements=[])
    newTapes, _ = qml.clifford_t_decomposition(
        tape, epsilon=eps, method="gridsynth"
    )
    decompOps = newTapes[0].operations

    out = []
    for op in decompOps:
        if len(op.wires) == 0:
            if debug_skip:
                print("Skipping wire-less op:", op.name)
            continue

        q = int(op.wires[0])
        name = op.name

        if name in ("Identity", "I", "PauliI"):
            if debug_skip:
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

# Compile circuit for exp(i t (XX + ZI + IZ)) with
# 2nd-order Suzuki step for r Trotter steps
def compileH3(t=pi / 7, r=1, eps=1e-3, debug_skip=False):
    alpha = t / r

    rzCache = {}

    def cacheRz(angle, wire):
        key = float(np.round(angle, 15))
        rzCache[(key, wire)] = rzToAllowedOps(angle, wire, eps, debug_skip=debug_skip)

    cacheRz(-alpha, 0)
    cacheRz(-alpha, 1)
    cacheRz(-2 * alpha, 1)

    fullOps = []
    for _ in range(r):
        stepOps = []

        def addCachedRz(angle, wire):
            key = float(np.round(angle, 15))
            stepOps.extend(rzCache[(key, wire)])

        addCachedRz(-alpha, 0)

        addCachedRz(-alpha, 1)

        # exp(i*alpha XX) = (H⊗H) exp(i*alpha ZZ) (H⊗H)
        # exp(i*alpha ZZ) = CNOT RZ(-2alpha) CNOT   (RZ on target q1)
        addGate(stepOps, "h", 0)
        addGate(stepOps, "h", 1)
        addCx(stepOps, 0, 1)
        addCachedRz(-2 * alpha, 1)
        addCx(stepOps, 0, 1)
        addGate(stepOps, "h", 0)
        addGate(stepOps, "h", 1)

        # half IZ
        addCachedRz(-alpha, 1)

        # half ZI
        addCachedRz(-alpha, 0)

        fullOps.extend(stepOps)

    return fullOps

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

# Build unitary from ops and compare to target
def unitaryFromOps(ops):
    I = np.eye(2, dtype=complex)
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    Tdg = np.conjugate(T).T

    # 2-qubit CNOT 0->1
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
    # H3 = XX + ZI + IZ
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    H3 = np.kron(X, X) + np.kron(Z, I) + np.kron(I, Z)

    # exp(i t H3) using eigendecomposition (Hermitian)
    evals, evecs = np.linalg.eigh(H3)
    phases = np.exp(1j * t * evals)
    return (evecs * phases) @ np.conjugate(evecs).T

def maxDiffUpToGlobalPhase(U, V):
    # phase = arg(trace(V U†))
    tr = np.trace(V @ np.conjugate(U).T)
    phase = (1.0 + 0j) if abs(tr) < 1e-16 else (tr / abs(tr))
    diff = V - phase * U
    return np.max(np.abs(diff))


if __name__ == "__main__":

    t = pi / 7
    r = 8
    eps = 1e-5

    ops = compileH3(t=t, r=r, eps=eps, debug_skip=False)
    qasmText = emitQasm(ops, nQubits=2)

    outPath = "problem6.qasm"
    with open(outPath, "w", encoding="utf-8") as f:
        f.write(qasmText)

    print(f"Wrote {outPath}")
    print(f"Total ops: {len(ops)}")

    # Verify numerically (2 qubits)
    Uapprox = unitaryFromOps(ops)
    Utarget = targetUnitary(t=t)

    err = maxDiffUpToGlobalPhase(Uapprox, Utarget)
    print(f"Max |Δ| up to global phase: {err:.6e}")
