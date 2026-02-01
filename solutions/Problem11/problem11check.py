# Compare problem11.qasm to original 4-qubit diagonal unitary

import sys
import os
import math
import cmath
import numpy as np

pi = math.pi

PHASE_TARGET = {
    "0000": 0.0,
    "0001": pi,
    "0010": 5*pi/4,
    "0011": 7*pi/4,
    "0100": 5*pi/4,
    "0101": 7*pi/4,
    "0110": 3*pi/2,
    "0111": 3*pi/2,
    "1000": 5*pi/4,
    "1001": 7*pi/4,
    "1010": 3*pi/2,
    "1011": 3*pi/2,
    "1100": 3*pi/2,
    "1101": 3*pi/2,
    "1110": 7*pi/4,
    "1111": 5*pi/4,
}

def parse_qasm(path):
    ops = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if "//" in line:
                line = line.split("//", 1)[0].strip()
            if not line:
                continue

            low = line.lower()
            if low.startswith(("openqasm", "include", "qreg", "creg")):
                continue

            if line.endswith(";"):
                line = line[:-1].strip()
            low = line.lower()

            if low.startswith("cx "):
                rest = line[3:].strip()
                left, right = [x.strip() for x in rest.split(",")]
                c = int(left[left.find("[")+1:left.find("]")])
                t = int(right[right.find("[")+1:right.find("]")])
                ops.append(("cx", c, t))
            elif low.startswith("t "):
                q = int(line[line.find("[")+1:line.find("]")])
                ops.append(("t", q))
            elif low.startswith("tdg "):
                q = int(line[line.find("[")+1:line.find("]")])
                ops.append(("tdg", q))
            elif low.startswith("h "):
                q = int(line[line.find("[")+1:line.find("]")])
                ops.append(("h", q))
            else:
                raise ValueError(f"Unsupported/unknown line: {raw.rstrip()}")
    return ops

def apply_single_qubit(U, gate2, q, n=4):
    dim = 1 << n
    out = U.copy()
    stride = 1 << q
    block = stride << 1
    for base in range(0, dim, block):
        for off in range(stride):
            i0 = base + off
            i1 = i0 + stride
            a = out[i0, :].copy()
            b = out[i1, :].copy()
            out[i0, :] = gate2[0, 0] * a + gate2[0, 1] * b
            out[i1, :] = gate2[1, 0] * a + gate2[1, 1] * b
    return out

def apply_cx(U, c, t, n=4):
    dim = 1 << n
    out = U.copy()
    cmask = 1 << c
    tmask = 1 << t
    for x in range(dim):
        if x & cmask:
            y = x ^ tmask
            if y > x:
                out[[x, y], :] = out[[y, x], :]
    return out

def unitary_from_ops(ops, n=4):
    dim = 1 << n
    U = np.eye(dim, dtype=complex)

    H = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    T = np.array([[1, 0], [0, cmath.exp(1j * pi/4)]], dtype=complex)
    TDG = np.array([[1, 0], [0, cmath.exp(-1j * pi/4)]], dtype=complex)

    for op in ops:
        if op[0] == "h":
            U = apply_single_qubit(U, H, op[1], n)
        elif op[0] == "t":
            U = apply_single_qubit(U, T, op[1], n)
        elif op[0] == "tdg":
            U = apply_single_qubit(U, TDG, op[1], n)
        elif op[0] == "cx":
            U = apply_cx(U, op[1], op[2], n)
        else:
            raise ValueError(f"Unknown op: {op}")
    return U

def key_little_endian(i, n=4):
    # q[0] is LSB, so key is x0x1x2x3
    return "".join(str((i >> k) & 1) for k in range(n))

def key_big_endian(i, n=4):
    # key is q[n-1]...q[0]
    return "".join(str((i >> k) & 1) for k in reversed(range(n)))

def target_unitary(endian, n=4):
    keys = []
    for i in range(1 << n):
        if endian == "little":
            k = key_little_endian(i, n)
        else:
            k = key_big_endian(i, n)
        keys.append(k)
    diag = np.array([cmath.exp(1j * PHASE_TARGET[k]) for k in keys], dtype=complex)
    return np.diag(diag)

def max_diff_up_to_global_phase(U, V):
    # align using |0000> element
    g = U[0, 0] / V[0, 0]
    return float(np.max(np.abs(U - g * V)))

def main():
    path = sys.argv[1] if len(sys.argv) >= 2 else "problem11.qasm"

    if not os.path.exists(path):
        print(f"Could not find QASM file: {path}")
        sys.exit(2)

    ops = parse_qasm(path)

    counts = {"h": 0, "t": 0, "tdg": 0, "cx": 0}
    for op in ops:
        counts[op[0]] += 1

    Uqasm = unitary_from_ops(ops, n=4)

    off = Uqasm.copy()
    np.fill_diagonal(off, 0)
    offdiag_max = float(np.max(np.abs(off)))

    Utgt_l = target_unitary("little", n=4)
    Utgt_b = target_unitary("big", n=4)

    diff_l = max_diff_up_to_global_phase(Utgt_l, Uqasm)
    diff_b = max_diff_up_to_global_phase(Utgt_b, Uqasm)

    print("File:", path)
    print("Gate counts:", counts)
    print("Max |off-diagonal|:", offdiag_max)
    print("Diff vs target (little-endian keys x0x1x2x3):", diff_l)
    print("Diff vs target (big-endian keys q3q2q1q0):", diff_b)

    best = min(diff_l, diff_b)
    which = "little" if diff_l <= diff_b else "big"
    print("Best match endian:", which)
    print("MATCH?", best < 1e-9)

if __name__ == "__main__":
    main()
