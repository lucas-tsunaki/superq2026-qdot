import itertools
import math
from collections import defaultdict
import numpy as np

pi = math.pi

# Phase table (in radians)
phase_table = {
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

NQ = 4

def bits_from_string(s):
    return [int(c) for c in s]

def parity(xbits, subset):
    p = 0
    for i in subset:
        p ^= xbits[i]
    return p

def all_nonempty_subsets(n):
    subs = []
    for r in range(1, n + 1):
        for comb in itertools.combinations(range(n), r):
            subs.append(tuple(comb))
    return subs

def to_Z8_phase_vector(phase_table):
    keys = sorted(phase_table.keys())
    xs = [bits_from_string(k) for k in keys]
    # f(x) = phi(x) / (pi/4) mod 8
    fx = []
    for k in keys:
        v = phase_table[k] / (pi/4)
        fx.append(int(round(v)) % 8)
    return keys, np.array(xs, dtype=int), np.array(fx, dtype=int)

def build_parity_matrix(xs, subsets):
    m = xs.shape[0]
    n = len(subsets)
    A = np.zeros((m, n), dtype=int)
    for row in range(m):
        for col, S in enumerate(subsets):
            A[row, col] = parity(xs[row], S)
    return A

def verify_diag(keys, phase_table, fhat):
    target = np.array([np.exp(1j * phase_table[k]) for k in keys])
    pred = np.array([np.exp(1j * (pi/4) * int(v)) for v in fhat])

    # global phase align using |0000>
    g = target[0] / pred[0] if abs(pred[0]) > 0 else 1.0
    pred2 = pred * g
    return float(np.max(np.abs(target - pred2)))

def greedy_order_by_symdiff(control_sets):
    remaining = list(control_sets)
    cur = frozenset()
    out = []
    while remaining:
        best_i = None
        best_cost = 10**9
        for i, s in enumerate(remaining):
            cost = len(cur.symmetric_difference(s))
            if cost < best_cost:
                best_cost = cost
                best_i = i
        nxt = remaining.pop(best_i)
        out.append(nxt)
        cur = nxt
    return out

def schedule_parity_walk(terms, n_qubits):
    """
    Identified XOR parities that need a T gate
    For each parity, pick qubit to store parity
    Only 4 qubits, so try reasonable choices and pick one that uses fewest CNOTs
    """
    best_plan = None
    best_cnot = 10**9

    # For each term, possible targets are elements of S
    candidates = [list(S) for S in terms]

    """
    For parity term, choose qubit to hold parity when applying T gate
    Check every possibility and keep one that uses the fewest CNOTs
    """
    for targets in itertools.product(*candidates):
        groups = defaultdict(list)
        for S, t in zip(terms, targets):
            controls = frozenset(q for q in S if q != t)
            groups[t].append(controls)

        plan = {}
        cnot = 0
        for t in range(n_qubits):
            if t not in groups:
                continue
            ordered = greedy_order_by_symdiff(groups[t])
            # toggles: empty->first + symdiff steps + last->empty
            cur = frozenset()
            for s in ordered:
                cnot += len(cur.symmetric_difference(s))
                cur = s
            cnot += len(cur)
            plan[t] = ordered

        if cnot < best_cnot:
            best_cnot = cnot
            best_plan = plan

    return best_plan, best_cnot

def emit_qasm(plan, n_qubits):
    lines = []
    lines.append("OPENQASM 2.0;")
    lines.append('include "qelib1.inc";')
    lines.append("")
    lines.append(f"qreg q[{n_qubits}];")
    lines.append("")

    cnot_count = 0
    t_count = 0

    for t in range(n_qubits):
        if t not in plan:
            continue

        cur = set()

        def toggle(ctrl):
            nonlocal cnot_count
            lines.append(f"cx q[{ctrl}], q[{t}];")
            cnot_count += 1

        for desired in plan[t]:
            desired = set(desired)
            diff = cur.symmetric_difference(desired)
            for ctrl in sorted(diff):
                toggle(ctrl)
                if ctrl in cur:
                    cur.remove(ctrl)
                else:
                    cur.add(ctrl)

            # coefficient is 1 => apply T
            lines.append(f"t q[{t}];")
            t_count += 1

        # uncompute
        for ctrl in sorted(cur):
            toggle(ctrl)

        lines.append("")

    return "\n".join(lines), {"cnot": cnot_count, "t": t_count}

def main():
    keys, xs, fx = to_Z8_phase_vector(phase_table)
    subsets = all_nonempty_subsets(NQ)
    A = build_parity_matrix(xs, subsets)

    # EXACT search over coefficients in {0,1} for each parity term
    best_mask = None
    best_t = 10**9

    for mask in range(1 << len(subsets)):
        # Skip if already worse than best by popcount
        pop = mask.bit_count()
        if pop >= best_t:
            continue

        # compute fhat = sum selected parity columns (mod 8)
        # since coefficients are 0/1, this is A @ sel mod 8
        sel = np.array([(mask >> i) & 1 for i in range(len(subsets))], dtype=int)
        fhat = (A @ sel) % 8

        if np.array_equal(fhat, fx):
            best_mask = mask
            best_t = pop

    if best_mask is None:
        raise RuntimeError("No exact {0,1} parity solution found. (Unexpected for this instance.)")

    chosen_terms = [subsets[i] for i in range(len(subsets)) if (best_mask >> i) & 1]

    # Verify
    sel = np.array([(best_mask >> i) & 1 for i in range(len(subsets))], dtype=int)
    fhat = (A @ sel) % 8
    max_diff = verify_diag(keys, phase_table, fhat)

    print("Found exact phase polynomial with T-count:", best_t)
    print("Max diagonal entrywise diff (after global phase align):", max_diff)
    print("Chosen parity terms:")
    for S in chosen_terms:
        print("  ", S)

    # Reduce CNOTs
    plan, est_cnot = schedule_parity_walk(chosen_terms, NQ)
    qasm, counts = emit_qasm(plan, NQ)

    print("Estimated CNOT toggles (parity-walk):", est_cnot)
    print("Emitted counts:", counts)

    out_file = "problem11.qasm"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(qasm)
    print("Wrote:", out_file)

if __name__ == "__main__":
    main()
