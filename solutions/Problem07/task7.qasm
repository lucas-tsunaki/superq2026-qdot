// JUST A TEST, unsure how to optimise

OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];

// State preparation: random_statevector(4, seed=42)
// Target: complex superposition of all 4 basis states

// Using two-qubit unitary decomposition:
// U = (A⊗B) CNOT (C⊗D) CNOT (E⊗F)
// Where each letter is a single-qubit unitary (Rz·Ry·Rz)

// Layer 1: Initial rotations
h q[0];
t q[0];
t q[0];
h q[0];
t q[0];
h q[0];
tdg q[0];
h q[0];

h q[1];
tdg q[1];
tdg q[1];
h q[1];
t q[1];
h q[1];

// First entangling CNOT
cx q[0], q[1];

// Layer 2: Mid-circuit rotations
h q[0];
tdg q[0];
h q[0];
t q[0];
t q[0];
h q[0];

h q[1];
t q[1];
h q[1];
tdg q[1];
tdg q[1];
h q[1];

// Second entangling CNOT
cx q[0], q[1];

// Layer 3: Final rotations
h q[0];
t q[0];
tdg q[0];
tdg q[0];
h q[0];

h q[1];
tdg q[1];
h q[1];
t q[1];
t q[1];
h q[1];
