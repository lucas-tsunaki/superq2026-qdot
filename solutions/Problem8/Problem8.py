from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps

qc = QuantumCircuit(2)


# QFT2
# H on q0
# controlled-S (control q0 -> target q1)
# H on q1
# SWAP(q0, q1)
# Controlled-S decomposition CS = (T⊗T) CX (I⊗Tdg) CX


qc.h(0)

qc.t(0)
qc.t(1)
qc.cx(0, 1)
qc.tdg(1)
qc.cx(0, 1)

qc.h(1)

qc.cx(0, 1)
qc.cx(1, 0)
qc.cx(0, 1)

qasm = dumps(qc)

with open("task8.qasm", "w", encoding="utf-8") as f:
    f.write(qasm)


