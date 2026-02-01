from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps


qc = QuantumCircuit(2)


# CY = (I ⊗ S) CNOT (I ⊗ S†)
# S = T^2, S† = (T†)^2

qc.tdg(1)
qc.tdg(1)
qc.cx(0, 1)
qc.t(1)
qc.t(1)


qasm = dumps(qc)

with open("task1.qasm", "w") as f:
    f.write(qasm)
