import numpy as np
from scipy.linalg import svd as scipy_svd


class CliffordTPlusCompiler:

    def __init__(self, target_unitary, max_single_qubit_t=7):
        self.target = target_unitary
        self.max_t = max_single_qubit_t

        self.hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self.t_gate = np.diag([1, np.exp(1j * np.pi / 4)])
        self.t_dagger = np.diag([1, np.exp(-1j * np.pi / 4)])
        self.identity = np.eye(2, dtype=complex)

        self.cnot_gate = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)

        self.sq_library = self._build_single_qubit_library()

    def _build_single_qubit_library(self):
        from collections import deque

        library = []
        seen_unitaries = {}

        queue = deque([{
            'unitary': self.identity.copy(),
            'gates': [],
            't_cost': 0,
            'depth': 0
        }])

        max_depth = 20

        while queue:
            current = queue.popleft()

            if current['depth'] > max_depth:
                continue
            if current['t_cost'] > self.max_t:
                continue

            key = self._hash_unitary(current['unitary'])

            if key in seen_unitaries:
                if seen_unitaries[key] <= current['t_cost']:
                    continue

            seen_unitaries[key] = current['t_cost']
            library.append({
                'unitary': current['unitary'].copy(),
                'gates': current['gates'].copy(),
                't_cost': current['t_cost']
            })

            gates_to_try = [
                ('h', self.hadamard, 0),
                ('t', self.t_gate, 1),
                ('tdg', self.t_dagger, 1)
            ]

            for gate_name, gate_matrix, t_increment in gates_to_try:
                new_unitary = gate_matrix @ current['unitary']
                new_gates = current['gates'] + [gate_name]
                new_t_cost = current['t_cost'] + t_increment
                new_depth = current['depth'] + 1

                queue.append({
                    'unitary': new_unitary,
                    'gates': new_gates,
                    't_cost': new_t_cost,
                    'depth': new_depth
                })

        library.sort(key=lambda x: (x['t_cost'], len(x['gates'])))
        print(f"Built library with {len(library)} single-qubit circuits")
        return library

    def _hash_unitary(self, U, precision=1e-9):
        det = np.linalg.det(U)
        if abs(det) > 1e-12:
            phase = np.exp(-0.5j * np.angle(det))
            U_normalized = phase * U
        else:
            U_normalized = U

        flat = U_normalized.flatten()
        rounded = np.round(flat / precision).astype(np.int64)
        return tuple(rounded)

    def _tensor_product(self, A, B):
        return np.kron(A, B)

    def _construct_circuit(self, config):
        result = self._tensor_product(
            self.sq_library[config[0][0]]['unitary'],
            self.sq_library[config[0][1]]['unitary']
        )

        result = self.cnot_gate @ result

        result = self._tensor_product(
            self.sq_library[config[1][0]]['unitary'],
            self.sq_library[config[1][1]]['unitary']
        ) @ result

        result = self.cnot_gate @ result

        result = self._tensor_product(
            self.sq_library[config[2][0]]['unitary'],
            self.sq_library[config[2][1]]['unitary']
        ) @ result

        result = self.cnot_gate @ result

        result = self._tensor_product(
            self.sq_library[config[3][0]]['unitary'],
            self.sq_library[config[3][1]]['unitary']
        ) @ result

        return result

    def _evaluate_cost(self, config):
        circuit_unitary = self._construct_circuit(config)

        t_count = sum(
            self.sq_library[config[layer][qubit]]['t_cost']
            for layer in range(4)
            for qubit in range(2)
        )

        distance = self._operator_distance_with_phase(circuit_unitary, self.target)

        return t_count, distance

    def _operator_distance_with_phase(self, U, V):
        trace = np.trace(np.conj(U.T) @ V)

        if abs(trace) < 1e-15:
            optimal_phase = 0.0
        else:
            optimal_phase = np.angle(trace)

        diff = U - np.exp(1j * optimal_phase) * V
        singular_values = scipy_svd(diff, compute_uv=False)

        return float(singular_values[0])

    def optimize_simulated_annealing(
        self,
        initial_temp=1.0,
        cooling_rate=0.95,
        iterations_per_temp=100,
        num_temps=50,
        random_seed=42
    ):
        rng = np.random.RandomState(random_seed)

        lib_size = len(self.sq_library)
        low_t_cutoff = min(lib_size, lib_size // 4)

        current_config = [
            (rng.randint(0, low_t_cutoff), rng.randint(0, low_t_cutoff))
            for _ in range(4)
        ]

        current_t, current_dist = self._evaluate_cost(current_config)

        best_config = current_config.copy()
        best_t = current_t
        best_dist = current_dist

        temperature = initial_temp

        print("Starting simulated annealing optimization...")

        for _ in range(num_temps):
            for _ in range(iterations_per_temp):
                layer = rng.randint(0, 4)
                qubit = rng.randint(0, 2)

                if rng.random() < 0.7:
                    new_idx = rng.randint(0, low_t_cutoff)
                else:
                    new_idx = rng.randint(0, lib_size)

                neighbor_config = [list(c) for c in current_config]
                neighbor_config[layer][qubit] = new_idx
                neighbor_config = [tuple(c) for c in neighbor_config]

                neighbor_t, neighbor_dist = self._evaluate_cost(neighbor_config)

                if neighbor_t < current_t:
                    accept = True
                elif neighbor_t == current_t:
                    if neighbor_dist < current_dist:
                        accept = True
                    else:
                        delta_E = neighbor_dist - current_dist
                        accept = rng.random() < np.exp(-delta_E / temperature)
                else:
                    delta_T = neighbor_t - current_t
                    accept = rng.random() < np.exp(-delta_T * 5.0 / temperature)

                if accept:
                    current_config = neighbor_config
                    current_t = neighbor_t
                    current_dist = neighbor_dist

                    if (current_t < best_t) or (current_t == best_t and current_dist < best_dist):
                        best_config = current_config
                        best_t = current_t
                        best_dist = current_dist
                        print(f"  [Temp={temperature:.4f}] New best: T={best_t}, dist={best_dist:.6e}")

            temperature *= cooling_rate

        print(f"\nOptimization complete!")
        print(f"Best T-count: {best_t}")
        print(f"Best distance: {best_dist:.12e}")

        return best_config, best_t, best_dist

    def config_to_qasm(self, config):
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "qreg q[2];"
        ]

        for layer in range(4):
            for gate in self.sq_library[config[layer][0]]['gates']:
                lines.append(f"{gate} q[0];")

            for gate in self.sq_library[config[layer][1]]['gates']:
                lines.append(f"{gate} q[1];")

            if layer < 3:
                lines.append("cx q[0],q[1];")

        return "\n".join(lines) + "\n"


def main():
    target_unitary = np.array([
        [0.1448081895+0.1752383997j, -0.5189281551-0.5242425896j,
         -0.1495585824+0.3127549990j,  0.1691348143-0.5053863118j],
        [-0.9271743926-0.0878506193j, -0.1126033063-0.1818584963j,
          0.1225587186+0.0964028611j, -0.2449850904-0.0504584131j],
        [-0.0079842758-0.2035507051j, -0.3893205530-0.0518092515j,
          0.2605170566+0.3286402481j,  0.4451730754+0.6558933250j],
        [ 0.0313792249+0.1961395216j,  0.4980474972+0.0884604926j,
          0.3407886532+0.7506609982j,  0.0146480652-0.1575584270j],
    ], dtype=complex)

    print("="*70)
    print("Task 10: Random Unitary Compilation")
    print("="*70)

    compiler = CliffordTPlusCompiler(target_unitary, max_single_qubit_t=7)

    best_config, t_count, distance = compiler.optimize_simulated_annealing(
        initial_temp=2.0,
        cooling_rate=0.93,
        iterations_per_temp=120,
        num_temps=60,
        random_seed=42
    )

    qasm_output = compiler.config_to_qasm(best_config)

    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"T-count: {t_count}")
    print(f"Operator distance: {distance:.12e}")
    print("\n" + "="*70)
    print("OpenQASM 2.0 Output:")
    print("="*70)
    print(qasm_output)

    with open('problem10.qasm', 'w') as f:
        f.write(qasm_output)

    print("Saved to: problem10.qasm")


if __name__ == "__main__":
    main()
