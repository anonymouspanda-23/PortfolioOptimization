import numpy as np
from qiskit import transpile
from qiskit_aer import Aer
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import ISRES
from qiskit_ibm_runtime import Estimator
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Pauli, SparsePauliOp

from qiskit.visualization import plot_histogram

# Define the Pauli Z operator for each asset (qubit)
Z_A = SparsePauliOp(Pauli("YII"))  # Operator for asset A
Z_B = SparsePauliOp(Pauli("IYI"))  # Operator for asset B
Z_C = SparsePauliOp(Pauli("IIY"))  # Operator for asset C

# Expected returns
r_A = 0.08
r_B = 0.12
r_C = 0.10

# Covariance terms
sigma_AB = 0.012
sigma_AC = 0.008
sigma_BC = 0.015

# Construct the Hamiltonian
H_returns = -r_A * Z_A - r_B * Z_B - r_C * Z_C
H_risk = sigma_AB * (Z_A @ Z_B) + sigma_AC * (Z_A @ Z_C) + sigma_BC * (Z_B @ Z_C)

# Full Hamiltonian
H = H_returns + H_risk

# Setup a quantum instance (simulator)
backend = Aer.get_backend('aer_simulator')

# Define a simple ansatz circuit
ansatz = TwoLocal(
    num_qubits=3,
    rotation_blocks='ry',
    entanglement_blocks='cz',
    reps=2,
    entanglement='linear'
)
print(ansatz)

# Define the optimizer
optimizer = ISRES(max_evals=100)

# Transpile the circuit
transpiled = transpile(ansatz, backend=backend)

# Run VQE
vqe = VQE(ansatz=transpiled, optimizer=optimizer, estimator=Estimator(backend=backend))

# Execute VQE with the Hamiltonian
result = vqe.compute_minimum_eigenvalue(operator=H)

# Print the result
print('Minimum Eigenvalue:', result.eigenvalue.real)
print('Optimal Parameters:', result.optimal_parameters)

# Assume 'ansatz' is your circuit and 'params' are the optimized parameters
params = result.optimal_parameters
backend = Aer.get_backend('aer_simulator')

# Set the optimized parameters into your circuit
test_params = [np.pi/4] * ansatz.num_parameters
circuit = ansatz.assign_parameters(params)

# Add measurement to all qubits
circuit.measure_all()

transpiled = transpile(circuit, backend=backend)
print(transpiled)

# Execute the circuit
job = backend.run(transpiled, shots=1024)
result = job.result()
counts = result.get_counts(circuit)

# Plot the histogram of outcomes
plot_histogram(counts)

# Calculate probabilities of |1> states for each asset
probabilities = {f'Asset {i+1}': counts.get('1'*i + '0' + '1'*(2-i), 0) / 1024 for i in range(3)}
print(probabilities)

