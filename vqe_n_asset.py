import numpy as np
import yfinance as yf

from qiskit import transpile, QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA, COBYLA, SLSQP
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_ibm_runtime import Estimator

# -> Get stonk and calculate daily returns.
# set params
start_date = "2013-01-01"
end_date = "2018-01-01"
tickers = ["GLD", "DBC", "TLT", "VTI", "AAPL"]
num_assets = len(tickers)

# Define global constants
risk_free_rate = 0.0001

# get stonks
stonks = yf.download(tickers, start=start_date, end=end_date)
stonks = stonks["Adj Close"]

# calculate daily returns (we use percent change here, but log returns can be used too.)
stonks_returns = stonks.pct_change().dropna()
expected_returns = np.mean(stonks_returns.to_numpy(), axis=0)
print(f"Expected Returns: {expected_returns}")
covariance_matrix = stonks_returns.cov().to_numpy()

# Find the absolute maximum value
max_cov = np.max(np.abs(covariance_matrix))

# Normalize to [-1, 1]
covariance_matrix = covariance_matrix / max_cov

print(f"Covariance Matrix: {covariance_matrix}")

# print(stonks_returns.to_numpy())
# print(covariance_matrix)

# --- Quantum Encoding and Ansatz ---
# Binary encoding: Each qubit represents an asset
# |0⟩: no investment, |1⟩: full investment
num_qubits = num_assets
ansatz = EfficientSU2(
    num_qubits=num_qubits,
    su2_gates=['ry', 'cx'],
    reps=5,
    entanglement='full'
)
# ansatz = EfficientSU2(num_qubits, 'ry', reps=3, entanglement='full')

# --- Cost Function ---
# Construct Hamiltonian for portfolio optimization
pauli_list = []
for i in range(num_assets):
    for j in range(num_assets):
        pauli_string = ['I'] * num_assets  # Initialize with all identities
        pauli_string[i] = 'Z'  # Set the i-th element to 'Z'
        pauli_string[j] = 'Z'  # Set the j-th element to 'Z'
        pauli_list.append((''.join(pauli_string), covariance_matrix[i, j]))

# Add expected returns with a risk aversion parameter (lambda)
avg_daily_return = stonks_returns.mean()
excess_returns = stonks_returns - risk_free_rate
portfolio_risk = excess_returns.std()
sharpe_ratio = (avg_daily_return - risk_free_rate) / portfolio_risk

# Find the absolute maximum value
max_sharpe = np.max(np.abs(sharpe_ratio))

# Normalize to [-1, 1]
sharpe_ratio = sharpe_ratio / max_sharpe

print(f"Individual Sharpe Ratio: {sharpe_ratio}")

for i in range(num_assets):
    pauli_string = ['I'] * num_assets
    pauli_string[i] = 'Z'
    pauli_list.append((''.join(pauli_string), sharpe_ratio[i]))

print(f"Pauli List: {pauli_list}")

hamiltonian = SparsePauliOp.from_list(pauli_list)

# Transpile the circuit
backend = Aer.get_backend('aer_simulator')  # Use simulator for now
transpiled = transpile(ansatz, backend=backend)

# print(transpiled.decompose())

# Define a custom Estimator to handle measurements
estimator = Estimator(
    backend=backend
)

estimator.options.resilience = 2

# --- VQE Optimization ---
optimizer = SPSA(maxiter=1024)  # Choose a classical optimizer
vqe = VQE(ansatz=transpiled, optimizer=optimizer, estimator=estimator)
result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

# --- Extract Optimal Weights ---
optimal_params = result.optimal_parameters
optimal_circuit = ansatz.assign_parameters(optimal_params)
print(result.optimal_value)

# --- 1. Circuit for Stock Selection (Z-basis) ---
amplitude_circuit = optimal_circuit.copy()
amplitude_circuit.measure_all()
transpiled = transpile(amplitude_circuit, backend=backend)
job = backend.run(transpiled, shots=1024)
result = job.result()
counts = result.get_counts()
# print(result)

# --- Print Results ---
print("Counts:", counts)

probabilities = {
    i: 0 for i in range(num_assets)
}

total = 0

for count in counts.keys():
    for i in range(num_assets):
        if count[i] == '1':
            probabilities[i] += counts[count]
            total += counts[count]

for idx in probabilities.keys():
    probabilities[idx] /= total

print(probabilities)

# Calculate Sharpe ratio
weights = np.array(list(probabilities.values()))
exp_ret = np.sum(stonks_returns.mean() * weights) * 252
exp_vol = np.sqrt(np.dot(weights.T, np.dot(stonks_returns.cov() * 252, weights)))
SR = exp_ret / exp_vol
print(SR)
