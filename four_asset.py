import numpy as np
import yfinance as yf

from qiskit import transpile
from qiskit_aer import Aer
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Estimator

# # Input data (provided by the client)
# stock_prices = np.array([
#     [-0.02297105, -0.01964028, -0.01811224,  0.01544793],
#     [0.08233189, 0.01996856, -0.01897386,  0.0018593],
#     [0.01145675, -0.03467822, -0.01095753,  0.00591753],
#     [0.05373614, -0.00612196, -0.01612535, -0.00181888],
#     [0.03198097,  0.00273547, -0.0511027,   0.01611615],
#     [-0.03417745,  0.01427037,  0.05118653, -0.01473555],
#     [0.03013212,  0.02882474,  0.01565933, -0.00371457],
#     [-0.01708758,  0.01579765, -0.02499509, -0.01441414],
#     [-0.01770277, -0.0045642,   0.01070438,  0.01029592],
#     [0.00543173,  0.01145219, -0.01562633,  0.0104876],
#     [-0.08483991, -0.01961813, -0.00766236, -0.00629529],
#     [0.07242778,  0.02268077,  0.0132791,   0.0203666],
#     [-0.03801727, -0.00242687,  0.03431295, -0.0035784],
#     [0.05890471, -0.01047515,  0.05289772, -0.01671266],
#     [0.00758122, -0.01004085,  0.05633647, -0.00345499],
#     [0.08455986, -0.00097388, -0.00778922,  0.00443869],
#     [-0.03358663, -0.00624007, -0.0157648,  -0.00287705],
#     [-0.01641907,  0.03660764,  0.00319217,  0.02788839],
#     [0.01808379, -0.01107574, 0.00410041,  0.01024406],
#     [-0.11608632,  0.00050851, -0.02336168,  0.02052345],
#     [0.0492567,  0.00043721, -0.02021471,  0.01609858],
#     [0.08741273,  0.00044048,  0.02704262, -0.01927199],
#     [0.05875563,  0.0008619,   0.053155,   -0.00440342],
#     [0.03717224, -0.00749295, -0.00696308, -0.03194582],
#     [-0.00693171, -0.00914615, -0.02005938, -0.00781793],
#     [0.06129558, -0.01364095, -0.00895204,  0.01359305],
#     [0.04838989, -0.02387127, -0.03172137, -0.01014084],
#     [0.01687285,  0.00355455, -0.00898317, -0.00773555],
# ])
# covariance_matrix = np.array([
#     [2.54138859e-03,  7.34022167e-05,  1.28600531e-04, -9.98612132e-05],
#     [7.34022167e-05,  2.58486713e-04,  5.30427595e-05,  4.44816208e-05],
#     [1.28600531e-04,  5.30427595e-05,  7.91504681e-04, -1.23887382e-04],
#     [9.98612132e-05,  4.44816208e-05, -1.23887382e-04,  1.97892585e-04]
# ])
#
# # --- Data Preprocessing ---
# num_assets = len(covariance_matrix)
# expected_returns = np.mean(stock_prices, axis=0)

# -> Get stonk and calculate daily returns.
# set params
start_date = "2013-01-01"
end_date = "2018-01-01"
tickers = ["GLD", "DBC", "TLT", "VTI"]
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

# print(stonks_returns.to_numpy())
# print(covariance_matrix)

# --- Quantum Encoding and Ansatz ---
# Binary encoding: Each qubit represents an asset
# |0⟩: no investment, |1⟩: full investment
num_qubits = num_assets
ansatz = TwoLocal(num_qubits, 'ry', 'cz', reps=3, entanglement='full')

# --- Cost Function ---
# Construct Hamiltonian for portfolio optimization
pauli_list = []
for i in range(num_assets):
    for j in range(num_assets):
        if i == j:
            continue

        pauli_string = ['I'] * num_assets  # Initialize with all identities
        pauli_string[i] = 'Z'  # Set the i-th element to 'Z'
        pauli_string[j] = 'Z'  # Set the j-th element to 'Z'
        pauli_list.append((''.join(pauli_string), covariance_matrix[i, j]))

# Add expected returns with a risk aversion parameter (lambda)
avg_daily_return = stonks_returns.mean()
excess_returns = stonks_returns - risk_free_rate
portfolio_risk = excess_returns.std()
sharpe_ratio = (avg_daily_return - risk_free_rate) / portfolio_risk

print(f"Individual Sharpe Ratio: {sharpe_ratio}")

for i in range(num_assets):
    pauli_string = 'I' * i + 'Z' + 'I' * (num_assets - i - 1)
    pauli_list.append((''.join(pauli_string), sharpe_ratio[i]))

hamiltonian = SparsePauliOp.from_list(pauli_list)

# Transpile the circuit
backend = Aer.get_backend('aer_simulator')  # Use simulator for now
transpiled = transpile(ansatz, backend=backend)

# --- VQE Optimization ---
optimizer = SPSA(maxiter=300)  # Choose a classical optimizer
vqe = VQE(ansatz=transpiled, optimizer=optimizer, estimator=Estimator(backend=backend))
result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

# --- Extract Optimal Weights ---
optimal_params = result.optimal_parameters
optimal_circuit = ansatz.assign_parameters(optimal_params)
# Extract weights from the optimal state (requires further processing)
# ...

# --- Prepare and run new circuit ---
optimal_circuit.measure_all()
transpiled = transpile(optimal_circuit, backend=backend)
job = backend.run(transpiled, shots=1024)
result = job.result()
counts = result.get_counts()

# --- Print Results ---
print("Counts:", counts)
# ... (Print optimal weights after processing)

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
"""
{0: 0.0005420054200542005, 1: 0.0, 2: 0.4444444444444444, 3: 0.5550135501355014}

"""
