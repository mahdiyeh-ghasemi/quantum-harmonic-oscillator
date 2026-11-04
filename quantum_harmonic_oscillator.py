
import numpy as np
import matplotlib.pyplot as plt

# 1. Physical parameters (dimensionless units)
hbar = 1.0
mass = 1.0
omega = 1.0

# 2. Define the spatial grid
N = 400
x_min = -5.0
x_max = 5.0

x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]

# 3. Calculate the harmonic oscillator potential
V = 0.5 * mass * omega**2 * x**2

# 4. Construct the Hamiltonian matrix
# Finite-difference approximation of the kinetic energy
diagonal = np.full(N, hbar**2 / (mass * dx**2))
off_diagonal = np.full(N - 1, -hbar**2 / (2 * mass * dx**2))

T = (
    np.diag(diagonal)
    + np.diag(off_diagonal, 1)
    + np.diag(off_diagonal, -1)
)

# Hamiltonian = Kinetic energy + Potential energy
H = T + np.diag(V)

# 5. Solve the eigenvalue problem
energies, wavefunctions = np.linalg.eigh(H)

# 6. Compare numerical and analytical energy levels
print("Quantum Harmonic Oscillator")
print("--------------------------------")

for n in range(5):
    analytical_energy = hbar * omega * (n + 0.5)
    numerical_energy = energies[n]

    print(
        f"n = {n}: "
        f"Numerical = {numerical_energy:.6f}, "
        f"Analytical = {analytical_energy:.6f}"
    )

# 7. Plot the potential and first five wavefunctions
plt.figure(figsize=(10, 6))

plt.plot(x, V, label="Potential energy", linewidth=2)

for n in range(5):
    psi = wavefunctions[:, n]

    # Normalize the wavefunction
    psi = psi / np.sqrt(np.sum(psi**2) * dx)

    # Display wavefunction around its energy level
    plt.plot(
        x,
        energies[n] + 0.25 * psi,
        label=f"n = {n}"
    )

    plt.axhline(
        energies[n],
        linestyle="--",
        alpha=0.3
    )

plt.xlabel("Position (x)")
plt.ylabel("Energy")
plt.title("One-Dimensional Quantum Harmonic Oscillator")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("quantum_harmonic_oscillator.png", dpi=300)
plt.show()

