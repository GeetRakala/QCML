# QCML: Quantum Cognition Machine Learning

![License](https://img.shields.io/badge/license-MIT-blue)
![JAX](https://img.shields.io/badge/JAX-Accelerated-green)
![Platform](https://img.shields.io/badge/platform-CPU%20%7C%20GPU%20%7C%20TPU-orange)

## Overview

**QCML** implements a state-of-the-art **intrinsic dimension estimator** for high-dimensional data, leveraging principles from **quantum cognition** and **quantum geometry**. This framework reformulates the problem of dimensionality estimation by mapping data points to quantum states and analyzing the geometric properties of the resulting Hilbert space.

Key advantages include:
- **High Performance**: Written in **JAX** for seamless hardware acceleration on CPUs, GPUs, and TPUs.
- **Robust Estimation**: Utilizes a quantum metric derived from the ground state of an error Hamiltonian to estimate intrinsic dimension with high accuracy.
- **Flexible Optimization**: Supports multiple optimization strategies including analytical gradients, L-BFGS, and pseudo-gradients.

This implementation is based on the research published in **Scientific Reports**:
> [A quantum cognition approach to intrinsic dimension estimation](https://www.nature.com/articles/s41598-025-91676-8)

Beyond scientific research, this framework is being adopted for commercial quantum computing pipelines by [Qognitive AI](https://www.qognitiveai.com/).

## Key Features

- **JAX-Based Implementation**: Fully vectorised and JIT-compiled code ensures maximum performance.
- **Multiple Solvers**:
  - **Analytic Gradient**: Uses the exact gradient of the loss function (eigenvalue perturbation theory).
  - **L-BFGS**: Quasi-Newton method via `jaxopt` for fast convergence.
  - **Adam**: First-order stochastic optimization via `optax`.
  - **Pseudo-Gradient**: A heuristic approach for rapid approximation.
- **Parametrization Strategies**:
  - **Upper Triangular**: Standard Cholesky-like decomposition for Hermitian matrices.
  - **Pauli Basis**: Decomposition into Pauli matrices for quantum-native representations.
- **Data Generation**: Built-in synthetic datasets (Sphere, Cubic, Campadelli) for benchmarking.

## Mathematical Foundation

The core idea is to learn a set of Hermitian operators $\{A_\mu\}$ such that the mapping from a high-dimensional data point $x \in \mathbb{R}^E$ to a quantum state $|\psi_0(x)\rangle$ preserves local geometry. The state $|\psi_0(x)\rangle$ is defined as the ground state of an **Error Hamiltonian**:

$$ H_E(x) = \sum_{\mu=1}^E (A_\mu - x_\mu I)^2 $$

The intrinsic dimension is then estimated from the spectrum of the **Quantum Metric tensor** $g$, which is derived from the perturbation of the ground state:

$$ g_{\mu\nu} = 2 \text{Re} \sum_{n \neq 0} \frac{\langle \psi_0 | \partial_\mu H | \psi_n \rangle \langle \psi_n | \partial_\nu H | \psi_0 \rangle}{(E_n - E_0)^2} $$

For a detailed derivation of the pseudo-gradient approach, see `gradient.pdf` included in this repository.

## Dependencies

Ensure you have Python installed. Install the dependencies using pip:

```bash
pip install jax jaxlib jaxopt optax matplotlib numpy
```

*Note: For GPU/TPU support, please follow the specific [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).*

## Usage

The main library is contained in `qcml.py`.

### Running the Script

You can run the script directly using Python to execute a standard experiment:

```bash
python qcml.py
```

### Library Usage

You can also import the `run_experiment` function to run custom experiments:

```python
import jax
from qcml import run_experiment

# Experiment Configuration
key = jax.random.PRNGKey(42)
solver = "jaxopt"           # Options: "analytic", "jaxopt", "optax", "pseudo"
dataset_type = "sphere"     # Options: "sphere", "cubic", "campadelli_beta", etc.
noise_level = 0.01

# Dimensions
N_points = 100              # Number of data points
H_dim = 4                   # Hilbert space dimension
E_dim = 3                   # Embedding dimension (data dimension)

# Training Hyperparameters
num_epochs = 1000
lr = 0.01
l2_lambda = 0.001

# Run Experiment
results = run_experiment(
    key=key,
    solver=solver,
    dataset_type=dataset_type,
    noise_level=noise_level,
    parametrization="upper", # Options: "upper", "pauli"
    N_points=N_points,
    H_dim=H_dim,
    E_dim=E_dim,
    num_epochs=num_epochs,
    A_init_scale=0.1,
    lr=lr,
    l2_lambda=l2_lambda,
    grad_clip_norm=1.0,      # For optax
    transition_steps=500,    # For optax schedule
    decay_rate=0.9           # For optax schedule
)

# Results contain: (Intrinsic Dim Array, Embeddings, Ground State Energies, Metric Eigenvalues)
I_dims, Y_embed, E0s, G_eigs = results
print("Estimated Intrinsic Dimension:", I_dims.mean())
```

### Supported Datasets

The `datasets_jax.py` module provides several synthetic datasets for validation:
- `sphere` / `sphere_other`: Hyperspheres tailored for specific dimensions.
- `cubic` / `cubic_other`: Hypercubes.
- `campadelli_beta` / `campadelli_n`: Complex manifolds from Campadelli et al. benchmark.

## Project Structure

- `qcml.py`: The core library containing:
    - Matrix parametrization logic (`upper`, `pauli`).
    - Loss functions and gradients (analytic & pseudo).
    - Training loops (`train_A_array_...`).
    - Intrinsic dimension estimator (`I_dimension_estimator_jax`).
    - Plotting utilities.
- `datasets_jax.py`: Functions to generate synthetic high-dimensional datasets.
- `plots.pdf`: Sample output plots comparing results to the white paper.
- `gradient.pdf`: Mathematical derivation of the gradient descent update rules.

## References

1. **Scientific Reports** (2025). *A quantum cognition approach to intrinsic dimension estimation*. [https://www.nature.com/articles/s41598-025-91676-8](https://www.nature.com/articles/s41598-025-91676-8)
2. **Qognitive AI**: [https://www.qognitiveai.com/](https://www.qognitiveai.com/)

## License

This project is licensed under the **MIT License**.

## Author

**Geet Rakala**
