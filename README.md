# QCML: Quantum Cognition Machine Learning

![License](https://img.shields.io/badge/license-MIT-blue)
![JAX](https://img.shields.io/badge/JAX-Accelerated-green)
![Platform](https://img.shields.io/badge/platform-CPU%20%7C%20GPU%20%7C%20TPU-orange)

## Overview

This repository implements an intrinsic dimension estimator based on quantum geometric methods. The approach constructs a set of Hermitian operators ${A_\mu}$ that encode the local geometry of a data manifold embedded in high-dimensional space. Each data point is mapped to the ground state of an error Hamiltonian, and the intrinsic dimension is extracted from the eigenvalue spectrum of the quantum metric tensor.

The implementation uses JAX for automatic differentiation and XLA compilation, enabling execution on CPU, GPU, and TPU backends.

This is an open-source reproduction of methods from:
> [A quantum cognition approach to intrinsic dimension estimation](https://www.nature.com/articles/s41598-025-91676-8) (Scientific Reports, 2025)

## Implementation Details

**Optimization Methods:**
- `analytic`: Full analytic gradient via eigenvalue perturbation theory
- `jaxopt`: L-BFGS quasi-Newton solver
- `optax`: Adam optimizer with exponential decay scheduling
- `pseudo`: Pseudo-gradient heuristic (first-order approximation)

**Matrix Parametrizations:**
- `upper`: Upper triangular + diagonal (Cholesky-like for Hermitian matrices)
- `pauli`: Expansion in generalized Pauli basis

**Datasets:**
- Hypersphere manifolds (`sphere`, `sphere_other`)
- Hypercube boundaries (`cubic`, `cubic_other`)
- Campadelli benchmark manifolds (`campadelli_beta`, `campadelli_n`)


## Method

Given data $\{x^{(i)}\}_{i=1}^N$ with $x \in \mathbb{R}^E$, the method learns Hermitian matrices $\{A_\mu\}_{\mu=1}^E$ of dimension $H \times H$ by minimizing the reconstruction error. 

For each data point $x$, the **error Hamiltonian** is:

$$ H_E(x) = \frac{1}{2}\sum_{\mu=1}^E (A_\mu - x_\mu I)^2 $$

The ground state $|\psi_0(x)\rangle$ satisfies $H_E(x)|\psi_0(x)\rangle = E_0|\psi_0(x)\rangle$ where $E_0$ is the minimum eigenvalue.

The reconstruction is $y_\mu = \langle \psi_0 | A_\mu | \psi_0 \rangle$, and the loss is $L = \sum_i ||y^{(i)} - x^{(i)}||^2$.

The **quantum metric** is computed from excited state contributions:

$$ g_{\mu\nu} = 2 \text{Re} \sum_{n \geq 1} \frac{\langle \psi_0 | A_\mu | \psi_n \rangle \langle \psi_n | A_\nu | \psi_0 \rangle}{E_n - E_0} $$

The intrinsic dimension is estimated from the spectrum of $g$ by identifying the spectral gap where eigenvalues transition from $O(1)$ to $O(\epsilon)$.

See `gradient.pdf` for the analytic gradient derivation.


## Dependencies

```bash
pip install jax jaxlib jaxopt optax matplotlib numpy
```

For GPU/TPU: see [JAX installation](https://jax.readthedocs.io/en/latest/installation.html).


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


## Files

- `qcml.py`: Core implementation (parametrizations, loss functions, solvers, dimension estimator, plotting)
- `datasets_jax.py`: Synthetic dataset generators
- `plots.pdf`: Benchmark results
- `gradient.pdf`: Analytic gradient derivation


## References

1. **Scientific Reports** (2025). *A quantum cognition approach to intrinsic dimension estimation*. [https://www.nature.com/articles/s41598-025-91676-8](https://www.nature.com/articles/s41598-025-91676-8)
2. **Qognitive AI**: [https://www.qognitiveai.com/](https://www.qognitiveai.com/)

## License

This project is licensed under the **MIT License**.

## Author

**Geet Rakala**
