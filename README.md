# QCML: Quantum Cognition Machine Learning

![License](https://img.shields.io/badge/license-MIT-blue)
![JAX](https://img.shields.io/badge/JAX-Accelerated-green)
![Platform](https://img.shields.io/badge/platform-CPU%20%7C%20GPU%20%7C%20TPU-orange)

## Overview

This repository implements an intrinsic dimension estimator based on quantum geometric methods. The approach constructs a set of Hermitian operators ${A_\mu}$ that encode the local geometry of a data manifold embedded in high-dimensional space. Each data point is mapped to the ground state of an error Hamiltonian, and the intrinsic dimension is extracted from the eigenvalue spectrum of the quantum metric tensor.

The implementation uses JAX for automatic differentiation and XLA compilation, enabling execution on CPU, GPU, and TPU backends.

This is an open-source reproduction of methods from:
> [A quantum cognition approach to intrinsic dimension estimation](https://www.nature.com/articles/s41598-025-91676-8) (Scientific Reports, 2025)

> [!NOTE]
> For a detailed technical reference of the codebase, see **[DOCUMENTATION.md](DOCUMENTATION.md)**.

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

Given a dataset of $N$ points $x \in \mathbb{R}^E$, the method learns a set of Hermitian matrices $A_\mu$ (for $\mu=1 \dots E$) of dimension $H \times H$ by minimizing the reconstruction error.

For each data point $x$, the **error Hamiltonian** is:

$$ H_E(x) = \frac{1}{2}\sum_{\mu=1}^E (A_\mu - x_\mu I)^2 $$

The ground state $|\psi_0(x)\rangle$ satisfies $H_E(x)|\psi_0(x)\rangle = E_0|\psi_0(x)\rangle$ where $E_0$ is the minimum eigenvalue.

The reconstruction is $y_\mu = \langle \psi_0 | A_\mu | \psi_0 \rangle$, and the loss is $L = \sum_i ||y^{(i)} - x^{(i)}||^2$.

The **quantum metric** is computed from excited state contributions:

$$ g_{\mu\nu} = 2 \text{Re} \sum_{n \geq 1} \frac{\langle \psi_0 | A_\mu | \psi_n \rangle \langle \psi_n | A_\nu | \psi_0 \rangle}{E_n - E_0} $$

The intrinsic dimension is estimated from the spectrum of $g$ by identifying the spectral gap where eigenvalues transition from $O(1)$ to $O(\epsilon)$.

See `gradient.pdf` for the analytic gradient derivation.


## Dependencies
This project uses `uv` for dependency management.

```bash
uv sync
```



## Usage

The main library is contained in `qcml.py`.

### Running the Script

You can run the reproduction script using `uv`:

```bash
uv run main.py
```

This will run experiments as configured in `config/default_config.yaml`.

### Library Usage

You can also import the `run_experiment` function from `main_reproduction.py` to run custom experiments programmatically:

```python
import jax
import yaml
from main import run_experiment

# Load base config
with open("config/default_config.yaml", "r") as f:
    config = yaml.safe_load(f)["experiment"]

# Override params
config["solver"] = "jaxopt"
config["dataset_type"] = "sphere"
config["epochs"] = 1000

# Run
run_experiment(config, noise_level=0.01, seed=42)
```

### Dashboard

You can explore results and run new experiments using the interactive Streamlit dashboard. It allows you to:
- **Visualize** experiment results and plots.
- **Configure** and launch new runs with full parameter control via the sidebar.

```bash
uv run streamlit run dashboard.py
```


## Files

- `src/`: Source code directory
  - `data.py`: Synthetic dataset generators
  - `models.py`: Parametrizations and quantum metric logic
  - `loss.py`: Loss functions
  - `training.py`: Training loops
  - `plotting.py`: Visualization
- `config/`: Configuration files (`default_config.yaml`)
- `main.py`: Main entry point
- `dashboard.py`: Streamlit experiment dashboard
- `DOCUMENTATION.md`: Detailed API and logic documentation
- `plots.pdf`: Benchmark results
- `gradient.pdf`: Analytic gradient derivation


## References

1. **Scientific Reports** (2025). *A quantum cognition approach to intrinsic dimension estimation*. [https://www.nature.com/articles/s41598-025-91676-8](https://www.nature.com/articles/s41598-025-91676-8)
2. **Qognitive AI**: [https://www.qognitiveai.com/](https://www.qognitiveai.com/)

## License

This project is licensed under the **MIT License**.

## Author

**Geet Rakala**
