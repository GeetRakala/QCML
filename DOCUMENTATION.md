# QCML Documentation

This document provides a detailed technical reference for the **QCML** codebase, specifically the core logic in `qcml.py` and the data generation utilities in `datasets_jax.py`.

## Module: `qcml.py`

The `qcml.py` file contains the main implementation of the Quantum Cognition Machine Learning framework for intrinsic dimension estimation. It relies heavily on **JAX** for high-performance linear algebra and automatic differentiation.

### Core Physics Logic

These functions implement the quantum geometric mapping and metric calculation.

#### `error_hamiltonian_jax(X_point, A_array)`
Constructs the Error Hamiltonian $H_E(x)$ for a single data point.
*   **Args**:
    *   `X_point`: Data point vector $x \in \mathbb{R}^E$.
    *   `A_array`: Array of Hermitian matrices $\{A_\mu\}$, shape $(E, H, H)$.
*   **Returns**: $H \times H$ Hermitian matrix.
*   **Math**: $H_E(x) = \frac{1}{2} \sum_\mu (A_\mu - x_\mu I)^2$.

#### `compute_ground_state_jax(X_point, A_array)`
Diagonalizes the Error Hamiltonian to find the ground state.
*   **Returns**:
    *   `E_eigvec_0`: Ground state eigenvector $|\psi_0\rangle$.
    *   `E_eigval_0`: Ground state energy $E_0$.
    *   `E_eigvals`: Full eigenvalue spectrum.
    *   `E_eigvecs`: Full eigenvector basis.

#### `point_mapping_jax(psi, A_array)`
Maps a quantum state back to the embedding space (reconstruction).
*   **Math**: $y_\mu = \text{Re}(\langle \psi | A_\mu | \psi \rangle)$.

#### `quantum_metric_jax(E_eigvec_0, ..., A_array)`
Computes the quantum metric tensor $g$ using perturbation theory.
*   **Math**: $g_{\mu\nu} = 2 \text{Re} \sum_{n \neq 0} \frac{\langle \psi_0 | A_\mu | \psi_n \rangle \langle \psi_n | A_\nu | \psi_0 \rangle}{E_n - E_0}$.
*   **Note**: This metric measures the sensitivity of the ground state to changes in the manifold parameters.

#### `estimate_single_point_jax(X_point, A_array)`
Estimates the intrinsic dimension for a single point.
*   **Logic**:
    1.  Computes ground state and quantum metric.
    2.  Finds eigenvalues of the metric $g$.
    3.  Identifies the spectral gap (largest ratio between consecutive sorted eigenvalues).
    4.  Intrinsic dimension $d = E - \text{gap\_index}$.

### Optimization Solvers

The framework provides multiple strategies to train the matrices $\{A_\mu\}$ by minimizing the reconstruction loss $L = \sum_i ||y^{(i)} - x^{(i)}||^2$.

#### `train_A_array_analytic_jax(...)`
Uses the **exact analytical gradient** derived from the Hellmann-Feynman theorem and eigenvalue perturbation.
*   **Key Args**: `lr`, `l2_lambda` (regularization).
*   **Pros**: Most theoretically accurate.
*   **Cons**: Computationally expensive due to full eigendecomposition per step.

#### `train_A_array_LBFGS(...)`
Uses `jaxopt.LBFGS` for quasi-Newton optimization.
*   **Pros**: Often faster convergence than simple gradient descent.
*   **Note**: The objective function must be JIT-compilable.

#### `train_A_array_optax(...)`
Uses the **Adam** optimizer from the `optax` library.
*   **Key Args**: `decay_rate`, `transition_steps` (for exponential learning rate schedule), `grad_clip_norm`.
*   **Pros**: robust standard for deep learning tasks.

#### `train_A_array_pseudo_jax(...)`
Uses a **pseudo-gradient** heuristic approximation: $\nabla \approx 2(y - x) |\psi_0\rangle\langle\psi_0|$.
*   **Pros**: Very fast iteration.
*   **Cons**: Approximation may not always converge to the global optimum.

### Parametrization

Two methods are available to ensure the matrices $A_\mu$ remain Hermitian:

1.  **"upper"**: Parametrizes the upper triangular part (complex) and diagonal (real).
    *   `matrix_from_upper`: Reconstructs the full matrix.
2.  **"pauli"**: Expands the matrix in a generalized Pauli basis.
    *   `generate_pauli_basis`: Creates the basis tensors.
    *   `matrix_from_pauli_params`: Reconstructs via linear combination.

### Experiment Runner

#### `run_experiment(...)`
The high-level entry point for running a complete pipeline.
1.  Generates synthetic data (via `datasets_jax`).
2.  Initializes parameters ($A_\mu$).
3.  Runs the selected Training loop.
4.  Estimates dimensions on the trained model.
5.  Generates and saves plots.

---

## Module: `datasets_jax.py`

Generates synthetic high-dimensional data points $x \in \mathbb{R}^{E}$ that lie on a manifold of intrinsic dimension $I$.

### Functions

#### `gen_sphere_data(..., E_dim, I_dim)`
Generates points on a unit hypersphere $S^I$ embedded in $\mathbb{R}^E$.
*   Dimensions $I < E$. The extra dimensions are padded with zeros (plus noise).

#### `gen_cubic_data(..., E_dim, I_dim)`
Generates points on a hypercube.

#### `gen_campadelli_beta_data(...)` & `gen_campadelli_n_data(...)`
Generates complex manifold structures used in the *Campadelli et al.* benchmark papers, involving trigonometric mappings and beta distributions.
