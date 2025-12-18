# QCML API Reference

Technical documentation for the QCML codebase.

## Core Functions (`src/models.py`)

### Error Hamiltonian Construction

**`error_hamiltonian_jax(X_point, A_array)`**

Builds the error Hamiltonian $H_E(x) = \frac{1}{2}\sum_{\mu=1}^E (A_\mu - x_\mu I)^2$.

- Input: `X_point` (shape `(E,)`), `A_array` (shape `(E, H, H)`)
- Output: Hermitian matrix (shape `(H, H)`)
- Uses `einsum` for vectorized computation over $\mu$

**`compute_ground_state_jax(X_point, A_array)`**

Diagonalizes $H_E$ via `jnp.linalg.eigh`.

Returns: `(eigvec_0, eigval_0, eigvals, eigvecs)` where ground state is `eigvec_0` with energy `eigval_0`.

**`point_mapping_jax(psi, A_array)`**

Reconstruction operator: $y_\mu = \text{Re}\langle \psi | A_\mu | \psi \rangle$.

**`quantum_metric_jax(E_eigvec_0, E_eigval_0, E_eigvals, E_eigvecs, A_array)`**

Computes the Fubini-Study metric via first-order perturbation theory:

$$g_{\mu\nu} = 2 \text{Re} \sum_{n \geq 1} \frac{\langle \psi_0 | A_\mu | \psi_n \rangle \langle \psi_n | A_\nu | \psi_0 \rangle}{E_n - E_0}$$

This is the pullback of the Hilbert space geometry to the parameter manifold. Regularization constant $\epsilon = 10^{-20}$ prevents division by zero for degenerate states.

**`estimate_single_point_jax(X_point, A_array)`**

Intrinsic dimension estimator:
1. Compute $g$ and diagonalize
2. Sort eigenvalues $\lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_E$
3. Find $\gamma = \arg\max_i \frac{\lambda_{i+1}}{\lambda_i}$ (spectral gap)
4. Return $d = E - \gamma$

Rationale: On a $d$-dimensional manifold, $g$ has rank $d$, so $E-d$ eigenvalues are $O(\epsilon)$.

---

## Optimization Methods (`src/training.py`)

All methods minimize $\mathcal{L} = \frac{1}{N}\sum_{i=1}^N \|y^{(i)} - x^{(i)}\|^2 + \lambda \|A\|_F^2$.

### `train_A_array_analytic_jax`

Gradient computed via Hellmann-Feynman theorem:

$$\frac{\partial \mathcal{L}}{\partial A_\mu} = \frac{2}{N}\sum_i (y_\mu^{(i)} - x_\mu^{(i)}) |\psi_0^{(i)}\rangle\langle\psi_0^{(i)}| + \text{correction terms}$$

Correction terms from $\partial |\psi_0\rangle / \partial A_\mu$ using degenerate perturbation theory (see `gradient.pdf`).

### `train_A_array_LBFGS`

Wrapper around `jaxopt.LBFGS`. Uses JAX autodiff for gradient. Memory efficient for $H \gg 10$.

### `train_A_array_optax`

Adam with exponential decay: $\alpha(t) = \alpha_0 \cdot \rho^{t/T}$ where $\rho$ = `decay_rate`, $T$ = `transition_steps`.

Gradient clipping via `optax.clip_by_global_norm(grad_clip_norm)`.

### `train_A_array_pseudo_jax`

Ignores excited state corrections:

$$\frac{\partial \mathcal{L}}{\partial A_\mu} \approx \frac{2}{N}\sum_i (y_\mu^{(i)} - x_\mu^{(i)}) |\psi_0^{(i)}\rangle\langle\psi_0^{(i)}|$$

Fastest iteration but may stall far from optimum.

---

## Matrix Parametrization (`src/models.py`)

Hermitian matrices have $H^2$ real degrees of freedom. Two parametrizations:

### Upper Triangular (`upper`)

Store $H(H+1)/2$ complex parameters: $H(H-1)/2$ off-diagonal + $H$ diagonal (real).

- `upper_from_matrix`: Extract parameters via `triu_indices(H, k=0)`
- `matrix_from_upper`: Reconstruct $A = U + U^\dagger - \text{diag}(U)$ where $U$ is upper triangular

### Pauli Basis (`pauli`)

Expand in Hermitian basis: $A = \sum_{k=1}^{H^2} c_k \sigma_k$ where $\{\sigma_k\}$ are:
- $H$ diagonal matrices $E_{ii}$
- $H(H-1)/2$ symmetric: $E_{ij} + E_{ji}$
- $H(H-1)/2$ antisymmetric: $i(E_{ij} - E_{ji})$

All coefficients $c_k \in \mathbb{R}$.

- `generate_pauli_basis`: Constructs $\{\sigma_k\}$ (shape `(H^2, H, H)`)
- `matrix_from_pauli_params`: Computes $A$ via `tensordot`

---

## Datasets (`src/data.py`)

All functions return arrays of shape `(N_points, E_dim)`.

**`gen_sphere_data(key, N_points, E_dim, I_dim, noise)`**

Samples uniformly from $S^{I} \subset \mathbb{R}^{I+1}$, embeds in $\mathbb{R}^E$ by zero-padding dimensions $I+2, \dots, E$.

**`gen_cubic_data(key, N_points, E_dim, I_dim, noise)`**

Samples from boundary of $[0,1]^{I+1}$ (each face contains $\sim N/(2(I+1))$ points).

**`gen_campadelli_beta_data(key, N_points, E_dim, I_dim, noise, alpha, beta)`**

Constraint: `E_dim = 4 * I_dim`.

Samples $u \sim \text{Beta}(\alpha, \beta)$, maps to:
$$[u \sin(\cos(2\pi u)), u \cos(\sin(2\pi u)), u \sin(\cos(2\pi u)), u \cos(\sin(2\pi u))]$$
repeated for each dimension.

**`gen_campadelli_n_data(key, N_points, E_dim, I_dim, noise)`**

Constraint: `E_dim = 4 * I_dim`.

Samples $u \sim \mathcal{U}(0,1)^{I}$, applies:
$$v_i = \tan(u_i \cos(u_{I-i})), \quad w_i = \arctan(u_{I-i} \sin(u_i))$$

Concatenates $[v, w, v, w]$.
