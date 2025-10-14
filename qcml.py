#author  : Geet Rakala
#license : MIT

# E_dim     : Embedding space dimension
# H_dim     : Hilbert space dimension
# I_dim     : Estimate of the intrinsic dimension
# N_points  : Number of data points
# A_array   : Matrix configuration as a jax array
# X_array   : All data points as a jax array
# X_point   : Single data point
# E_matrix  : Error Hamiltonian
# G_matrix  : Quantum metric

from functools import partial
import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
import numpy as np
import optax
import jaxopt
import matplotlib.pyplot as plt
import os
from datasets_jax import *

if jax.default_backend() == "tpu":
    jnp_complex_dtype = jnp.complex64
    jax.config.update("jax_enable_x64", False)
else:
    jnp_complex_dtype = jnp.complex128
    jax.config.update("jax_enable_x64", True)

# upper triangular + diagonal parametrisation
@partial(jit, static_argnums=(1,2))
def initialize_A_matrix_jax(key, H_dim, scale=0.1):
    """Initialize a Hermitian A_matrix with random uptri + diag parameters.
    Args:
        key: JAX PRNG key.
        H_dim: Hilbert space dimension.
        scale: Scaling factor for random initialization.
    Returns:
        A_matrix: A Hermitian jax array of shape (H_dim, H_dim).
    """
    A_matrix = jnp.zeros((H_dim, H_dim), dtype=jnp_complex_dtype)  # shape: (H_dim, H_dim)
    # off-diagonal elements (complex values)
    key_diag, key_upper_real, key_upper_imag = jax.random.split(key, 3)
    num_off_diag = H_dim * (H_dim - 1) // 2
    real_parts = jax.random.normal(key_upper_real, (num_off_diag,)) * scale
    imag_parts = jax.random.normal(key_upper_imag, (num_off_diag,)) * scale
    upper_values = real_parts + 1j * imag_parts
    A_matrix = A_matrix.at[jnp.triu_indices(H_dim, k=1)].set(upper_values)
    A_matrix = A_matrix + A_matrix.conj().T
    # diagonal elements (real values)
    diag_values = jax.random.normal(key_diag, (H_dim,)) * scale
    A_matrix = A_matrix.at[jnp.diag_indices(H_dim)].set(diag_values)
    return A_matrix  # shape: (H_dim, H_dim)

@partial(jit, static_argnums=(1, 2, 3))
def initialize_A_array_jax(key, E_dim, H_dim, scale=0.1):
    """Initialize an array of Hermitian A_matrices.
    Args:
        key: JAX PRNG key.
        E_dim: Embedding space dimension.
        H_dim: Hilbert space dimension.
        scale: Scaling factor for random initialization.
    Returns:
        A_array: jax array of shape (E_dim, H_dim, H_dim).
    """
    keys = jax.random.split(key, E_dim)
    A_array = jax.vmap(lambda k: initialize_A_matrix_jax(k, H_dim, scale=scale))(keys)
    return A_array  # shape: (E_dim, H_dim, H_dim)

@partial(jit, static_argnums=(1,))
def matrix_from_upper(upper_params, H_dim):
    """Reconstruct a Hermitian matrix from its uptri + diag parameters.
    Args:
        upper_params: jax array of shape (H_dim*(H_dim+1)//2,) containing the uptri+diag.
        H_dim: Hilbert space dimension.
    Returns:
        A_matrix: A Hermitian jax array of shape (H_dim, H_dim).
    """
    A_matrix = jnp.zeros((H_dim, H_dim), dtype=jnp_complex_dtype)
    A_matrix = A_matrix.at[jnp.triu_indices(H_dim, k=0)].set(upper_params)  # includes diagonal
    #diag_values = A_matrix.diagonal()
    #diag_values = jnp.real(diag_values)
    #A_matrix = A_matrix.at[jnp.diag_indices(N)].set(diag_values)
    A_matrix = A_matrix + jnp.triu(A_matrix, k=1).conj().T  # excludes diagonal
    return A_matrix  # shape: (H_dim, H_dim)

@partial(jit, static_argnums=(1,))
def upper_from_matrix(A_array, H_dim):
    """Extract uptri + diag parameters from a Hermitian matrix.
    Args:
        A_array: jax array of shape (E_dim, H_dim, H_dim).
        H_dim: Hilbert space dimension.
    Returns:
        A jax array of shape (E_dim, H_dim*(H_dim+1)//2) uptri+diag parameters.
    """
    i_upper, j_upper = jnp.triu_indices(H_dim, k=0) #includes diagonal
    return A_array[:, i_upper, j_upper]  # shape: (E_dim, H_dim*(H_dim+1)//2)

# Pauli basis parametrisation
def generate_pauli_basis(H_dim):
    """Generate a basis for H_dim x H_dim Hermitian matrices.
    The basis contains H_dim^2 matrices:
      - H_dim diagonal matrices: E_{ii}
      - H_dim*(H_dim-1)/2 symmetric off-diagonals: E_{ij}+E_{ji} for i<j
      - H_dim*(H_dim-1)/2 anti-symmetric off-diagonals: i*(E_{ij}-E_{ji}) for i<j
    Returns:
        pauli_basis: JAX array of shape (H_dim**2, H_dim, H_dim).
    """
    basis = []
    # diagonal elements
    for i in range(H_dim):
        M = jnp.zeros((H_dim, H_dim), dtype=jnp.complex128)
        M = M.at[i, i].set(1.0)
        basis.append(M)
    # off-diagonals: symmetric part
    for i in range(H_dim):
        for j in range(i+1, H_dim):
            M = jnp.zeros((H_dim, H_dim), dtype=jnp.complex128)
            M = M.at[i, j].set(1.0)
            M = M.at[j, i].set(1.0)
            basis.append(M)
    # off-diagonals: anti-symmetric part (multiplied by i)
    for i in range(H_dim):
        for j in range(i+1, H_dim):
            M = jnp.zeros((H_dim, H_dim), dtype=jnp.complex128)
            M = M.at[i, j].set(1j)
            M = M.at[j, i].set(-1j)
            basis.append(M)
    return jnp.stack(basis, axis=0)

@jit
def matrix_from_pauli_params(pauli_params, H_dim, pauli_basis):
    """Construct the A_matrix as a linear combination of the pauli basis.
    Args:
        pauli_params: jax array of shape (H_dim**2,) with real coefficients.
        H_dim: Hilbert space dimension.
        pauli_basis: jax array of shape (H_dim**2, H_dim, H_dim) representing the basis.
    Returns:
        A_matrix: jax array of shape (H_dim, H_dim).
    """
    return jnp.tensordot(pauli_params, pauli_basis, axes=([0], [0]))

@partial(jit, static_argnums=(1,2,3))
def initialize_A_array_pauli_jax(key, E_dim, H_dim, scale=0.1):
    """Initialize the pauli parametrisation for all matrices.
    Args:
        key: JAX PRNG key.
        E_dim: Embedding space dimension.
        H_dim: Hilbert space dimension.
        scale: Scaling factor for random initialization.
    Returns:
        pauli_params_array: jax array of shape (E_dim, H_dim**2) containing real coefficients.
    """
    return jax.random.normal(key, (E_dim, H_dim**2)) * scale

# loss function
@partial(jit, static_argnums=(2,3,4))
def loss_from_params(params, X_array, H_dim, parametrization, l2_lambda, pauli_basis=None):
    """Compute the total loss from the parameters and data using the chosen parametrisation.
    Args:
        params: Parameter array.
                - For "upper": shape (E_dim, H_dim*(H_dim+1)//2)
                - For "pauli": shape (E_dim, H_dim**2)
        X_array: jax array of shape (N_points, E_dim).
        H_dim: Hilbert space dimension.
        parametrization: String flag, either "upper" or "pauli".
        l2_lambda: Regularisation coefficient.
        pauli_basis: jax array of shape (H_dim**2, H_dim, H_dim) (used if parametrization=="pauli").
    Returns:
        total_loss: Scalar loss value.
    """
    if parametrization == "upper":
        A_array = jax.vmap(lambda p: matrix_from_upper(p, H_dim))(params)
    elif parametrization == "pauli":
        A_array = jax.vmap(lambda p: matrix_from_pauli_params(p, H_dim, pauli_basis))(params)
    main_loss = total_loss_fn(A_array, X_array)
    #reg_loss = l2_lambda * jnp.sum(params ** 2)
    reg_loss = l2_lambda * jnp.sum(jnp.real(params * params.conj()))
    total_loss = main_loss + reg_loss
    return jnp.real(total_loss)

# error hamiltonian and point mapping
@jit
def error_hamiltonian_jax(X_point, A_array):
    """Calculate the error Hamiltonian given a data point and the A_array configuration.
    Args:
        X_point: jax array of shape (E_dim,).
        A_array: jax array of shape (E_dim, H_dim, H_dim).
    Returns:
        E_matrix: jax array of shape (H_dim, H_dim) representing the error Hamiltonian.
    """
    H_dim = A_array.shape[1]  # Hilbert space dimension
    I = jnp.eye(H_dim, dtype=jnp_complex_dtype)  # shape: (H_dim, H_dim)
    X_point_reshaped = X_point.reshape((-1, 1, 1))  # shape: (E_dim, 1, 1)
    diff = A_array - X_point_reshaped * I  # shape: (E_dim, H_dim, H_dim) via broadcasting
    E_matrix = 0.5 * jnp.einsum('ijl,ilk->jk', diff, diff)
    return E_matrix  # shape: (H_dim, H_dim)

@jit
def compute_ground_state_jax(X_point, A_array):
    """Compute the ground state of the error Hamiltonian for a data point.
    Args:
        X_point: jax array of shape (E_dim,).
        A_array: jax array of shape (E_dim, H_dim, H_dim).
    Returns:
        E_eigvec_0: Ground state eigenvector (jax array of shape (H_dim,)).
        E_eigval_0: Ground state eigenvalue (scalar).
        E_eigvals: All eigenvalues (jax array of shape (H_dim,)).
        E_eigvecs: Eigenvectors (jax array of shape (H_dim, H_dim)).
    """
    E_matrix = error_hamiltonian_jax(X_point, A_array)  # shape: (H_dim, H_dim)
    E_eigvals, E_eigvecs = jnp.linalg.eigh(E_matrix)
    E_eigvec_0 = E_eigvecs[:, 0]
    E_eigval_0 = E_eigvals[0]
    return E_eigvec_0, E_eigval_0, E_eigvals, E_eigvecs

@jit
def point_mapping_jax(psi, A_array):
    """Map a state psi to the embedding space via the function y_mu = Re(<psi|A_mu|psi>).
    Args:
        psi: jax array of shape (H_dim,).
        A_array: jax array of shape (E_dim, H_dim, H_dim).
    Returns:
        y_vals: jax array of shape (E_dim,) representing the embedding.
    """
    A_psi = jnp.matmul(A_array, psi)  # shape: (E_dim, H_dim)
    y_vals = jnp.real(jnp.sum(jnp.conjugate(psi) * A_psi, axis=1))
    return y_vals  # shape: (E_dim,)

@jit
def single_point_loss(A_array, X_point):
    """Compute the loss for a single data point.
    Args:
        A_array: jax array of shape (E_dim, H_dim, H_dim).
        X_point: jax array of shape (E_dim,).
    Returns:
        loss: Scalar loss value for the data point.
    """
    E_eigvec_0, _, _, _ = compute_ground_state_jax(X_point, A_array)
    y = point_mapping_jax(E_eigvec_0, A_array)
    loss = jnp.sum((y - X_point) ** 2)
    return loss

vmapped_loss = vmap(single_point_loss, in_axes=(None, 0), out_axes=0)

@jit
def total_loss_fn(A_array, X_array):
    """Compute the mean loss over all data points.
    Args:
        A_array: jax array of shape (E_dim, H_dim, H_dim).
        X_array: jax array of shape (N_points, E_dim).
    Returns:
        A scalar representing the mean loss.
    """
    losses = vmapped_loss(A_array, X_array)
    return jnp.mean(losses)

# analytical gradient computation (single point)
@partial(jit, static_argnums=(2,3))
def analytic_gradient_single_point_jax(X_point, A_array, H_dim, E_dim):
    """Compute the analytical gradient of the single point loss w.r.t. A_array.
    Args:
        X_point: jax array of shape (E_dim,).
        A_array: jax array of shape (E_dim, H_dim, H_dim).
        H_dim: Hilbert space dimension.
        E_dim: Embedding space dimension.
    Returns:
        grad_A_array: jax array of shape (E_dim, H_dim, H_dim), the gradient dL/dA.
    """
    E_matrix = error_hamiltonian_jax(X_point, A_array)      # (H_dim, H_dim)
    E_vals, E_vecs = jnp.linalg.eigh(E_matrix)  # ascending eigenvalues
    E0 = E_vals[0]  # ground energy
    psi0 = E_vecs[:, 0] # (H_dim,)
    psi_exc = E_vecs[:, 1:] # (H_dim, H_dim-1)
    E_exc = E_vals[1:]  # (H_dim-1,)
    Y_point = point_mapping_jax(psi0, A_array)  # (E_dim,)
    diff = Y_point - X_point # (E_dim,)

    # pseudo-gradient term
    outer00 = jnp.outer(psi0, psi0.conj())  # (H_dim, H_dim)
    grad_term1 = 2 * diff[:, None, None] * outer00[None] # (E_dim, H_dim, H_dim)

    # excited-state coefficients C_n
    # elems[n,mu] = ⟨psi_0| A_mu |psi_n⟩
    elems = jnp.einsum('i,eij,jn->en', psi0.conj(), A_array, psi_exc)  # (E_dim, H_dim-1)
    sigma = jnp.dot(diff, jnp.real(elems))  # (H_dim-1,)
    ediff = E0 - E_exc  # (H_dim-1,)
    inv_ediff = 1.0 / (ediff - jnp.sign(ediff) * 1e-20) # (H_dim-1,)
    C = 2 * sigma * inv_ediff   # (H_dim-1,)

    #correction terms via einsum
    I = jnp.eye(H_dim, dtype=A_array.dtype)
    A_minus = A_array - X_point[:, None, None] * I  # (E_dim, H_dim, H_dim)

    # Prepare |psi_0><psi_n| and |psi_n><psi_0|
    # outer_0n[n,i,j] = psi0[i] * conj( psi_exc[j,n] )
    outer_0n = jnp.einsum('i,jn->nij', psi0, psi_exc.conj()) # shape: (H_dim-1, H_dim, H_dim)
    # outer_n0[n,i,j] = psi_exc[i,n] * conj( psi0[j] )
    outer_n0 = jnp.einsum('in,j->nij', psi_exc, psi0.conj()) # shape: (H_dim-1, H_dim, H_dim)

    # grad_term2[l,a,c] = 0.5 * sum_n [ C[n] * (A_minus[l] @ outer_0n[n] + outer_n0[n] @ A_minus[l]) ]
    term1 = jnp.einsum('n,lab,nbc->lac', C, A_minus, outer_0n)  # (E_dim, H_dim, H_dim)
    term2 = jnp.einsum('n,nab,lbc->lac', C, outer_n0, A_minus)  # (E_dim, H_dim, H_dim)
    grad_term2 = 0.5 * (term1 + term2)  # (E_dim, H_dim, H_dim)

    # total gradient
    grad_A_array = grad_term1 + grad_term2  # (E_dim, H_dim, H_dim)
    return grad_A_array

# training Functions
def train_A_array_analytic_jax(X_array, A_array_init, H_dim, E_dim, num_epochs, lr, l2_lambda):
    """Train the parameters using the full analytical gradient.
    Args:
        X_array: jax array of shape (N_points, E_dim).
        A_array_init: Initial A_array, jax array of shape (E_dim, H_dim, H_dim).
        H_dim: Hilbert space dimension.
        E_dim: Embedding space dimension.
        num_epochs: Number of training iterations.
        lr: Learning rate.
        l2_lambda: L2 regularisation coefficient.
    Returns:
        A_array_final: Optimized A_array of shape (E_dim, H_dim, H_dim).
    """
    print(f"Starting training with Analytical Gradient for {num_epochs} epochs (lr={lr}, l2={l2_lambda})...")
    A_array = A_array_init
    N_points = X_array.shape[0]

    batch_analytic_grad_func = jax.vmap(analytic_gradient_single_point_jax, in_axes=(0, None, None, None), out_axes=0)

    @jit
    def update_step(current_A_array, X_batch):
        # Compute the analytical gradient for each data point in the batch
        batch_grad = batch_analytic_grad_func(X_batch, current_A_array, H_dim, E_dim) # shape: (N_points, E_dim, H_dim, H_dim)
        avg_analytic_grad = jnp.mean(batch_grad, axis=0)  # shape: (E_dim, H_dim, H_dim)

        # L2 regularisation gradient (gradient of 0.5 * l2 * sum(|A_ij|^2) is l2 * A_ij)
        # Gradient of l2_lambda * ||A||_F^2 = l2_lambda * sum |A_mu,ij|^2
        reg_grad = 2 * l2_lambda * current_A_array # shape: (E_dim, H_dim, H_dim)

        total_grad = avg_analytic_grad + reg_grad # shape: (E_dim, H_dim, H_dim)

        updated_A_array = current_A_array - lr * total_grad
        return updated_A_array, total_grad

    for epoch in range(num_epochs):
        A_array, grad = update_step(A_array, X_array)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            # compute loss (can be expensive, maybe do less often)
            loss_val = total_loss_fn(A_array, X_array)
            reg_loss_val = l2_lambda * jnp.sum(jnp.real(A_array * A_array.conj()))
            total_loss_val = loss_val + reg_loss_val
            grad_norm = jnp.linalg.norm(grad)
            print(f"Analytic Epoch {epoch}: Total Loss = {total_loss_val:.6f} (Main={loss_val:.6f}, Reg={reg_loss_val:.6f}), Grad Norm = {grad_norm:.6f}")

    print("Analytical gradient training finished.")
    return A_array  # shape: (E_dim, H_dim, H_dim)

def train_A_array_LBFGS(X_array, A_array_init, H_dim, E_dim, num_epochs, lr, l2_lambda):
    """Train the parameters using the LBFGS solver.
    Args:
        X_array: jax array of shape (N_points, E_dim).
        A_array_init: Initial A_array, jax array of shape (E_dim, H_dim, H_dim).
        H_dim: Hilbert space dimension.
        E_dim: Embedding space dimension.
        num_epochs: Number of training iterations.
        lr: Learning rate.
        l2_lambda: L2 regularisation coefficient.
    Returns:
        A_array_final: Optimized A_array of shape (E_dim, H_dim, H_dim).
    """
    print(f"Starting training with Analytical Gradient for {num_epochs} epochs (l2={l2_lambda})...")
    A_array = A_array_init
    N_points = X_array.shape[0]

    A_flat_init = jnp.reshape(A_array_init, (-1,))

    def objective(params_flat):
        A_array = jnp.reshape(params_flat, (E_dim, H_dim, H_dim))
        loss_val = total_loss_fn(A_array, X_array)
        reg_loss_val = l2_lambda * jnp.sum(jnp.real(A_array * A_array.conj()))
        return loss_val + reg_loss_val

    # initialize BFGS solver
    #solver = jaxopt.LBFGS(fun=objective)
    #solver = jaxopt.LBFGS(fun=objective, maxiter=num_epochs, tol=1e-6, verbose=2)
    solver = jaxopt.LBFGS(fun=objective, maxiter=num_epochs, verbose=2)
    sol = solver.run(init_params=A_flat_init)

    A_array_final = jnp.reshape(sol.params, (E_dim, H_dim, H_dim))

    return A_array_final  # shape: (E_dim, H_dim, H_dim)

def train_A_array_jaxopt(X_array, params_array_init, H_dim, num_epochs, lr, l2_lambda, parametrization, pauli_basis=None):
    """Train the parameters using the jaxopt solver (LBFGS).
    Args:
        X_array: jax array of shape (N_points, E_dim).
        params_array_init: Initial parameter array.
            - For "upper": shape (E_dim, H_dim*(H_dim+1)//2)
            - For "pauli": shape (E_dim, H_dim**2)
        H_dim: Hilbert space dimension.
        num_epochs: Number of training iterations.
        lr: Learning rate.
        l2_lambda: L2 regularisation coefficient.
        parametrization: String flag; "upper" or "pauli".
        pauli_basis: jax array for the pauli basis (required if parametrization=="pauli").
    Returns:
        A_array_final: Optimized A_array of shape (E_dim, H_dim, H_dim).
    """
    print(f"Starting training for {num_epochs} iterations with lr={lr}")
    solver = jaxopt.LBFGS(
        fun=lambda params, X_array: loss_from_params(params, X_array, H_dim, parametrization, l2_lambda, pauli_basis),
        maxiter=num_epochs,
        #stepsize=lr,
        verbose=True
    )
    res = solver.run(init_params=params_array_init, X_array=X_array)
    params_final = res.params  # shape depends on parametrisation
    if parametrization == "upper":
        A_array_final = jax.vmap(lambda p: matrix_from_upper(p, H_dim))(params_final)
    elif parametrization == "pauli":
        A_array_final = jax.vmap(lambda p: matrix_from_pauli_params(p, H_dim, pauli_basis))(params_final)
    final_loss = total_loss_fn(A_array_final, X_array)  # scalar
    print(f"Training finished. Final Loss: {final_loss:.6f}")
    return A_array_final  # shape: (E_dim, H_dim, H_dim)

def train_A_array_optax(X_array, params_array_init, H_dim, num_epochs,
                        lr, l2_lambda, grad_clip_norm, transition_steps, decay_rate,
                        parametrization, pauli_basis=None):
    """Train the parameters using the optax optimizer.
    Args:
        X_array: jax array of shape (N_points, E_dim).
        params_array_init: Initial parameter array.
            - For "upper": shape (E_dim, H_dim*(H_dim+1)//2)
            - For "pauli": shape (E_dim, H_dim**2)
        H_dim: Hilbert space dimension.
        num_epochs: Number of training iterations.
        lr: Learning rate.
        l2_lambda: L2 regularisation coefficient.
        grad_clip_norm: Gradient clipping norm.
        transition_steps: Transition steps for the learning rate schedule.
        decay_rate: Decay rate for the learning rate schedule.
        parametrization: String flag; "upper" or "pauli".
        pauli_basis: jax array for the pauli basis (required if parametrization=="pauli").
    Returns:
        A_array_final: Optimized A_array of shape (E_dim, H_dim, H_dim).
    """
    print(f"Starting training.. ")
    lr_schedule = optax.exponential_decay(init_value=lr, transition_steps=transition_steps,
                                          decay_rate=decay_rate, staircase=False)
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(learning_rate=lr_schedule)
    )
    params = params_array_init  # shape depends on parametrisation
    opt_state = optimizer.init(params)
    @jax.jit
    def train_step(params, opt_state, X_array):
        loss, grads = jax.value_and_grad(loss_from_params)(params, X_array, H_dim, parametrization, l2_lambda, pauli_basis)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    for epoch in range(num_epochs):
        params, opt_state, loss = train_step(params, opt_state, X_array)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    if parametrization == "upper":
        A_array_final = jax.vmap(lambda p: matrix_from_upper(p, H_dim))(params)
    elif parametrization == "pauli":
        A_array_final = jax.vmap(lambda p: matrix_from_pauli_params(p, H_dim, pauli_basis))(params)
    final_loss = total_loss_fn(A_array_final, X_array)  # scalar
    print(f"Training finished. Final Loss: {final_loss:.6f}")
    return A_array_final  # shape: (E_dim, H_dim, H_dim)

def train_A_array_pseudo_jax(X_array, A_array_init, H_dim, num_epochs, lr, l2_lambda):
    """Train the parameters using a pseudo gradient approach.
    gradient with respect to A_mu is approximated as:
        2*(<psi_0|A_mu|psi_0> - x_mu) * |psi_0><psi_0|
    Args:
        X_array: jax array of shape (N_points, E_dim).
        A_array_init: Initial A_array, jax array of shape (E_dim, H_dim, H_dim).
        H_dim: Hilbert space dimension.
        num_epochs: Number of training iterations.
        lr: Learning rate.
        l2_lambda: L2 regularisation coefficient.
    Returns:
        A_array_final: Optimized A_array of shape (E_dim, H_dim, H_dim).
    """
    A_array = A_array_init
    N_points = X_array.shape[0]
    for epoch in range(num_epochs):
        # Compute the pseudo gradient for each data point
        def pseudo_grad_one(X_point):
            psi0, _, _, _ = compute_ground_state_jax(X_point, A_array)
            y = point_mapping_jax(psi0, A_array)
            # |psi0><psi0|
            outer_prod = jnp.outer(psi0, psi0.conj())
            # For each A_mu: gradient = 2*(y[mu] - X_point[mu]) * |psi0><psi0|
            return 2 * (y - X_point).reshape((-1, 1, 1)) * outer_prod  # shape: (E_dim, H_dim, H_dim)
        batch_grad = jax.vmap(pseudo_grad_one)(X_array)  # shape: (N_points, E_dim, H_dim, H_dim)
        avg_grad = jnp.mean(batch_grad, axis=0)  # shape: (E_dim, H_dim, H_dim)
        # L2 regularisation gradient
        reg_grad = 2 * l2_lambda * A_array
        total_grad = avg_grad + reg_grad
        A_array = A_array - lr * total_grad
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            loss_val = total_loss_fn(A_array, X_array)
            print(f"Pseudo Epoch {epoch}: Loss = {loss_val:.6f}")
    return A_array  # shape: (E_dim, H_dim, H_dim)

# intrinsic dimension estimation using quantum metric
@jit
def quantum_metric_jax(E_eigvec_0, E_eigval_0, E_eigvals, E_eigvecs, A_array):
    """Compute the quantum metric matrix for the ground state.
    Args:
        E_eigvec_0: Ground state eigenvector (jax array of shape (H_dim,)).
        E_eigval_0: Ground state eigenvalue (scalar).
        E_eigvals: All eigenvalues (jax array of shape (H_dim,)).
        E_eigvecs: Eigenvectors (jax array of shape (H_dim, H_dim)).
        A_array: jax array of shape (E_dim, H_dim, H_dim).
    Returns:
        G_matrix: Quantum metric matrix (jax array of shape (E_dim, E_dim)).
    """
    H_dim = E_eigvecs.shape[0]  # scalar: Hilbert space dimension
    E_dim = A_array.shape[0]    # scalar: Embedding space dimension
    A_mu_E_psi_0 = jnp.matmul(A_array, E_eigvec_0)  # shape: (E_dim, H_dim)
    matrix_elems = jnp.dot(E_eigvecs.conj().T, A_mu_E_psi_0.T)  # shape: (H_dim, E_dim)
    excited_state_elems = matrix_elems[1:, :]  # shape: (H_dim-1, E_dim)
    energy_diffs = E_eigvals[1:] - E_eigval_0  # shape: (H_dim-1,)
    epsilon = 1e-20  # small constant
    inv_energy_diffs = 1.0 / (energy_diffs + epsilon)  # shape: (H_dim-1,)
    G_matrix = 2 * jnp.real(jnp.einsum('ij,i,ik->jk', excited_state_elems.conj(), inv_energy_diffs, excited_state_elems))
    return G_matrix  # shape: (E_dim, E_dim)

@jit
def estimate_single_point_jax(X_point, A_array):
    """Estimate the intrinsic dimension and related quantities for a single data point.
    Args:
        X_point: jax array of shape (E_dim,).
        A_array: jax array of shape (E_dim, H_dim, H_dim).
    Returns:
        I_dim: Estimated intrinsic dimension (scalar).
        Y_point: Mapping of the ground state to the embedding space (jax array of shape (E_dim,)).
        E_eigval_0: Ground state eigenvalue (scalar).
        G_eigvals: Sorted eigenvalues of the quantum metric (jax array of shape (E_dim,)).
    """
    E_eigvec_0, E_eigval_0, E_eigvals, E_eigvecs = compute_ground_state_jax(X_point, A_array)
    Y_point = point_mapping_jax(E_eigvec_0, A_array)
    G_matrix = quantum_metric_jax(E_eigvec_0, E_eigval_0, E_eigvals, E_eigvecs, A_array)
    G_eigvals_complex = jnp.linalg.eigh(G_matrix)[0]  # shape: (E_dim,)
    G_eigvals = jnp.real(G_eigvals_complex)  # shape: (E_dim,)
    G_eigvals = jnp.sort(G_eigvals)  # shape: (E_dim,)
    tolerance = 1e-12
    G_eigvals = jnp.where(G_eigvals < tolerance, 0, G_eigvals)  # shape: (E_dim,)
    epsilon_ratio = 1e-20
    denominators = jnp.maximum(epsilon_ratio, G_eigvals[:-1])  # shape: (E_dim-1,)
    numerators = G_eigvals[1:]  # shape: (E_dim-1,)
    ratios = numerators / denominators  # shape: (E_dim-1,)
    gamma_index = jnp.argmax(ratios) + 1  # scalar index
    I_dim = A_array.shape[0] - gamma_index  # scalar (estimated intrinsic dimension)
    return I_dim, Y_point, E_eigval_0, G_eigvals

@jit
def I_dimension_estimator_jax(X_array, A_array):
    """Estimate the intrinsic dimension for each data point in X_array.
    Args:
        X_array: jax array of shape (N_points, E_dim).
        A_array: jax array of shape (E_dim, H_dim, H_dim).
    Returns:
        I_dim_array: jax array of shape (N_points,) with intrinsic dimension estimates.
        Y_array: jax array of shape (N_points, E_dim) of embedded points.
        E_eigval_0_array: jax array of shape (N_points,) of ground state eigenvalues.
        G_eigvals_array: jax array of shape (N_points, E_dim) of quantum metric eigenvalues.
    """
    I_dim_array, Y_array, E_eigval_0_array, G_eigvals_array = jax.vmap(
            estimate_single_point_jax, in_axes=(0, None))(X_array, A_array)
    return I_dim_array, Y_array, E_eigval_0_array, G_eigvals_array

# plotting functions
def plot_point_cloud(Y_array_np, X_array_np, E_eigval_0_array_np, noise_level, solver, dataset_type, save=False, filename=None):
    """
    Plot the point cloud obtained from the embedding together with the original data.
    Args:
        Y_array_np: Numpy array of shape (N_points, E_dim) from the embedding.
        X_array_np: Numpy array of shape (N_points, E_dim) of original data.
        E_eigval_0_array_np: Numpy array of ground state eigenvalues.
        noise_level: Noise level used in data generation.
        solver: Solver used in experiment.
        dataset_type: Type of dataset used.
        save: Boolean flag for saving the plot.
        filename: File name for saving the plot.
    Returns:
        None.
    """
    E_dim = Y_array_np.shape[1]
    num_pairs = E_dim // 2

    E_eigval_0_norm = (E_eigval_0_array_np - np.min(E_eigval_0_array_np)) / (
        np.max(E_eigval_0_array_np) - np.min(E_eigval_0_array_np) + 1e-8
    )

    ncols = int(np.ceil(np.sqrt(num_pairs)))
    nrows = int(np.ceil(num_pairs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axes = np.array(axes).flatten()

    for i in range(num_pairs):
        dim1 = 2 * i
        dim2 = 2 * i + 1
        ax = axes[i]
        ax.scatter(Y_array_np[:, dim1], Y_array_np[:, dim2], c=1 - E_eigval_0_norm,
                   cmap='viridis', label='Point Cloud', s=25)

        if X_array_np is not None:
            ax.scatter(X_array_np[:, dim1], X_array_np[:, dim2], color='red',
                       alpha=0.4, label='Original Data', s=5)

        ax.set_xlabel(f"$y({dim1+1})$")
        ax.set_ylabel(f"$y({dim2+1})$")
        ax.set_title(f"Dimensions {dim1 + 1} vs {dim2 + 1}")
        ax.grid(True)
        ax.legend(loc='upper left')

        # Ensure perfect square aspect ratio
        ax.set_aspect('equal', adjustable='box')

    for i in range(num_pairs, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f"Point Cloud (Noise = {noise_level}) \n [Dataset: {dataset_type}, Solver: {solver}]", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    if save and filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_mean_eigenvalues(G_eigvals_array_np, noise_level, solver, dataset_type, save=False, filename=None):
    """Plot the mean eigenvalues of the quantum metric over all data points with error bars,
    Args:
        G_eigvals_array_np: Numpy array of shape (N_points, E_dim).
        noise_level: Noise level used in data generation.
        save: Boolean flag indicating whether to save the plot.
        filename: File name for saving the plot.
    Returns:
        None.
    """
    mean_eigs = np.mean(G_eigvals_array_np, axis=0)
    std_eigs = np.std(G_eigvals_array_np, axis=0)
    indices = np.arange(len(mean_eigs))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    err_line1 = ax1.errorbar(indices, mean_eigs, yerr=std_eigs, fmt='o', capsize=5,
                             label='Linear Scale')
    ax1.set_xlabel("Eigenvalue Index")
    ax1.set_ylabel("Mean Eigenvalue (Linear)")
    ax1.grid(True)
    ax2 = ax1.twinx()
    err_line2 = ax2.errorbar(indices, mean_eigs, yerr=std_eigs, fmt='s', capsize=5, alpha=0.3,
                             color='black', label='Log Scale')
    ax2.set_yscale('log')
    ax2.set_ylabel("Mean Eigenvalue (Log)")
    ax1.set_title(f"Mean Quantum Metric Eigenvalues (Noise = {noise_level}) \n [Dataset: {dataset_type},Solver: {solver}]")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='best')
    plt.tight_layout()
    if save and filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_quantum_metric_spectra(G_eigvals_array, noise_level, save=False, filename=None):
    """Plot the eigenvalue spectra of the quantum metric for all data points.
    Args:
        G_eigvals_array: jax array of shape (N_points, E_dim).
        noise_level: Noise level used in data generation.
        save: Boolean flag for saving the plot.
        filename: File name for saving the plot.
    Returns:
        None.
    """
    G_eigvals_array_np = np.array(G_eigvals_array)
    num_points = G_eigvals_array_np.shape[0]
    E_dim = G_eigvals_array_np.shape[1]
    plt.figure(figsize=(8, 5))
    markers = ['o', 's', '^', 'v', 'D', 'x', '*', 'p', 'h', '<', '>']
    for i in range(E_dim):
        marker = markers[i % len(markers)]
        plt.plot(range(num_points), G_eigvals_array_np[:, i], marker=marker, markersize=4, linestyle='-', label=f"Eigenvalue {i}")
    plt.title(f"Quantum Metric Spectra (Noise = {noise_level})")
    plt.xlabel("Data Point Index")
    plt.ylabel("Eigenvalues of $g(x)$")
    plt.legend()
    plt.grid(True)
    if save and filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_I_dim_array(I_dim_array, noise_level, save=False, filename=None):
    """Plot the intrinsic dimension estimates as a function of the data point index.
    Args:
        I_dim_array: Numpy array of shape (N_points,).
        noise_level: Noise level used in data generation.
        save: Boolean flag for saving the plot.
        filename: File name for saving the plot.
    Returns:
        None.
    """
    I_dim_array_np = np.array(I_dim_array)
    plt.figure(figsize=(8, 5))
    indices = range(len(I_dim_array_np))
    plt.plot(indices, I_dim_array_np, marker='o', linestyle='-', label='Intrinsic Dimension Estimate')
    plt.title(f"Intrinsic Dimension Estimates (Noise = {noise_level})")
    plt.xlabel("Data Point Index")
    plt.ylabel("Estimated Intrinsic Dimension")
    unique_dims = np.unique(I_dim_array_np)
    if np.all(np.mod(unique_dims, 1) == 0):
         plt.yticks(np.arange(int(min(unique_dims)), int(max(unique_dims)) + 1))
    plt.legend()
    plt.grid(True)
    if save and filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_I_dim_array_hist(I_dim_array, noise_level, save=False, filename=None):
    """Plot a histogram of the intrinsic dimension estimates.
    Args:
        I_dim_array: Numpy array of shape (N_points,).
        noise_level: Noise level used in data generation.
        save: Boolean flag for saving the plot.
        filename: File name for saving the plot.
    Returns:
        None.
    """
    d_vals = np.array(I_dim_array)
    if len(d_vals) == 0:
        print("Warning: I_dim_array is empty, skipping histogram plot.")
        return
    min_d = d_vals.min()
    max_d = d_vals.max()
    bins = np.arange(min_d - 0.5, max_d + 1.5, 1)
    plt.figure(figsize=(8, 5))
    plt.hist(d_vals, bins=bins, edgecolor='black', rwidth=0.8, alpha=0.75)
    plt.title(f"Histogram of Intrinsic Dimension Estimates (Noise = {noise_level})")
    plt.xlabel("Estimated Intrinsic Dimension")
    plt.ylabel("Count")
    plt.xticks(np.arange(int(min_d), int(max_d) + 1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if save and filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

# -------------------------
# Experiment Runner
# -------------------------
def run_experiment(key, solver, dataset_type, noise_level,
                   parametrization, N_points, H_dim, E_dim,
                   num_epochs, A_init_scale, lr, l2_lambda,
                   grad_clip_norm, transition_steps, decay_rate):
    """Run the full experiment: data generation, parameter initialization, training, and plotting.
    Args:
        key: JAX random key.
        solver: Solver to use ("jaxopt", "optax", or "pseudo").
        dataset_type: String identifier for the dataset type.
        noise_level: Noise level to add to the data.
        N_points: Number of data points.
        H_dim: Hilbert space dimension.
        E_dim: Embedding space dimension.
        num_epochs: Number of training iterations.
        A_init_scale: Scale for initial random parameters.
        lr: Learning rate.
        l2_lambda: L2 regularisation coefficient.
        grad_clip_norm: Gradient clipping norm (for optax).
        transition_steps: Transition steps for learning rate scheduling.
        decay_rate: Decay rate for learning rate scheduling.
        parametrization: Either "upper" for the original or "pauli" for the pauli basis parametrisation.
    Returns:
        I_dim_array_np, Y_array_np, E_eigval_0_array_np, G_eigvals_array_np:
            Numpy arrays with the intrinsic dimension, embedding, ground state eigenvalues, and quantum metric eigenvalues.
    """
    save_folder = f"plots_jax16/{solver}_{dataset_type}_noise{noise_level:.2f}/"
    os.makedirs(save_folder, exist_ok=True)
    print("-" * 50)

    # Generate data
    key, subkey_data, subkey_init = jax.random.split(key, 3)
    X_array = (lambda key, N, noise, dtype: globals()["gen_" + dtype + "_data"](key, N, noise=noise))(
        subkey_data, N_points, noise_level, dataset_type)

    # initialize A_array (needed for analytic solver) and parameters
    A_array_init_full = initialize_A_array_jax(subkey_init, E_dim, H_dim, scale=A_init_scale)
    pauli_basis_tensor = None # only needed for pauli param
    params_array_init = None # not needed for analytic, needed for others

    # initialize full Hermitian matrices or parameters.
    if parametrization == "upper":
        A_array_init = initialize_A_array_jax(subkey_init, E_dim, H_dim, scale=A_init_scale)
        params_array_init = upper_from_matrix(A_array_init, H_dim)
        pauli_basis_tensor = None
    elif parametrization == "pauli":
        params_array_init = initialize_A_array_pauli_jax(subkey_init, E_dim, H_dim, scale=A_init_scale)
        pauli_basis_tensor = generate_pauli_basis(H_dim)
        if solver == "analytic":
           print("using analytic solver with pauli parametrization")
           A_array_init_full = jax.vmap(lambda p: matrix_from_pauli_params(p, H_dim, pauli_basis_tensor))(params_array_init)

    # train
    if solver == "jaxopt":
        A_array_opt = train_A_array_jaxopt(X_array, params_array_init, H_dim,
                                           num_epochs=num_epochs, lr=lr, l2_lambda=l2_lambda,
                                           parametrization=parametrization, pauli_basis=pauli_basis_tensor)
    elif solver == "optax":
        A_array_opt = train_A_array_optax(X_array, params_array_init, H_dim,
                                          num_epochs=num_epochs, lr=lr, l2_lambda=l2_lambda,
                                          grad_clip_norm=grad_clip_norm, transition_steps=transition_steps,
                                          decay_rate=decay_rate, parametrization=parametrization, pauli_basis=pauli_basis_tensor)
    elif solver == "pseudo":
        if parametrization != "upper":
            raise ValueError("Pseudo gradient is only for 'upper' parametrisation.")
        A_array_opt = train_A_array_pseudo_jax(X_array, A_array_init, H_dim, num_epochs=num_epochs, lr=lr, l2_lambda=l2_lambda)

    elif solver == "analytic":
        A_array_opt = train_A_array_analytic_jax(X_array, A_array_init_full, H_dim, E_dim, num_epochs=num_epochs, lr=lr, l2_lambda=l2_lambda)
    elif solver == "LBFGS":
        A_array_opt = train_A_array_LBFGS(X_array, A_array_init_full, H_dim, E_dim, num_epochs=num_epochs, lr=lr, l2_lambda=l2_lambda)
    else:
        raise ValueError(f"unknown solver: {solver}")

    #TODO need to recheck I_dim_array calculation
    print("Estimating intrinsic dimension for all points...")
    I_dim_array, Y_array, E_eigval_0_array, G_eigvals_array = I_dimension_estimator_jax(X_array, A_array_opt)
    I_dim_array_np = np.array(I_dim_array)
    Y_array_np = np.array(Y_array)
    X_array_np = np.array(X_array)
    E_eigval_0_array_np = np.array(E_eigval_0_array)
    G_eigvals_array_np = np.array(G_eigvals_array)
    mean_d = jnp.mean(I_dim_array)
    median_d = jnp.median(I_dim_array)
    mode_d = jnp.argmax(jnp.bincount(I_dim_array.astype(int))) if len(I_dim_array) > 0 else 'N/A'
    #print(f"Mean intrinsic dimension estimate: {mean_d:.4f}")
    #print(f"Median intrinsic dimension estimate: {median_d:.4f}")
    #print(f"Mode intrinsic dimension estimate: {mode_d}")
    print("Generating plots...")

    #plots
    plot_point_cloud(Y_array_np, X_array_np, E_eigval_0_array_np, noise_level, solver, dataset_type, save=True,
                     filename=os.path.join(save_folder, f"point_cloud_noise_{noise_level:.2f}.png"))
    plot_mean_eigenvalues(G_eigvals_array_np, noise_level, solver, dataset_type, save=True,
                          filename=os.path.join(save_folder, f"mean_eigenvalues_noise_{noise_level:.2f}.png"))
    #plot_quantum_metric_spectra(G_eigvals_array_np, noise_level, save=True,
    #                            filename=os.path.join(save_folder, f"quantum_metric_spectra_noise_{noise_level:.2f}.png"))
    #plot_I_dim_array(I_dim_array_np, noise_level, save=True,
    #                 filename=os.path.join(save_folder, f"I_dim_array_noise_{noise_level:.2f}.png"))
    #plot_I_dim_array_hist(I_dim_array_np, noise_level, save=True,
    #                      filename=os.path.join(save_folder, f"I_dim_array_hist_noise_{noise_level:.2f}.png"))
    print(f"Plots saved to {save_folder}")
    print("-" * 50)
    return I_dim_array_np, Y_array_np, E_eigval_0_array_np, G_eigvals_array_np

# main
if __name__ == "__main__":
    dataset_dims = {
        "sphere": 3,
        "cubic": 18,
        "campadelli_beta": 40,
        "campadelli_n": 72
    }

    solver = "LBFGS"      # "optax","pseudo","analytic" or "LBFGS"
    parametrization = "upper"  # "pauli" or "upper" (pseudo gradient and LBFGS only for "upper")
    N_points = 1000
    H_dim = 16
    epochs = 100
    A_init_scale = 5.0
    learning_rate = 0.9       # not used by LBFGS
    l2_lambda = 1e-4
    grad_clip_norm = 1.0      # only for optax
    transition_steps = 25     # only for optax
    decay_rate = 0.99         # only for optax

    main_key = jax.random.PRNGKey(137)

    #for dataset_type in ["sphere", "cubic", "campadelli_beta", "campadelli_n"]:
    for dataset_type in ["sphere"]:
        E_dim = dataset_dims[dataset_type]

        for noise_level in [0.0, 0.1, 0.2]:
            main_key, subkey = jax.random.split(main_key)
            key_run, _ = jax.random.split(subkey)

            # run the experiment
            results = run_experiment(
                key=key_run,
                solver=solver,
                dataset_type=dataset_type,
                noise_level=noise_level,
                parametrization=parametrization,
                N_points=N_points,
                H_dim=H_dim,
                E_dim=E_dim,
                num_epochs=epochs,
                A_init_scale=A_init_scale,
                lr=learning_rate,
                l2_lambda=l2_lambda,
                grad_clip_norm=grad_clip_norm,
                transition_steps=transition_steps,
                decay_rate=decay_rate
            )

            # print a summary for this run
            print("\nRunning Experiment:")
            print(f"  solver          = {solver}")
            print(f"  dataset_type    = {dataset_type}")
            print(f"  noise_level     = {noise_level}")
            print(f"  parametrization = {parametrization}")
            print(f"  N_points        = {N_points}")
            print(f"  H_dim           = {H_dim}")
            print(f"  E_dim           = {E_dim}")
            print(f"  epochs          = {epochs}")
            print(f"  A_init_scale    = {A_init_scale}")

            if solver == "optax":
                print(f"  learning_rate     = {learning_rate}")
                print(f"  l2_lambda         = {l2_lambda}")
                print(f"  grad_clip_norm    = {grad_clip_norm}")
                print(f"  transition_steps  = {transition_steps}")
                print(f"  decay_rate        = {decay_rate}")
            elif solver in ["LBFGS"]:
                print(f"  l2_lambda         = {l2_lambda}")
            else:
                print(f"  learning_rate     = {learning_rate}")
                print(f"  l2_lambda         = {l2_lambda}")
