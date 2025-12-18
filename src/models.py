import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# Determine dtype based on backend (similar to original code)
if jax.default_backend() == "tpu":
    jnp_complex_dtype = jnp.complex64
    jax.config.update("jax_enable_x64", False)
else:
    jnp_complex_dtype = jnp.complex128
    jax.config.update("jax_enable_x64", True)

# --- Matrix Parametrization ---

# upper triangular + diagonal parametrisation
@partial(jit, static_argnums=(1,2))
def initialize_A_matrix_jax(key, H_dim, scale=0.1):
    """Initialize a Hermitian A_matrix with random uptri + diag parameters."""
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
    """Initialize an array of Hermitian A_matrices."""
    keys = jax.random.split(key, E_dim)
    A_array = jax.vmap(lambda k: initialize_A_matrix_jax(k, H_dim, scale=scale))(keys)
    return A_array  # shape: (E_dim, H_dim, H_dim)

@partial(jit, static_argnums=(1,))
def matrix_from_upper(upper_params, H_dim):
    """Reconstruct a Hermitian matrix from its uptri + diag parameters."""
    A_matrix = jnp.zeros((H_dim, H_dim), dtype=jnp_complex_dtype)
    A_matrix = A_matrix.at[jnp.triu_indices(H_dim, k=0)].set(upper_params)  # includes diagonal
    A_matrix = A_matrix + jnp.triu(A_matrix, k=1).conj().T  # excludes diagonal
    return A_matrix  # shape: (H_dim, H_dim)

@partial(jit, static_argnums=(1,))
def upper_from_matrix(A_array, H_dim):
    """Extract uptri + diag parameters from a Hermitian matrix."""
    i_upper, j_upper = jnp.triu_indices(H_dim, k=0) #includes diagonal
    return A_array[:, i_upper, j_upper]  # shape: (E_dim, H_dim*(H_dim+1)//2)

# Pauli basis parametrisation
def generate_pauli_basis(H_dim):
    """Generate a basis for H_dim x H_dim Hermitian matrices."""
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
    """Construct the A_matrix as a linear combination of the pauli basis."""
    return jnp.tensordot(pauli_params, pauli_basis, axes=([0], [0]))

@partial(jit, static_argnums=(1,2,3))
def initialize_A_array_pauli_jax(key, E_dim, H_dim, scale=0.1):
    """Initialize the pauli parametrisation for all matrices."""
    return jax.random.normal(key, (E_dim, H_dim**2)) * scale

# --- Hamiltonian & Logic ---

@jit
def error_hamiltonian_jax(X_point, A_array):
    """Calculate the error Hamiltonian given a data point and the A_array configuration."""
    H_dim = A_array.shape[1]  # Hilbert space dimension
    I = jnp.eye(H_dim, dtype=jnp_complex_dtype)  # shape: (H_dim, H_dim)
    X_point_reshaped = X_point.reshape((-1, 1, 1))  # shape: (E_dim, 1, 1)
    diff = A_array - X_point_reshaped * I  # shape: (E_dim, H_dim, H_dim) via broadcasting
    E_matrix = 0.5 * jnp.einsum('ijl,ilk->jk', diff, diff)
    return E_matrix  # shape: (H_dim, H_dim)

@jit
def compute_ground_state_jax(X_point, A_array):
    """Compute the ground state of the error Hamiltonian for a data point."""
    E_matrix = error_hamiltonian_jax(X_point, A_array)  # shape: (H_dim, H_dim)
    E_eigvals, E_eigvecs = jnp.linalg.eigh(E_matrix)
    E_eigvec_0 = E_eigvecs[:, 0]
    E_eigval_0 = E_eigvals[0]
    return E_eigvec_0, E_eigval_0, E_eigvals, E_eigvecs

@jit
def point_mapping_jax(psi, A_array):
    """Map a state psi to the embedding space via the function y_mu = Re(<psi|A_mu|psi>)."""
    A_psi = jnp.matmul(A_array, psi)  # shape: (E_dim, H_dim)
    y_vals = jnp.real(jnp.sum(jnp.conjugate(psi) * A_psi, axis=1))
    return y_vals  # shape: (E_dim,)

# intrinsic dimension estimation using quantum metric
@jit
def quantum_metric_jax(E_eigvec_0, E_eigval_0, E_eigvals, E_eigvecs, A_array):
    """Compute the quantum metric matrix for the ground state."""
    #H_dim = E_eigvecs.shape[0]  # scalar: Hilbert space dimension
    #E_dim = A_array.shape[0]    # scalar: Embedding space dimension
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
    """Estimate the intrinsic dimension and related quantities for a single data point."""
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
    """Estimate the intrinsic dimension for each data point in X_array."""
    I_dim_array, Y_array, E_eigval_0_array, G_eigvals_array = jax.vmap(
            estimate_single_point_jax, in_axes=(0, None))(X_array, A_array)
    return I_dim_array, Y_array, E_eigval_0_array, G_eigvals_array
