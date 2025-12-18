import jax
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap
from src.models import (
    matrix_from_upper, matrix_from_pauli_params,
    compute_ground_state_jax, point_mapping_jax, error_hamiltonian_jax
)


@jit
def single_point_loss(A_array, X_point):
    """Compute the loss for a single data point."""
    E_eigvec_0, _, _, _ = compute_ground_state_jax(X_point, A_array)
    y = point_mapping_jax(E_eigvec_0, A_array)
    loss = jnp.sum((y - X_point) ** 2)
    return loss

vmapped_loss = vmap(single_point_loss, in_axes=(None, 0), out_axes=0)

@jit
def total_loss_fn(A_array, X_array):
    """Compute the mean loss over all data points."""
    losses = vmapped_loss(A_array, X_array)
    return jnp.mean(losses)

@partial(jit, static_argnums=(2,3,4))
def loss_from_params(params, X_array, H_dim, parametrization, l2_lambda, pauli_basis=None):
    """Compute the total loss from the parameters and data using the chosen parametrisation."""
    if parametrization == "upper":
        A_array = jax.vmap(lambda p: matrix_from_upper(p, H_dim))(params)
    elif parametrization == "pauli":
        A_array = jax.vmap(lambda p: matrix_from_pauli_params(p, H_dim, pauli_basis))(params)
    main_loss = total_loss_fn(A_array, X_array)
    #reg_loss = l2_lambda * jnp.sum(params ** 2)
    reg_loss = l2_lambda * jnp.sum(jnp.real(params * params.conj()))
    total_loss = main_loss + reg_loss
    return jnp.real(total_loss)

# analytical gradient computation (single point)
@partial(jit, static_argnums=(2,3))
def analytic_gradient_single_point_jax(X_point, A_array, H_dim, E_dim):
    """Compute the analytical gradient of the single point loss w.r.t. A_array."""
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
