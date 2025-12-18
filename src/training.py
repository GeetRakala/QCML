import jax
import jax.numpy as jnp
from jax import jit
import jaxopt
import optax
from src.models import matrix_from_upper, matrix_from_pauli_params
from src.loss import total_loss_fn, loss_from_params, analytic_gradient_single_point_jax

# Helper to filter params for loss function to avoid passing None when not needed
# Actually, partial in loss_from_params handles it if we structure it right.
# Let's import loss_from_params from src.loss

def train_A_array_analytic_jax(X_array, A_array_init, H_dim, E_dim, num_epochs, lr, l2_lambda):
    """Train the parameters using the full analytical gradient."""
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
    """Train the parameters using the LBFGS solver."""
    print(f"Starting training with LBFGS for {num_epochs} epochs (l2={l2_lambda})...")
    # A_array = A_array_init
    # N_points = X_array.shape[0]

    A_flat_init = jnp.reshape(A_array_init, (-1,))

    def objective(params_flat):
        A_array = jnp.reshape(params_flat, (E_dim, H_dim, H_dim))
        loss_val = total_loss_fn(A_array, X_array)
        reg_loss_val = l2_lambda * jnp.sum(jnp.real(A_array * A_array.conj()))
        return loss_val + reg_loss_val

    # initialize BFGS solver
    solver = jaxopt.LBFGS(fun=objective, maxiter=num_epochs, verbose=2)
    sol = solver.run(init_params=A_flat_init)

    A_array_final = jnp.reshape(sol.params, (E_dim, H_dim, H_dim))

    return A_array_final  # shape: (E_dim, H_dim, H_dim)

def train_A_array_jaxopt(X_array, params_array_init, H_dim, num_epochs, lr, l2_lambda, parametrization, pauli_basis=None):
    """Train the parameters using the jaxopt solver (LBFGS)."""
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
    """Train the parameters using the optax optimizer."""
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

from src.models import (
    compute_ground_state_jax, point_mapping_jax
)

def train_A_array_pseudo_jax(X_array, A_array_init, H_dim, num_epochs, lr, l2_lambda):
    """Train the parameters using a pseudo gradient approach."""
    A_array = A_array_init
    # N_points = X_array.shape[0]
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
