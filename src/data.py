import jax
import jax.numpy as jnp
from functools import partial

# --- Data Generation Functions ---
# Copied from datasets_jax.py and qcml.py logic

def gen_sphere_data(key, N_points, E_dim=3, I_dim=2, noise=0.0):
    """Generate points on a unit sphere in R^(I_dim+1), embedded in R^E_dim."""
    # Generate points in R^(I_dim+1)
    key, subkey = jax.random.split(key)
    V = jax.random.normal(subkey, (N_points, I_dim + 1))  # shape: (N_points, I_dim+1)
    norms = jnp.linalg.norm(V, axis=1, keepdims=True)  # shape: (N_points, 1)
    sphere_points = V / norms  # shape: (N_points, I_dim+1)
    # Pad with zeros to reach E_dim
    pad_width = E_dim - (I_dim + 1)
    if pad_width > 0:
        zeros_pad = jnp.zeros((N_points, pad_width))  # shape: (N_points, pad_width)
        data = jnp.concatenate([sphere_points, zeros_pad], axis=1)  # shape: (N_points, E_dim)
    else:
        data = sphere_points
    # Add noise if specified
    if noise > 0.0:
        key, subkey = jax.random.split(key)
        data = data + noise * jax.random.normal(subkey, data.shape)  # shape: (N_points, E_dim)
    return data

def gen_sphere_other_data(key, N_points, E_dim=9, I_dim=6, noise=0.0):
    return gen_sphere_data(key, N_points, E_dim, I_dim, noise)

def gen_cubic_data(key, N_points, E_dim=18, I_dim=17, noise=0.0):
    """Generate points on a hypercube surface."""
    # cubic data is generated in R^(I_dim+1) and then padded to E_dim.
    # Determine number of points per side for each coordinate.
    n_once = int(N_points / (2 * (I_dim + 1)) + 1)
    parts = []
    key_local = key
    for i in range(I_dim + 1):
        for side in range(2):
            key_local, subkey = jax.random.split(key_local)
            data_once = jax.random.uniform(subkey, shape=(I_dim + 1, n_once))  # shape: (I_dim+1, n_once)
            # Force the i-th coordinate to a fixed value: 0 for first side, 1 for second side.
            if side == 0:
                data_once = data_once.at[i, :].set(0.0)
            else:
                data_once = data_once.at[i, :].set(1.0)
            parts.append(data_once)
    # Concatenate along columns then transpose so each row is a sample.
    cubic_data = jnp.concatenate(parts, axis=1).T  # shape: (total_samples, I_dim+1)
    cubic_data = cubic_data[:N_points, :]  # shape: (N_points, I_dim+1)
    # Pad with zeros to reach E_dim.
    pad_width = E_dim - (I_dim + 1)
    if pad_width > 0:
        zeros_pad = jnp.zeros((N_points, pad_width))  # shape: (N_points, pad_width)
        data = jnp.concatenate([cubic_data, zeros_pad], axis=1)  # shape: (N_points, E_dim)
    else:
        data = cubic_data
    if noise > 0.0:
        key_local, subkey = jax.random.split(key_local)
        data = data + noise * jax.random.normal(subkey, data.shape)  # shape: (N_points, E_dim)
    return data

def gen_cubic_other_data(key, N_points, E_dim=10, I_dim=5, noise=0.0):
    return gen_cubic_data(key, N_points, E_dim, I_dim, noise)

def gen_campadelli_beta_data(key, N_points, E_dim=40, I_dim=10, noise=0.0, alpha=10.0, beta=0.5):
    # E_dim must equal 4 * I_dim.
    if E_dim != I_dim * 4:
         raise ValueError(f"E_dim ({E_dim}) must equal 4 * I_dim ({I_dim}) for campadelli_beta")
    key, subkey = jax.random.split(key)
    X = jax.random.beta(subkey, a=alpha, b=beta, shape=(N_points, I_dim))  # shape: (N_points, I_dim)
    temp1 = X * jnp.sin(jnp.cos(2 * jnp.pi * X))  # shape: (N_points, I_dim)
    temp2 = X * jnp.cos(jnp.sin(2 * jnp.pi * X))  # shape: (N_points, I_dim)
    # Concatenate to build final data: shape: (N_points, 4*I_dim) == (N_points, E_dim)
    data = jnp.concatenate([temp1, temp2, temp1, temp2], axis=1)  # shape: (N_points, E_dim)
    if noise > 0.0:
        key, subkey = jax.random.split(key)
        data = data + noise * jax.random.normal(subkey, data.shape)  # shape: (N_points, E_dim)
    return data

def gen_campadelli_n_data(key, N_points, E_dim=72, I_dim=18, noise=0.0):
    # E_dim must equal 4 * I_dim.
    if E_dim != I_dim * 4:
         raise ValueError(f"E_dim ({E_dim}) must equal 4 * I_dim ({I_dim}) for campadelli_n")
    key, subkey = jax.random.split(key)
    X = jax.random.uniform(subkey, shape=(N_points, I_dim))  # shape: (N_points, I_dim)
    temp1_list, temp2_list = [], []
    for i in range(I_dim):
        # Compute column vectors: shapes (N_points,)
        temp1_col = jnp.tan(X[:, i] * jnp.cos(X[:, I_dim - 1 - i]))
        temp2_col = jnp.arctan(X[:, I_dim - 1 - i] * jnp.sin(X[:, i]))
        temp1_list.append(temp1_col[:, None])  # shape: (N_points, 1)
        temp2_list.append(temp2_col[:, None])  # shape: (N_points, 1)
    temp1 = jnp.concatenate(temp1_list, axis=1)  # shape: (N_points, I_dim)
    temp2 = jnp.concatenate(temp2_list, axis=1)  # shape: (N_points, I_dim)
    # Concatenate four copies to form final data: shape: (N_points, 4*I_dim) == (N_points, E_dim)
    data = jnp.concatenate([temp1, temp2, temp1, temp2], axis=1)  # shape: (N_points, E_dim)
    if noise > 0.0:
        key, subkey = jax.random.split(key)
        data = data + noise * jax.random.normal(subkey, data.shape)  # shape: (N_points, E_dim)
    return data

def get_data_generator(dataset_type):
    """Factory function to get the data generation function."""
    generators = {
        "sphere": gen_sphere_data,
        "sphere_other": gen_sphere_other_data,
        "cubic": gen_cubic_data,
        "cubic_other": gen_cubic_other_data,
        "campadelli_beta": gen_campadelli_beta_data,
        "campadelli_n": gen_campadelli_n_data
    }
    if dataset_type not in generators:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    return generators[dataset_type]
