#author  : Geet Rakala
#license : MIT

import jax
import jax.numpy as jnp

def gen_sphere_data(key, N_points, E_dim=3, I_dim=2, noise=0.0):
    # Generate points in R^(I_dim+1)
    key, subkey = jax.random.split(key)
    V = jax.random.normal(subkey, (N_points, I_dim + 1))  # shape: (N_points, I_dim+1)
    norms = jnp.linalg.norm(V, axis=1, keepdims=True)  # shape: (N_points, 1)
    sphere_points = V / norms  # shape: (N_points, I_dim+1)
    # Pad with zeros to reach E_dim
    pad_width = E_dim - (I_dim + 1)
    zeros_pad = jnp.zeros((N_points, pad_width))  # shape: (N_points, pad_width)
    data = jnp.concatenate([sphere_points, zeros_pad], axis=1)  # shape: (N_points, E_dim)
    # Add noise if specified
    if noise:
        key, subkey = jax.random.split(key)
        data = data + noise * jax.random.normal(subkey, data.shape)  # shape: (N_points, E_dim)
    return data


def gen_sphere_other_data(key, N_points, E_dim=9, I_dim=6, noise=0.0):
    # Generate points in R^(I_dim+1)
    key, subkey = jax.random.split(key)
    V = jax.random.normal(subkey, (N_points, I_dim + 1))  # shape: (N_points, I_dim+1)
    norms = jnp.linalg.norm(V, axis=1, keepdims=True)  # shape: (N_points, 1)
    sphere_points = V / norms  # shape: (N_points, I_dim+1)
    # Pad with zeros to reach E_dim
    pad_width = E_dim - (I_dim + 1)
    zeros_pad = jnp.zeros((N_points, pad_width))  # shape: (N_points, pad_width)
    data = jnp.concatenate([sphere_points, zeros_pad], axis=1)  # shape: (N_points, E_dim)
    # Add noise if specified
    if noise:
        key, subkey = jax.random.split(key)
        data = data + noise * jax.random.normal(subkey, data.shape)  # shape: (N_points, E_dim)
    return data

def gen_cubic_data(key, N_points, E_dim=18, I_dim=17, noise=0.0):
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
    zeros_pad = jnp.zeros((N_points, pad_width))  # shape: (N_points, pad_width)
    data = jnp.concatenate([cubic_data, zeros_pad], axis=1)  # shape: (N_points, E_dim)
    if noise:
        key_local, subkey = jax.random.split(key_local)
        data = data + noise * jax.random.normal(subkey, data.shape)  # shape: (N_points, E_dim)
    return data

def gen_cubic_other_data(key, N_points, E_dim=10, I_dim=5, noise=0.0):
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
    zeros_pad = jnp.zeros((N_points, pad_width))  # shape: (N_points, pad_width)
    data = jnp.concatenate([cubic_data, zeros_pad], axis=1)  # shape: (N_points, E_dim)
    if noise:
        key_local, subkey = jax.random.split(key_local)
        data = data + noise * jax.random.normal(subkey, data.shape)  # shape: (N_points, E_dim)
    return data

def gen_campadelli_beta_data(key, N_points, E_dim=40, I_dim=10, noise=0.0, alpha=10.0, beta=0.5):
    # E_dim must equal 4 * I_dim.
    assert E_dim == I_dim * 4, "E_dim must equal 4 * I_dim"
    key, subkey = jax.random.split(key)
    X = jax.random.beta(subkey, a=alpha, b=beta, shape=(N_points, I_dim))  # shape: (N_points, I_dim)
    temp1 = X * jnp.sin(jnp.cos(2 * jnp.pi * X))  # shape: (N_points, I_dim)
    temp2 = X * jnp.cos(jnp.sin(2 * jnp.pi * X))  # shape: (N_points, I_dim)
    # Concatenate to build final data: shape: (N_points, 4*I_dim) == (N_points, E_dim)
    data = jnp.concatenate([temp1, temp2, temp1, temp2], axis=1)  # shape: (N_points, E_dim)
    if noise:
        key, subkey = jax.random.split(key)
        data = data + noise * jax.random.normal(subkey, data.shape)  # shape: (N_points, E_dim)
    return data

def gen_campadelli_n_data(key, N_points, E_dim=72, I_dim=18, noise=0.0):
    # E_dim must equal 4 * I_dim.
    assert E_dim == I_dim * 4, "E_dim must equal 4 * I_dim"
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
    if noise:
        key, subkey = jax.random.split(key)
        data = data + noise * jax.random.normal(subkey, data.shape)  # shape: (N_points, E_dim)
    return data
