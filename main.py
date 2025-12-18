import yaml
import jax
import jax.numpy as jnp
import numpy as np
import os
import argparse
from src.data import get_data_generator
from src.models import (
    initialize_A_array_jax, upper_from_matrix, initialize_A_array_pauli_jax,
    generate_pauli_basis, matrix_from_pauli_params, I_dimension_estimator_jax
)
from src.training import (
    train_A_array_analytic_jax, train_A_array_LBFGS,
    train_A_array_jaxopt, train_A_array_optax, train_A_array_pseudo_jax
)
from src.plotting import (
    plot_point_cloud, plot_mean_eigenvalues, plot_quantum_metric_spectra,
    plot_I_dim_array, plot_I_dim_array_hist
)

def run_experiment(config, noise_level, seed):
    """Run a single experiment configuration."""
    solver = config['solver']
    dataset_type = config['dataset_type']
    parametrization = config['parametrization']
    N_points = config['N_points']
    H_dim = config['H_dim']
    num_epochs = config['epochs']
    A_init_scale = config['A_init_scale']
    lr = config['learning_rate']
    l2_lambda = float(config['l2_lambda']) # Ensure float
    grad_clip_norm = config.get('grad_clip_norm', 1.0)
    transition_steps = config.get('transition_steps', 25)
    decay_rate = config.get('decay_rate', 0.99)

    dataset_dims = {
        "sphere": 3,
        "sphere_other": 9,
        "cubic": 18,
        "cubic_other": 10,
        "campadelli_beta": 40,
        "campadelli_n": 72
    }
    E_dim = dataset_dims[dataset_type]

    
    # Best practice: Hierarchical structure for easy navigation
    # plots/dataset_type/solver/H16/noise_0.00
    save_folder = os.path.join(
        config['output_dir'], 
        dataset_type, 
        solver, 
        f"H{H_dim}", 
        f"noise_{noise_level:.2f}"
    )
    os.makedirs(save_folder, exist_ok=True)
    print("-" * 50)
    print(f"Running Experiment: {dataset_type} (Noise={noise_level}) with {solver}")

    key = jax.random.PRNGKey(seed)
    key, subkey_data, subkey_init = jax.random.split(key, 3)

    # Generate data
    gen_data = get_data_generator(dataset_type)
    # The generator kwargs might need adjustment based on dataset type if defaults are not enough
    # but the defaults seem consistent with qcml.py usage.
    # For now, we assume defaults in get_data_generator handle E_dim if passed or defaults.
    # Actually, E_dim is constant per dataset type in qcml.py main block map.
    # Let's pass E_dim explicitly if the generator accepts it.
    
    # We need to inspect signature or just pass what we know.
    # In src/data.py, gen_sphere_data takes E_dim.
    X_array = gen_data(subkey_data, N_points, E_dim=E_dim, noise=noise_level)

    # Initialize A_array
    pauli_basis_tensor = None
    params_array_init = None
    A_array_init_full = None

    if parametrization == "upper":
        A_array_init = initialize_A_array_jax(subkey_init, E_dim, H_dim, scale=A_init_scale)
        params_array_init = upper_from_matrix(A_array_init, H_dim)
        A_array_init_full = A_array_init # For analytic/LBFGS
    elif parametrization == "pauli":
        params_array_init = initialize_A_array_pauli_jax(subkey_init, E_dim, H_dim, scale=A_init_scale)
        pauli_basis_tensor = generate_pauli_basis(H_dim)
        if solver in ["analytic", "LBFGS"]:
             # These need full matrix
             A_array_init_full = jax.vmap(lambda p: matrix_from_pauli_params(p, H_dim, pauli_basis_tensor))(params_array_init)

    # Train
    A_array_opt = None
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
        A_array_opt = train_A_array_pseudo_jax(X_array, A_array_init_full, H_dim, num_epochs=num_epochs, lr=lr, l2_lambda=l2_lambda)
    elif solver == "analytic":
        A_array_opt = train_A_array_analytic_jax(X_array, A_array_init_full, H_dim, E_dim, num_epochs=num_epochs, lr=lr, l2_lambda=l2_lambda)
    elif solver == "LBFGS":
        A_array_opt = train_A_array_LBFGS(X_array, A_array_init_full, H_dim, E_dim, num_epochs=num_epochs, lr=lr, l2_lambda=l2_lambda)
    else:
        raise ValueError(f"unknown solver: {solver}")

    print("Estimating intrinsic dimension for all points...")
    I_dim_array, Y_array, E_eigval_0_array, G_eigvals_array = I_dimension_estimator_jax(X_array, A_array_opt)
    
    # Convert to numpy for plotting
    I_dim_array_np = np.array(I_dim_array)
    Y_array_np = np.array(Y_array)
    X_array_np = np.array(X_array)
    E_eigval_0_array_np = np.array(E_eigval_0_array)
    G_eigvals_array_np = np.array(G_eigvals_array)

    print("Generating plots...")
    plot_point_cloud(Y_array_np, X_array_np, E_eigval_0_array_np, noise_level, solver, dataset_type, save=True,
                     filename=os.path.join(save_folder, f"point_cloud_noise_{noise_level:.2f}.png"))
    plot_mean_eigenvalues(G_eigvals_array_np, noise_level, solver, dataset_type, save=True,
                          filename=os.path.join(save_folder, f"mean_eigenvalues_noise_{noise_level:.2f}.png"))
    print(f"Plots saved to {save_folder}")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="QCML Reproduction Script")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)["experiment"]

    seed = config.get("seed", 137)
    noise_levels = config.get("noise_levels", [0.0])

    for noise in noise_levels:
        run_experiment(config, noise, seed)

if __name__ == "__main__":
    main()
