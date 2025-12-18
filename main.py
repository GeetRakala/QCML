import yaml
import jax
import jax.numpy as jnp
import numpy as np
import os
import argparse
import hashlib
import json
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

def run_experiment(run_config):
    """Run a single experiment configuration."""
    # Extract parameters
    solver = run_config['solver']
    dataset_type = run_config['dataset_type']
    parametrization = run_config['parametrization']
    N_points = run_config['N_points']
    H_dim = run_config['H_dim']
    num_epochs = run_config['epochs']
    A_init_scale = run_config['A_init_scale']
    lr = run_config['learning_rate']
    l2_lambda = float(run_config['l2_lambda'])
    noise_level = run_config['noise_level']
    seed = run_config['seed']

    grad_clip_norm = run_config.get('grad_clip_norm', 1.0)
    transition_steps = run_config.get('transition_steps', 25)
    decay_rate = run_config.get('decay_rate', 0.99)

    dataset_dims = {
        "sphere": 3,
        "sphere_other": 9,
        "cubic": 18,
        "cubic_other": 10,
        "campadelli_beta": 40,
        "campadelli_n": 72
    }
    E_dim = dataset_dims.get(dataset_type, 3) # Default to 3 if unknown

    # Hierarchical structure:
    # plots/dataset/solver/H{}_lr{}_l2{}_{hash}/noise_{}
    
    # Compute a stable hash of the configuration to ensure uniqueness
    # We filter out 'output_dir' and 'seed' from the hash if we want seed repeats to be same folder? 
    # The user probably wants different seeds to be different runs? usually yes.
    # Let's include everything relevant to the physics/training.
    config_str = json.dumps(run_config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:6]

    param_str = f"H{H_dim}_lr{lr}_l2{l2_lambda:.1e}_{config_hash}"
    save_folder = os.path.join(
        run_config.get('output_dir', 'plots'), 
        dataset_type, 
        solver, 
        param_str, 
        f"noise_{noise_level:.2f}"
    )
    os.makedirs(save_folder, exist_ok=True)
    
    # Save the specific config for this run
    with open(os.path.join(save_folder, "config.yaml"), "w") as f:
        yaml.dump(run_config, f)

    print("-" * 50)
    print(f"Running Experiment:")
    print(f"  Dataset: {dataset_type}, Solver: {solver}")
    print(f"  Params: H={H_dim}, lr={lr}, l2={l2_lambda}")
    print(f"  Noise: {noise_level}")

    key = jax.random.PRNGKey(seed)
    key, subkey_data, subkey_init = jax.random.split(key, 3)

    # Generate data
    gen_data = get_data_generator(dataset_type)
    X_array = gen_data(subkey_data, N_points, E_dim=E_dim, noise=noise_level)

    # Initialize A_array
    pauli_basis_tensor = None
    params_array_init = None
    A_array_init_full = None

    if parametrization == "upper":
        A_array_init = initialize_A_array_jax(subkey_init, E_dim, H_dim, scale=A_init_scale)
        params_array_init = upper_from_matrix(A_array_init, H_dim)
        A_array_init_full = A_array_init 
    elif parametrization == "pauli":
        params_array_init = initialize_A_array_pauli_jax(subkey_init, E_dim, H_dim, scale=A_init_scale)
        pauli_basis_tensor = generate_pauli_basis(H_dim)
        if solver in ["analytic", "LBFGS"]:
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

    print("Estimating intrinsic dimension...")
    I_dim_array, Y_array, E_eigval_0_array, G_eigvals_array = I_dimension_estimator_jax(X_array, A_array_opt)
    
    # Convert to numpy for plotting
    I_dim_array_np = np.array(I_dim_array)
    Y_array_np = np.array(Y_array)
    X_array_np = np.array(X_array)
    E_eigval_0_array_np = np.array(E_eigval_0_array)
    G_eigvals_array_np = np.array(G_eigvals_array)

    print("Generating plots...")
    plot_point_cloud(Y_array_np, X_array_np, E_eigval_0_array_np, noise_level, solver, dataset_type, save=True,
                     filename=os.path.join(save_folder, f"point_cloud.png"))
    plot_mean_eigenvalues(G_eigvals_array_np, noise_level, solver, dataset_type, save=True,
                          filename=os.path.join(save_folder, f"mean_eigenvalues.png"))
    # Save I_dim histogram too as it's useful
    plot_I_dim_array_hist(I_dim_array_np, noise_level, save=True,
                          filename=os.path.join(save_folder, f"I_dim_hist.png"))
    
    print(f"Results saved to {save_folder}")
    print("-" * 50)

import itertools

def main():
    parser = argparse.ArgumentParser(description="QCML Reproduction Script")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)["experiment"]

    # Handle parameters that should be grid-searched
    # We look for lists in the config and create a grid
    grid_params = {}
    fixed_params = {}
    
    # Helper: ensure everything is a list for iteration if it's meant to be swept
    # If the user put a list in the yaml, we assume they want to sweep it.
    # Exception: noise_levels is explicitly plural in the old config, let's map it to noise_level list
    
    if "noise_levels" in base_config:
        base_config["noise_level"] = base_config.pop("noise_levels")

    for k, v in base_config.items():
        if isinstance(v, list):
            grid_params[k] = v
        else:
            fixed_params[k] = v

    # Create all combinations
    keys = list(grid_params.keys())
    values = list(grid_params.values())
    combinations = list(itertools.product(*values))

    print(f"Found {len(keys)} sweep parameters: {keys}")
    print(f"Total experiments to run: {len(combinations)}")

    for combo in combinations:
        # Merge fixed run params with current combo
        run_config = fixed_params.copy()
        run_config.update(dict(zip(keys, combo)))
        
        # Run
        try:
            run_experiment(run_config)
        except Exception as e:
            print(f"ERROR running configuration: {dict(zip(keys, combo))}")
            print(e)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
