import matplotlib.pyplot as plt
import numpy as np
import os
import textwrap

# Set default style parameters but keep high DPI
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.size': 10,  # Reset to standard
    'axes.grid': True, 
    'grid.alpha': 0.7,
})

# Ensure we are using default style
plt.style.use('default')

def plot_point_cloud(Y_array_np, X_array_np, E_eigval_0_array_np, noise_level, solver, dataset_type, save=False, filename=None):
    """
    Plot the point cloud obtained from the embedding together with the original data.
    """
    E_dim = Y_array_np.shape[1]
    num_pairs = E_dim // 2

    min_val = np.min(E_eigval_0_array_np)
    max_val = np.max(E_eigval_0_array_np)
    if max_val - min_val < 1e-8:
        E_eigval_0_norm = np.zeros_like(E_eigval_0_array_np)
    else:
        E_eigval_0_norm = (E_eigval_0_array_np - min_val) / (max_val - min_val)

    ncols = int(np.ceil(np.sqrt(num_pairs)))
    nrows = int(np.ceil(num_pairs / ncols))

    # Keep constrained_layout for good spacing
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5), constrained_layout=True)
    if num_pairs == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i in range(num_pairs):
        dim1 = 2 * i
        dim2 = 2 * i + 1
        ax = axes[i]
        
        # Plot embedded points - Classic look
        scatter = ax.scatter(Y_array_np[:, dim1], Y_array_np[:, dim2], c=1 - E_eigval_0_norm,
                   cmap='viridis', label='Point Cloud', s=25)

        # Plot original data - Classic look
        if X_array_np is not None:
             if X_array_np.shape[1] > dim2:
                ax.scatter(X_array_np[:, dim1], X_array_np[:, dim2], color='red',
                           alpha=0.4, label='Original Data', s=5)

        ax.set_xlabel(f"$y({dim1+1})$")
        ax.set_ylabel(f"$y({dim2+1})$")
        ax.set_title(f"Dimensions {dim1 + 1} vs {dim2 + 1}")
        ax.grid(True)
        
        if i == 0:
            ax.legend(loc='upper right', frameon=True)

        ax.set_aspect('equal', adjustable='box')

    for i in range(num_pairs, len(axes)):
        fig.delaxes(axes[i])

    title_str = f"Point Cloud (Noise={noise_level}) \n [Dataset: {dataset_type}, Solver: {solver}]"
    plt.suptitle(title_str, fontsize=14)

    if save and filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_mean_eigenvalues(G_eigvals_array_np, noise_level, solver, dataset_type, save=False, filename=None):
    """Plot the mean eigenvalues of the quantum metric over all data points with error bars."""
    mean_eigs = np.mean(G_eigvals_array_np, axis=0)
    std_eigs = np.std(G_eigvals_array_np, axis=0)
    indices = np.arange(len(mean_eigs))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    ax1.errorbar(indices, mean_eigs, yerr=std_eigs, fmt='o', capsize=5,
                 label='Linear Scale', markersize=5)
    ax1.set_xlabel("Eigenvalue Index")
    ax1.set_ylabel("Mean Eigenvalue (Linear)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    mask = mean_eigs > 1e-12
    # Always plot log scale even if empty (just to match style)
    ax2.errorbar(indices, mean_eigs, yerr=std_eigs, fmt='s', capsize=5, alpha=0.3,
                 color='black', label='Log Scale', markersize=5)
    ax2.set_yscale('log')
    ax2.set_ylabel("Mean Eigenvalue (Log)")

    ax1.set_title(f"Mean Quantum Metric Eigenvalues (Noise = {noise_level}) \n [Dataset: {dataset_type},Solver: {solver}]")
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='best')

    plt.tight_layout()

    if save and filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_quantum_metric_spectra(G_eigvals_array, noise_level, save=False, filename=None):
    """Plot the eigenvalue spectra of the quantum metric for all data points."""
    G_eigvals_array_np = np.array(G_eigvals_array)
    num_points = G_eigvals_array_np.shape[0]
    E_dim = G_eigvals_array_np.shape[1]
    
    plt.figure(figsize=(8, 5))
    markers = ['o', 's', '^', 'v', 'D', 'x', '*', 'p', 'h', '<', '>']
    
    for i in range(E_dim):
        marker = markers[i % len(markers)]
        plt.plot(range(num_points), G_eigvals_array_np[:, i], 
                 marker=marker, markersize=4, linestyle='-', label=f"Eigenvalue {i}")

    plt.title(f"Quantum Metric Spectra (Noise = {noise_level})")
    plt.xlabel("Data Point Index")
    plt.ylabel("Eigenvalues of $g(x)$")
    plt.legend()
    plt.grid(True)

    if save and filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_I_dim_array(I_dim_array, noise_level, save=False, filename=None):
    """Plot the intrinsic dimension estimates as a function of the data point index."""
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
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_I_dim_array_hist(I_dim_array, noise_level, save=False, filename=None):
    """Plot a histogram of the intrinsic dimension estimates."""
    d_vals = np.array(I_dim_array)
    if len(d_vals) == 0:
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
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
