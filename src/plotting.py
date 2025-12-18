import matplotlib.pyplot as plt
import numpy as np
import os
import textwrap

# Set professional style parameters
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'grid.alpha': 0.5,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})
# Try to use a clean style if available
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    try:
        plt.style.use('ggplot')
    except OSError:
        pass

def plot_point_cloud(Y_array_np, X_array_np, E_eigval_0_array_np, noise_level, solver, dataset_type, save=False, filename=None):
    """
    Plot the point cloud obtained from the embedding together with the original data.
    """
    E_dim = Y_array_np.shape[1]
    num_pairs = E_dim // 2

    # Avoid division by zero in normalization
    min_val = np.min(E_eigval_0_array_np)
    max_val = np.max(E_eigval_0_array_np)
    if max_val - min_val < 1e-8:
        E_eigval_0_norm = np.zeros_like(E_eigval_0_array_np)
    else:
        E_eigval_0_norm = (E_eigval_0_array_np - min_val) / (max_val - min_val)

    # Calculate layout: rectangular grid close to square total shape
    ncols = int(np.ceil(np.sqrt(num_pairs)))
    nrows = int(np.ceil(num_pairs / ncols))

    # Figure size: Scale with number of plots. 
    # Use constrained_layout=True for automatic spacing adjustment
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.5), constrained_layout=True)
    if num_pairs == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i in range(num_pairs):
        dim1 = 2 * i
        dim2 = 2 * i + 1
        ax = axes[i]
        
        # Plot embedded points
        scatter = ax.scatter(Y_array_np[:, dim1], Y_array_np[:, dim2], c=1 - E_eigval_0_norm,
                   cmap='magma', label='Embedded', s=20, alpha=0.9, edgecolors='none')

        # Plot original data (projected) if available, lighter
        if X_array_np is not None:
             # Ensure dims exist
             if X_array_np.shape[1] > dim2:
                ax.scatter(X_array_np[:, dim1], X_array_np[:, dim2], color='cyan',
                           alpha=0.15, label='Original', s=10)

        ax.set_xlabel(f"$y_{{{dim1+1}}}$")
        ax.set_ylabel(f"$y_{{{dim2+1}}}$")
        ax.set_title(f"Dims {dim1 + 1} vs {dim2 + 1}")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add legend only to the first plot to avoid clutter
        if i == 0:
            ax.legend(loc='upper right', frameon=True)

        # Ensure perfect square aspect ratio
        ax.set_aspect('equal', adjustable='box')

    # Remove empty axes
    for i in range(num_pairs, len(axes)):
        fig.delaxes(axes[i])

    # Wrap long titles to avoid cutoff
    title_str = f"Point Cloud (Noise={noise_level} | {dataset_type} | {solver})"
    wrapped_title = "\n".join(textwrap.wrap(title_str, width=60))
    plt.suptitle(wrapped_title, fontweight='bold')

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
    indices = np.arange(1, len(mean_eigs) + 1) # 1-based indexing for eigenvalues

    fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    
    # Linear scale
    ax1.errorbar(indices, mean_eigs, yerr=std_eigs, fmt='-o', capsize=4,
                 label='Linear Scale', color='tab:blue', linewidth=1.5, markersize=5)
    ax1.set_xlabel("Eigenvalue Index")
    ax1.set_ylabel("Mean Eigenvalue (Linear)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='major', linestyle='-', alpha=0.6)
    ax1.set_xticks(indices)

    # Log scale
    ax2 = ax1.twinx()
    # Filter out zeros for log plot
    mask = mean_eigs > 1e-12
    if np.any(mask):
        ax2.errorbar(indices[mask], mean_eigs[mask], yerr=std_eigs[mask], fmt='--s', capsize=4,
                     color='tab:gray', label='Log Scale', alpha=0.6)
        ax2.set_yscale('log')
    
    ax2.set_ylabel("Mean Eigenvalue (Log)", color='tab:gray')
    ax2.tick_params(axis='y', labelcolor='tab:gray')

    ax1.set_title(f"Metric Eigenvalues\n(Noise={noise_level} | {dataset_type})", fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

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
    
    plt.figure(figsize=(10, 6), constrained_layout=True)
    
    # Use distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, E_dim))
    
    for i in range(E_dim):
        plt.plot(range(num_points), G_eigvals_array_np[:, i], 
                 marker=None, linewidth=1.0, color=colors[i], label=f"$\\lambda_{{{i+1}}}$")

    plt.title(f"Metric Spectra (Noise={noise_level})", fontweight='bold')
    plt.xlabel("Data Point Index")
    plt.ylabel("Eigenvalues")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, alpha=0.5)

    if save and filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Use raw string for latex to avoid warning
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_I_dim_array(I_dim_array, noise_level, save=False, filename=None):
    """Plot the intrinsic dimension estimates as a function of the data point index."""
    I_dim_array_np = np.array(I_dim_array)
    plt.figure(figsize=(8, 4), constrained_layout=True)
    indices = range(len(I_dim_array_np))
    
    plt.plot(indices, I_dim_array_np, marker='o', markersize=2, linestyle='-', linewidth=0.5, 
             color='tab:purple', label='I_dim Estimate')
    
    plt.title(f"I_dim Trace (Noise={noise_level})", fontweight='bold')
    plt.xlabel("Data Point Index")
    plt.ylabel("Est. Dimension")
    
    # Integer ticks for y-axis if range is small
    unique_dims = np.unique(np.round(I_dim_array_np))
    if len(unique_dims) < 20:
         plt.yticks(np.arange(int(unique_dims.min()), int(unique_dims.max()) + 1))
    
    plt.grid(True, alpha=0.5)

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
        
    min_d = np.floor(d_vals.min())
    max_d = np.ceil(d_vals.max())
    
    # Center bins on integers
    bins = np.arange(min_d - 0.5, max_d + 1.5, 1)

    plt.figure(figsize=(8, 5), constrained_layout=True)
    n, bins, patches = plt.hist(d_vals, bins=bins, color='tab:blue', 
                                edgecolor='black', alpha=0.8, rwidth=0.85)

    plt.title(f"I_dim Distribution (Noise={noise_level})", fontweight='bold')
    plt.xlabel("Estimated Intrinsic Dimension")
    plt.ylabel("Count")
    
    # Set x-ticks to be integers
    plt.xticks(np.arange(min_d, max_d + 1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    if save and filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
