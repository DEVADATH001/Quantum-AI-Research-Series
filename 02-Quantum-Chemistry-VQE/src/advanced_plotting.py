"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Advanced plotting (PCA trajectories, Error Surfaces)."""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Dict, Any

_FIGURE_DIR = "results/figures"

def _fig_path(filename: str, output_dir: str = _FIGURE_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)

def plot_parameter_trajectory_pca(
    history: List[Dict[str, Any]],
    molecule_name: str,
    ansatz_name: str,
    output_dir: str = _FIGURE_DIR
) -> str:
    """
    Applies PCA to the parameter history to visualize optimization trajectory
    in 2D space.
    """
    if not history or "parameters" not in history[0]:
        return ""
        
    # Extract parameters
    params = np.array([h["parameters"] for h in history])
    energies = np.array([h["energy"] for h in history])
    
    if params.shape[1] < 2:
        return "" # Not enough dimensions to PCA
        
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(params)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        pca_result[:, 0], 
        pca_result[:, 1], 
        c=energies, 
        cmap="viridis", 
        s=50, 
        alpha=0.8,
        edgecolor='k'
    )
    
    # Draw path
    ax.plot(pca_result[:, 0], pca_result[:, 1], 'k-', alpha=0.3)
    
    # Mark start and end
    ax.scatter(pca_result[0, 0], pca_result[0, 1], marker='*', color='red', s=200, label='Start', zorder=5)
    ax.scatter(pca_result[-1, 0], pca_result[-1, 1], marker='X', color='lime', s=200, label='End (Optimum)', zorder=5)
    
    plt.colorbar(sc, label="Energy (Hartree)")
    ax.set_title(f"Parameter Trajectory (PCA): {molecule_name} {ansatz_name}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    path = _fig_path(f"pca_trajectory_{molecule_name}_{ansatz_name}.png", output_dir)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path
