"""
Visualization Module

Creates comprehensive visualizations for QAOA research:

Required Plots:
1. Energy Landscape Heatmap - cost(γ, β) surface
2. Graph Cut Visualization - show partition and cut edges
3. Approximation Ratio vs p - performance scaling
4. Optimization Convergence - iteration vs cost

Author: Quantum AI Research Team
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Comprehensive visualization for QAOA Max-Cut research.
    """
    
    def __init__(
        self,
        style: str = "seaborn-v0_8-darkgrid",
        dpi: int = 150,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
            dpi: Figure DPI
            figsize: Default figure size
        """
        self.dpi = dpi
        self.figsize = figsize
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        logger.info(f"Visualizer initialized: dpi={dpi}, style={style}")
    
    def plot_energy_landscape(
        self,
        gamma_grid: np.ndarray,
        beta_grid: np.ndarray,
        cost_grid: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "QAOA Energy Landscape"
    ) -> plt.Figure:
        """
        Plot energy landscape as heatmap.
        
        Shows cost function over parameter space γ ∈ [0, π], β ∈ [0, π]
        
        Args:
            gamma_grid: Gamma values grid
            beta_grid: Beta values grid
            cost_grid: Cost values grid
            save_path: Optional save path
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create heatmap
        im = ax.contourf(
            gamma_grid, beta_grid, cost_grid,
            levels=50,
            cmap='viridis'
        )
        
        # Add contour lines
        ax.contour(
            gamma_grid, beta_grid, cost_grid,
            levels=20,
            colors='white',
            alpha=0.3
        )
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Cost Function Value')
        
        # Labels
        ax.set_xlabel(r'$\gamma$ (gamma)', fontsize=12)
        ax.set_ylabel(r'$\beta$ (beta)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add optimal point marker
        min_idx = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
        ax.plot(
            gamma_grid[min_idx], beta_grid[min_idx],
            'r*', markersize=15, label='Optimal'
        )
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved energy landscape to {save_path}")
        
        return fig
    
    def plot_graph_cut(
        self,
        graph: nx.Graph,
        partition: List[int],
        cut_edges: List[Tuple[int, int]],
        save_path: Optional[str] = None,
        title: str = "Max-Cut Solution"
    ) -> plt.Figure:
        """
        Visualize graph partition for Max-Cut.
        
        Args:
            graph: NetworkX graph
            partition: List of nodes in partition A
            cut_edges: Edges crossing the cut
            save_path: Optional save path
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create position layout
        pos = nx.spring_layout(graph, seed=42)
        
        # Separate nodes by partition
        partition_set = set(partition)
        partition_a = partition_set
        partition_b = set(graph.nodes()) - partition_set
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=list(partition_a),
            node_color='#FF6B6B',
            node_size=300,
            alpha=0.9,
            ax=ax,
            label='Partition A'
        )
        
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=list(partition_b),
            node_color='#4ECDC4',
            node_size=300,
            alpha=0.9,
            ax=ax,
            label='Partition B'
        )
        
        # Draw edges - non-cut
        non_cut_edges = [
            (u, v) for u, v in graph.edges()
            if (u, v) not in cut_edges and (v, u) not in cut_edges
        ]
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=non_cut_edges,
            edge_color='gray',
            width=1,
            alpha=0.3,
            ax=ax
        )
        
        # Draw cut edges (highlighted)
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=cut_edges,
            edge_color='red',
            width=2.5,
            style='dashed',
            alpha=0.9,
            ax=ax
        )
        
        # Labels
        nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold', ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved graph cut visualization to {save_path}")
        
        return fig
    
    def plot_approximation_ratio(
        self,
        depths: List[int],
        ratios: List[float],
        optimal_value: Optional[float] = None,
        save_path: Optional[str] = None,
        title: str = "Approximation Ratio vs QAOA Depth"
    ) -> plt.Figure:
        """
        Plot approximation ratio as function of QAOA depth.
        
        Args:
            depths: List of p values
            ratios: List of approximation ratios
            optimal_value: Optimal value (for reference)
            save_path: Optional save path
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot ratio line
        ax.plot(depths, ratios, 'o-', color='#2196F3', linewidth=2, 
                markersize=10, label='QAOA Ratio')
        
        # Fill area under curve
        ax.fill_between(depths, ratios, alpha=0.3, color='#2196F3')
        
        # Reference lines
        ax.axhline(y=0.8, color='green', linestyle='--', 
                   linewidth=1.5, label='Target (r=0.8)')
        ax.axhline(y=1.0, color='red', linestyle='--', 
                   linewidth=1.5, label='Optimal (r=1.0)')
        
        # Labels
        ax.set_xlabel('QAOA Depth (p)', fontsize=12)
        ax.set_ylabel('Approximation Ratio (r)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set axis limits
        ax.set_xlim(min(depths) - 0.5, max(depths) + 0.5)
        ax.set_ylim(0, 1.1)
        
        # Grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved approximation ratio plot to {save_path}")
        
        return fig
    
    def plot_optimization_convergence(
        self,
        iterations: np.ndarray,
        values: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Optimization Convergence"
    ) -> plt.Figure:
        """
        Plot optimization convergence curve.
        
        Args:
            iterations: Iteration numbers
            values: Cost values
            save_path: Optional save path
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot convergence
        ax.plot(iterations, values, '-', color='#FF9800', linewidth=2)
        ax.scatter(iterations, values, color='#FF9800', s=30, alpha=0.7)
        
        # Labels
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Cost Function Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Mark optimal
        optimal_idx = np.argmin(values)
        ax.axhline(y=values[optimal_idx], color='green', linestyle='--',
                   alpha=0.7, label=f'Best: {values[optimal_idx]:.4f}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved convergence plot to {save_path}")
        
        return fig
    
    def plot_solution_distribution(
        self,
        solutions: Dict[str, int],
        save_path: Optional[str] = None,
        title: str = "Solution Distribution"
    ) -> plt.Figure:
        """
        Plot histogram of solution frequencies.
        
        Args:
            solutions: Dict mapping bitstring to count
            save_path: Optional save path
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Sort by count
        sorted_solutions = sorted(solutions.items(), key=lambda x: x[1], reverse=True)
        top_n = 20
        
        labels = [s[0][:20] + '...' if len(s[0]) > 20 else s[0] 
                  for s in sorted_solutions[:top_n]]
        counts = [s[1] for s in sorted_solutions[:top_n]]
        
        # Bar plot
        bars = ax.bar(range(len(labels)), counts, color='#9C27B0', alpha=0.7)
        
        # Labels
        ax.set_xlabel('Bitstring', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # X-tick labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved solution distribution to {save_path}")
        
        return fig
    
    def plot_comparison_bar(
        self,
        labels: List[str],
        values: List[float],
        reference: Optional[float] = None,
        save_path: Optional[str] = None,
        title: str = "Method Comparison"
    ) -> plt.Figure:
        """
        Plot bar chart comparing multiple methods.
        
        Args:
            labels: Method labels
            values: Cut values
            reference: Reference (optimal) value
            save_path: Optional save path
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        x = range(len(labels))
        
        # Bar colors
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
        
        bars = ax.bar(x, values, color=colors[:len(values)], alpha=0.8)
        
        # Reference line
        if reference:
            ax.axhline(y=reference, color='red', linestyle='--', 
                      linewidth=2, label=f'Optimal: {reference}')
        
        # Labels
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Cut Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        
        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val}', ha='center', va='bottom', fontsize=10)
        
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
        return fig
    
    def plot_correlation_heatmap(
        self,
        correlation_matrix: np.ndarray,
        node_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        title: str = "Variable Correlations"
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            correlation_matrix: NxN correlation matrix
            node_labels: Optional labels for nodes
            save_path: Optional save path
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            annot=False,
            square=True,
            ax=ax,
            cbar_kws={'label': 'Correlation'}
        )
        
        # Labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if node_labels:
            ax.set_xticklabels(node_labels, rotation=45, ha='right')
            ax.set_yticklabels(node_labels, rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved correlation heatmap to {save_path}")
        
        return fig
    
    def create_dashboard(
        self,
        graph: nx.Graph,
        partition: List[int],
        energy_data: Optional[Dict] = None,
        convergence_data: Optional[Dict] = None,
        output_dir: str = "."
    ) -> None:
        """
        Create complete visualization dashboard.
        
        Args:
            graph: Problem graph
            partition: Solution partition
            energy_data: Optional energy landscape data
            convergence_data: Optional convergence data
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Cut visualization
        cut_edges = [
            (u, v) for u, v in graph.edges()
            if (u in partition) != (v in partition)
        ]
        
        self.plot_graph_cut(
            graph=graph,
            partition=partition,
            cut_edges=cut_edges,
            save_path=str(output_path / "graph_cut_visualization.png")
        )
        
        # Energy landscape if provided
        if energy_data:
            self.plot_energy_landscape(
                gamma_grid=energy_data['gamma_grid'],
                beta_grid=energy_data['beta_grid'],
                cost_grid=energy_data['cost_grid'],
                save_path=str(output_path / "energy_landscape.png")
            )
        
        # Convergence if provided
        if convergence_data:
            self.plot_optimization_convergence(
                iterations=convergence_data['iterations'],
                values=convergence_data['values'],
                save_path=str(output_path / "optimization_convergence.png")
            )
        
        logger.info(f"Dashboard saved to {output_dir}")


def save_metrics_csv(
    metrics: List[Dict],
    save_path: str
) -> None:
    """
    Save metrics to CSV file.
    
    Args:
        metrics: List of metric dictionaries
        save_path: Output file path
    """
    import pandas as pd
    
    df = pd.DataFrame(metrics)
    df.to_csv(save_path, index=False)
    
    logger.info(f"Saved metrics to {save_path}")

