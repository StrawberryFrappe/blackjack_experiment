"""Network capacity analysis for hybrid quantum-classical networks.

This module analyzes the representational capacity of each component
to determine if performance is limited by quantum, encoder, or postprocessing.

Key insight: With minimal postprocessing (no hidden layers), declining quantum
contribution ≠ classical compensating. Instead, encoder learns to bypass quantum.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class CapacityAnalyzer:
    """
    Analyzes network capacity and visualizes architectural constraints.
    
    Shows:
    1. Parameter distribution across components
    2. Capacity bottlenecks (e.g., minimal postprocessing)
    3. Information flow paths and where compensation is/isn't possible
    4. Comparison with theoretical optimal architectures
    """
    
    def __init__(self, network: nn.Module):
        """
        Initialize capacity analyzer.
        
        Args:
            network: Hybrid network to analyze
        """
        self.network = network
        self.capacity = self._analyze_capacity()
        self.bottlenecks = self._identify_bottlenecks()
    
    def _analyze_capacity(self) -> Dict:
        """Comprehensive capacity analysis."""
        cap = {
            'encoder': {},
            'quantum': {},
            'postprocessing': {},
            'total': {}
        }
        
        # Encoder analysis
        if hasattr(self.network, 'feature_encoder'):
            encoder_params = []
            layer_dims = []
            
            for module in self.network.feature_encoder.modules():
                if isinstance(module, nn.Linear):
                    in_f, out_f = module.weight.shape[1], module.weight.shape[0]
                    layer_dims.append((in_f, out_f))
                    encoder_params.append(module.weight.numel() + module.bias.numel())
            
            cap['encoder'] = {
                'total_params': sum(encoder_params),
                'num_layers': len(encoder_params),
                'hidden_layers': len(encoder_params) - 1,
                'layer_dims': layer_dims,
                'params_per_layer': encoder_params,
                'has_hidden_capacity': len(encoder_params) > 1
            }
        else:
            cap['encoder'] = {
                'total_params': 0,
                'num_layers': 0,
                'hidden_layers': 0,
                'layer_dims': [],
                'params_per_layer': [],
                'has_hidden_capacity': False
            }
        
        # Quantum analysis
        if hasattr(self.network, 'weights'):
            n_qubits = getattr(self.network, 'n_qubits', self.network.weights.shape[1])
            n_layers = self.network.weights.shape[0]
            
            cap['quantum'] = {
                'total_params': self.network.weights.numel(),
                'n_qubits': n_qubits,
                'n_layers': n_layers,
                'hilbert_space_dim': 2 ** n_qubits,
                'theoretical_capacity': 2 ** n_qubits,  # Simplified
            }
        else:
            cap['quantum'] = {
                'total_params': 0,
                'n_qubits': 0,
                'n_layers': 0,
                'hilbert_space_dim': 0,
                'theoretical_capacity': 0,
            }
        
        # Postprocessing analysis
        if hasattr(self.network, 'postprocessing'):
            postproc_params = []
            layer_dims = []
            
            for module in self.network.postprocessing.modules():
                if isinstance(module, nn.Linear):
                    in_f, out_f = module.weight.shape[1], module.weight.shape[0]
                    layer_dims.append((in_f, out_f))
                    postproc_params.append(module.weight.numel() + module.bias.numel())
            
            cap['postprocessing'] = {
                'total_params': sum(postproc_params),
                'num_layers': len(postproc_params),
                'hidden_layers': len(postproc_params) - 1,
                'layer_dims': layer_dims,
                'params_per_layer': postproc_params,
                'is_minimal': len(postproc_params) == 1,  # Direct mapping only
                'can_compensate': len(postproc_params) > 1
            }
        else:
            cap['postprocessing'] = {
                'total_params': 0,
                'num_layers': 0,
                'hidden_layers': 0,
                'layer_dims': [],
                'params_per_layer': [],
                'is_minimal': False,
                'can_compensate': False
            }
        
        # Total
        total_params = (cap['encoder'].get('total_params', 0) + 
                       cap['quantum'].get('total_params', 0) + 
                       cap['postprocessing'].get('total_params', 0))
        
        cap['total'] = {
            'total_params': total_params,
            'quantum_ratio': cap['quantum'].get('total_params', 0) / total_params if total_params > 0 else 0,
            'encoder_ratio': cap['encoder'].get('total_params', 0) / total_params if total_params > 0 else 0,
            'postproc_ratio': cap['postprocessing'].get('total_params', 0) / total_params if total_params > 0 else 0,
        }
        
        return cap
    
    def _identify_bottlenecks(self) -> Dict:
        """Identify capacity bottlenecks."""
        bottlenecks = {
            'has_bottleneck': False,
            'bottleneck_type': None,
            'explanation': None,
            'implications': []
        }
        
        postproc = self.capacity.get('postprocessing', {})
        
        # Minimal postprocessing is the key bottleneck
        if postproc.get('is_minimal', False):
            bottlenecks['has_bottleneck'] = True
            bottlenecks['bottleneck_type'] = 'MINIMAL_POSTPROCESSING'
            bottlenecks['explanation'] = (
                f"Postprocessing has only {postproc['total_params']} parameters "
                f"with no hidden layers. It can only perform a linear transformation: "
                f"output = W @ quantum_features + b"
            )
            bottlenecks['implications'] = [
                "[NO] Cannot learn complex decision logic from quantum features",
                "[NO] Cannot compensate for weak/noisy quantum output",
                "[YES] Forces quantum circuit to encode decision-relevant features",
                "[YES] Any performance requires quantum to do meaningful work",
                "[WARN] If quantum contribution declines, performance will suffer"
            ]
        elif postproc.get('hidden_layers', 0) == 0:
            bottlenecks['has_bottleneck'] = True
            bottlenecks['bottleneck_type'] = 'NO_HIDDEN_LAYERS'
            bottlenecks['explanation'] = "No hidden layers in postprocessing limits expressiveness"
            bottlenecks['implications'] = [
                "[WARN] Limited capacity to extract complex features from quantum output",
                "[YES] Still forces quantum to provide useful representations"
            ]
        
        return bottlenecks
    
    def visualize_architecture(
        self, 
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (16, 10)
    ):
        """
        Create comprehensive visualization of network capacity and information flow.
        
        Args:
            save_path: Path to save figure
            show: Whether to display
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Parameter distribution pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_param_distribution(ax1)
        
        # 2. Architecture flow diagram
        ax2 = fig.add_subplot(gs[0, 1:])
        self._plot_architecture_flow(ax2)
        
        # 3. Capacity comparison bar chart
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_capacity_bars(ax3)
        
        # 4. Layer-wise parameter breakdown
        ax4 = fig.add_subplot(gs[1, 1:])
        self._plot_layer_breakdown(ax4)
        
        # 5. Bottleneck analysis text
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_bottleneck_analysis(ax5)
        
        fig.suptitle('Hybrid Network Capacity Analysis', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Capacity analysis saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _plot_param_distribution(self, ax):
        """Plot parameter distribution pie chart."""
        sizes = [
            self.capacity['encoder']['total_params'],
            self.capacity['quantum']['total_params'],
            self.capacity['postprocessing']['total_params']
        ]
        labels = ['Encoder', 'Quantum', 'Postprocessing']
        colors = ['#3498db', '#FFD700', '#e74c3c']
        
        # Only plot if there are non-zero values
        if sum(sizes) == 0:
            ax.text(0.5, 0.5, 'No parameters detected\n(Classical network)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Parameter Distribution', fontsize=12, fontweight='bold')
            return
        
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10, 'weight': 'bold'}
        )
        
        ax.set_title('Parameter Distribution', fontsize=12, fontweight='bold')
        
        # Add parameter counts
        for i, (label, size) in enumerate(zip(labels, sizes)):
            texts[i].set_text(f'{label}\n({size} params)')
    
    def _plot_architecture_flow(self, ax):
        """Plot architecture information flow diagram."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Draw components
        encoder = plt.Rectangle((0.5, 4), 1.5, 2, facecolor='#3498db', alpha=0.6, edgecolor='black', linewidth=2)
        quantum = plt.Rectangle((3, 3.5), 2, 3, facecolor='#FFD700', alpha=0.6, edgecolor='black', linewidth=2)
        postproc = plt.Rectangle((6, 4), 1.5, 2, facecolor='#e74c3c', alpha=0.6, edgecolor='black', linewidth=2)
        
        ax.add_patch(encoder)
        ax.add_patch(quantum)
        ax.add_patch(postproc)
        
        # Labels
        ax.text(1.25, 5, 'ENCODER', ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(1.25, 4.5, f"{self.capacity['encoder']['layer_dims'][0][0]}→{self.capacity['encoder']['layer_dims'][-1][1]}", 
                ha='center', va='center', fontsize=9)
        ax.text(1.25, 4.2, f"{self.capacity['encoder']['total_params']} params", ha='center', va='center', fontsize=8)
        
        ax.text(4, 5.5, 'QUANTUM', ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(4, 5, f"{self.capacity['quantum']['n_qubits']} qubits × {self.capacity['quantum']['n_layers']} layers", 
                ha='center', va='center', fontsize=9)
        ax.text(4, 4.5, f"{self.capacity['quantum']['total_params']} params", ha='center', va='center', fontsize=8)
        
        ax.text(6.75, 5, 'POSTPROC', ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(6.75, 4.5, f"{self.capacity['postprocessing']['layer_dims'][0][0]}→{self.capacity['postprocessing']['layer_dims'][-1][1]}", 
                ha='center', va='center', fontsize=9)
        ax.text(6.75, 4.2, f"{self.capacity['postprocessing']['total_params']} params", ha='center', va='center', fontsize=8)
        
        # Arrows
        ax.arrow(2.1, 5, 0.8, 0, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=2)
        ax.arrow(5.1, 5, 0.8, 0, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=2)
        
        # Input/Output
        ax.text(0.5, 5, 'Input:\n(3 features)', ha='right', va='center', fontsize=9, style='italic')
        ax.text(8.5, 5, 'Output:\n(2 actions)', ha='left', va='center', fontsize=9, style='italic')
        
        # Warning if minimal
        if self.capacity['postprocessing']['is_minimal']:
            ax.text(6.75, 3.5, '[WARN] MINIMAL', ha='center', va='top', fontsize=10, 
                   color='red', fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            ax.text(6.75, 3.0, 'No hidden layers!\nCannot compensate', ha='center', va='top', fontsize=8, color='red')
        
        ax.set_title('Information Flow Architecture', fontsize=12, fontweight='bold', pad=10)
    
    def _plot_capacity_bars(self, ax):
        """Plot capacity comparison bars."""
        components = ['Encoder', 'Quantum', 'Postproc']
        params = [
            self.capacity['encoder']['total_params'],
            self.capacity['quantum']['total_params'],
            self.capacity['postprocessing']['total_params']
        ]
        colors = ['#3498db', '#FFD700', '#e74c3c']
        
        bars = ax.barh(components, params, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, param in zip(bars, params):
            ax.text(param + 1, bar.get_y() + bar.get_height()/2, f'{param}', 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Number of Parameters', fontsize=11, fontweight='bold')
        ax.set_title('Parameter Capacity', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Highlight minimal postprocessing
        if self.capacity['postprocessing']['is_minimal']:
            ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.text(0, 0.5, 'BOTTLENECK →', ha='right', va='center', 
                   fontsize=9, color='red', fontweight='bold')
    
    def _plot_layer_breakdown(self, ax):
        """Plot layer-wise parameter breakdown."""
        # Gather all layers
        layers = []
        params = []
        colors = []
        
        # Encoder layers
        for i, (p, dims) in enumerate(zip(
            self.capacity['encoder']['params_per_layer'],
            self.capacity['encoder']['layer_dims']
        )):
            layers.append(f'Enc {i+1}\n{dims[0]}→{dims[1]}')
            params.append(p)
            colors.append('#3498db')
        
        # Quantum (single component)
        layers.append(f"Quantum\n{self.capacity['quantum']['n_qubits']}q×{self.capacity['quantum']['n_layers']}L")
        params.append(self.capacity['quantum']['total_params'])
        colors.append('#FFD700')
        
        # Postprocessing layers
        for i, (p, dims) in enumerate(zip(
            self.capacity['postprocessing']['params_per_layer'],
            self.capacity['postprocessing']['layer_dims']
        )):
            layers.append(f'Post {i+1}\n{dims[0]}→{dims[1]}')
            params.append(p)
            colors.append('#e74c3c')
        
        x = np.arange(len(layers))
        bars = ax.bar(x, params, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(layers, fontsize=9)
        ax.set_ylabel('Parameters', fontsize=11, fontweight='bold')
        ax.set_title('Layer-wise Parameter Count', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, param in zip(bars, params):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{param}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    def _plot_bottleneck_analysis(self, ax):
        """Plot bottleneck analysis as formatted text."""
        ax.axis('off')
        
        if not self.bottlenecks['has_bottleneck']:
            text = "✅ No significant capacity bottlenecks detected"
            ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=14, 
                   transform=ax.transAxes, color='green', fontweight='bold')
            return
        
        # Build bottleneck report
        y_pos = 0.95
        
        # Title
        ax.text(0.05, y_pos, '[WARN] CAPACITY BOTTLENECK DETECTED', 
               fontsize=14, fontweight='bold', color='red', transform=ax.transAxes)
        y_pos -= 0.15
        
        # Type
        ax.text(0.05, y_pos, f"Type: {self.bottlenecks['bottleneck_type']}", 
               fontsize=11, fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.12
        
        # Explanation
        ax.text(0.05, y_pos, self.bottlenecks['explanation'], 
               fontsize=10, transform=ax.transAxes, wrap=True)
        y_pos -= 0.15
        
        # Implications
        ax.text(0.05, y_pos, "Implications:", fontsize=11, fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.1
        
        for impl in self.bottlenecks['implications']:
            ax.text(0.08, y_pos, impl, fontsize=9, transform=ax.transAxes, family='monospace')
            y_pos -= 0.08
        
        # Add box around everything
        bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.3, edgecolor='red', linewidth=2)
        ax.text(0.5, 0.5, '', transform=ax.transAxes, bbox=bbox, 
               zorder=-1, ha='center', va='center', fontsize=100)
    
    def print_report(self):
        """Print formatted capacity analysis report."""
        print("\n" + "="*80)
        print("NETWORK CAPACITY ANALYSIS")
        print("="*80)
        
        # Overall summary
        print(f"\n[DATA] OVERALL CAPACITY:")
        print(f"   Total parameters: {self.capacity['total']['total_params']}")
        
        if self.capacity['quantum']:
            print(f"   Quantum:          {self.capacity['quantum'].get('total_params', 0)} ({self.capacity['total']['quantum_ratio']*100:.1f}%)")
        
        if self.capacity['encoder']:
            print(f"   Encoder:          {self.capacity['encoder'].get('total_params', 0)} ({self.capacity['total']['encoder_ratio']*100:.1f}%)")
        
        if self.capacity['postprocessing']:
            print(f"   Postprocessing:   {self.capacity['postprocessing'].get('total_params', 0)} ({self.capacity['total']['postproc_ratio']*100:.1f}%)")
        
        # Component details
        if self.capacity['encoder']:
            print(f"\n[ENCODER]:")
            print(f"   Layers: {self.capacity['encoder']['num_layers']} ({self.capacity['encoder']['hidden_layers']} hidden)")
            for i, dims in enumerate(self.capacity['encoder']['layer_dims']):
                print(f"   Layer {i+1}: {dims[0]} -> {dims[1]} ({self.capacity['encoder']['params_per_layer'][i]} params)")
        
        if self.capacity['quantum']:
            print(f"\n[QUANTUM]  CIRCUIT:")
            print(f"   Qubits: {self.capacity['quantum']['n_qubits']}")
            print(f"   Layers: {self.capacity['quantum']['n_layers']}")
            print(f"   Hilbert space dimension: {self.capacity['quantum']['hilbert_space_dim']}")
            print(f"   Variational parameters: {self.capacity['quantum']['total_params']}")
        
        if self.capacity['postprocessing']:
            print(f"\n[POSTPROCESSING]:")
            print(f"   Layers: {self.capacity['postprocessing']['num_layers']} ({self.capacity['postprocessing']['hidden_layers']} hidden)")
            for i, dims in enumerate(self.capacity['postprocessing']['layer_dims']):
                print(f"   Layer {i+1}: {dims[0]} -> {dims[1]} ({self.capacity['postprocessing']['params_per_layer'][i]} params)")
            print(f"   Is minimal: {self.capacity['postprocessing']['is_minimal']}")
            print(f"   Can compensate: {self.capacity['postprocessing']['can_compensate']}")
        
        
        # Bottlenecks
        if self.bottlenecks['has_bottleneck']:
            print(f"\n[BOTTLENECK] DETECTED: {self.bottlenecks['bottleneck_type']}")
            print(f"   {self.bottlenecks['explanation']}")
            print(f"\n   Implications:")
            for impl in self.bottlenecks['implications']:
                print(f"     {impl}")
        else:
            print(f"\n[OK] No significant bottlenecks detected")
        
        print("\n" + "="*80 + "\n")
    
    def save_analysis(self, save_dir: str):
        """Save capacity analysis to JSON."""
        save_path = Path(save_dir) / 'capacity_analysis.json'
        
        analysis = {
            'capacity': self.capacity,
            'bottlenecks': self.bottlenecks
        }
        
        with open(save_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"[INFO] Capacity analysis saved to {save_path}")


def analyze_network_capacity(network: nn.Module, save_dir: Optional[str] = None) -> CapacityAnalyzer:
    """
    Convenience function to analyze network capacity.
    
    Args:
        network: Hybrid network to analyze
        save_dir: Directory to save visualizations and JSON
    
    Returns:
        CapacityAnalyzer instance
    """
    analyzer = CapacityAnalyzer(network)
    analyzer.print_report()
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        analyzer.visualize_architecture(
            save_path=str(save_dir / 'capacity_analysis.png'),
            show=False
        )
        analyzer.save_analysis(str(save_dir))
    
    return analyzer
