"""Gradient flow analysis for hybrid quantum-classical networks.

This module answers the key question: 
"Is the quantum part actually doing the heavy lifting, or are the classical parts doing all the work?"

Provides:
1. Gradient flow analysis through quantum vs classical components
2. Contribution metrics (what % comes from each component)
3. Quantum parameter utilization analysis
4. Ablation-style comparison tools

Note: Previously named quantum.py. Renamed to gradient_flow.py to distinguish
from signal_noise.py which analyzes whether quantum output is signal vs noise.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


class GradientFlowAnalyzer:
    """
    Analyzes gradient flow through quantum vs classical components during training.
    
    Key metrics:
    - Gradient contribution %: What fraction of gradients flow through quantum vs classical
    - Parameter utilization: Are quantum parameters being used or staying at init values
    - Decision influence: How much do quantum outputs affect final decisions
    - Learning velocity: Is the quantum part improving over training
    
    Previously named QuantumContributionAnalyzer in quantum.py.
    """
    
    def __init__(self, network: nn.Module, history_size: int = 10000, n_qubits: int = None, **kwargs):
        """
        Initialize the analyzer.
        
        Args:
            network: Hybrid network with quantum_circuit, feature_encoder, postprocessing
            history_size: How many updates to track
            n_qubits: Number of qubits (optional, auto-detected from network if not provided)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.network = network
        self.history_size = history_size
        self.n_qubits = n_qubits or getattr(network, 'n_qubits', None)
        
        # Validate network structure
        self._validate_network()
        
        # Analyze network capacity
        self.network_capacity = self._analyze_network_capacity()
        
        # History tracking
        self.history = {
            'step': deque(maxlen=history_size),
            'quantum_grad_norm': deque(maxlen=history_size),
            'classical_grad_norm': deque(maxlen=history_size),
            'quantum_contrib_pct': deque(maxlen=history_size),
            'quantum_param_change': deque(maxlen=history_size),
            'encoder_grad_norm': deque(maxlen=history_size),
            'postproc_grad_norm': deque(maxlen=history_size),
        }
        
        self.step_count = 0
        self.initial_quantum_params = None
        self._capture_initial_params()
        
        # Thresholds
        self.vanishing_threshold = 1e-7
        self.minimum_contribution_threshold = 1.0  # Quantum should contribute at least 1%
    
    def _validate_network(self):
        """Check that network has required hybrid components."""
        required = ['feature_encoder', 'quantum_circuit', 'postprocessing']
        missing = [r for r in required if not hasattr(self.network, r)]
        
        if missing:
            raise ValueError(
                f"Network missing required hybrid components: {missing}. "
                f"Expected: feature_encoder, quantum_circuit, postprocessing"
            )
        
        # Check for quantum weights
        if not hasattr(self.network, 'weights'):
            raise ValueError("Network must have 'weights' parameter for quantum circuit")
    
    def _analyze_network_capacity(self) -> Dict:
        """Analyze the parameter capacity of quantum vs classical components.
        
        This is critical for interpretation: if classical components have minimal
        capacity (e.g., no hidden layers in postprocessing), they cannot 'take over'
        from quantum even if quantum contribution declines.
        """
        capacity = {
            'quantum_params': 0,
            'encoder_params': 0,
            'postproc_params': 0,
            'postproc_hidden_layers': 0,
            'encoder_hidden_layers': 0,
        }
        
        # Count quantum parameters
        if hasattr(self.network, 'weights'):
            capacity['quantum_params'] = self.network.weights.numel()
        
        # Count encoder parameters and layers
        if hasattr(self.network, 'feature_encoder'):
            capacity['encoder_params'] = sum(p.numel() for p in self.network.feature_encoder.parameters())
            # Count layers (excluding activations)
            capacity['encoder_hidden_layers'] = sum(1 for m in self.network.feature_encoder.modules() 
                                                   if isinstance(m, nn.Linear)) - 1
        
        # Count postprocessing parameters and layers
        if hasattr(self.network, 'postprocessing'):
            capacity['postproc_params'] = sum(p.numel() for p in self.network.postprocessing.parameters())
            # Count layers (excluding activations)
            capacity['postproc_hidden_layers'] = sum(1 for m in self.network.postprocessing.modules() 
                                                     if isinstance(m, nn.Linear)) - 1
        
        capacity['total_params'] = (capacity['quantum_params'] + 
                                   capacity['encoder_params'] + 
                                   capacity['postproc_params'])
        capacity['quantum_param_ratio'] = (capacity['quantum_params'] / capacity['total_params'] 
                                          if capacity['total_params'] > 0 else 0)
        
        # Determine if postprocessing is minimal (direct mapping)
        capacity['postproc_is_minimal'] = capacity['postproc_hidden_layers'] == 0
        
        return capacity
    
    def _capture_initial_params(self):
        """Capture initial quantum parameters for comparison."""
        if hasattr(self.network, 'weights') and self.network.weights is not None:
            self.initial_quantum_params = self.network.weights.detach().clone()
    
    def _get_gradient_norms(self) -> Dict[str, float]:
        """Extract gradient norms from each component."""
        norms = {
            'encoder': 0.0,
            'quantum': 0.0,
            'postproc': 0.0
        }
        
        # Feature encoder
        for param in self.network.feature_encoder.parameters():
            if param.grad is not None:
                norms['encoder'] += param.grad.norm(2).item() ** 2
        norms['encoder'] = np.sqrt(norms['encoder'])
        
        # Quantum weights
        if self.network.weights is not None and self.network.weights.grad is not None:
            norms['quantum'] = self.network.weights.grad.norm(2).item()
        
        # Postprocessing
        for param in self.network.postprocessing.parameters():
            if param.grad is not None:
                norms['postproc'] += param.grad.norm(2).item() ** 2
        norms['postproc'] = np.sqrt(norms['postproc'])
        
        return norms
    
    def capture_gradients(self) -> Dict[str, float]:
        """
        Capture and analyze current gradient state.
        
        Call after loss.backward() but before optimizer.step().
        
        Returns:
            Dictionary with contribution metrics
        """
        self.step_count += 1
        
        norms = self._get_gradient_norms()
        
        # Compute totals
        classical_norm = np.sqrt(norms['encoder']**2 + norms['postproc']**2)
        quantum_norm = norms['quantum']
        total_norm = np.sqrt(classical_norm**2 + quantum_norm**2)
        
        # Compute contributions
        if total_norm > 0:
            quantum_contrib = (quantum_norm / total_norm) * 100
        else:
            quantum_contrib = 0.0
        
        # Compute quantum parameter change from initial
        if self.initial_quantum_params is not None:
            current_params = self.network.weights.detach()
            param_change = (current_params - self.initial_quantum_params).norm(2).item()
        else:
            param_change = 0.0
        
        # Store metrics
        metrics = {
            'quantum_grad_norm': quantum_norm,
            'classical_grad_norm': classical_norm,
            'quantum_contrib_pct': quantum_contrib,
            'quantum_param_change': param_change,
            'encoder_grad_norm': norms['encoder'],
            'postproc_grad_norm': norms['postproc'],
            'total_grad_norm': total_norm
        }
        
        # Update history
        self.history['step'].append(self.step_count)
        for key in ['quantum_grad_norm', 'classical_grad_norm', 'quantum_contrib_pct',
                    'quantum_param_change', 'encoder_grad_norm', 'postproc_grad_norm']:
            self.history[key].append(metrics[key])
        
        return metrics
    
    def get_contribution_summary(self) -> Dict:
        """
        Get comprehensive summary of quantum contribution.
        
        Returns:
            Dictionary with contribution analysis
        """
        if len(self.history['step']) < 10:
            return {'error': 'Insufficient data (need at least 10 updates)'}
        
        quantum_contribs = list(self.history['quantum_contrib_pct'])
        quantum_norms = list(self.history['quantum_grad_norm'])
        classical_norms = list(self.history['classical_grad_norm'])
        param_changes = list(self.history['quantum_param_change'])
        
        # Overall contribution
        avg_quantum_contrib = np.mean(quantum_contribs)
        avg_classical_contrib = 100 - avg_quantum_contrib
        
        # Trend analysis (is quantum contribution increasing or decreasing?)
        n_recent = min(100, len(quantum_contribs))
        n_early = min(100, len(quantum_contribs))
        
        early_avg = np.mean(quantum_contribs[:n_early])
        recent_avg = np.mean(quantum_contribs[-n_recent:])
        trend = recent_avg - early_avg
        
        # Gradient health
        vanishing_count = sum(1 for q in quantum_norms if q < self.vanishing_threshold)
        vanishing_pct = (vanishing_count / len(quantum_norms)) * 100
        
        # Parameter utilization
        final_param_change = param_changes[-1] if param_changes else 0
        param_utilization = final_param_change / (self.initial_quantum_params.norm(2).item() + 1e-8) * 100
        
        # Determine verdict (accounting for network capacity)
        postproc_minimal = self.network_capacity.get('postproc_is_minimal', False)
        quantum_ratio = self.network_capacity.get('quantum_param_ratio', 0)
        
        if avg_quantum_contrib < self.minimum_contribution_threshold:
            verdict = "QUANTUM NOT CONTRIBUTING"
            if postproc_minimal:
                verdict_detail = (f"Quantum gradients negligible. With minimal postprocessing "
                                f"({self.network_capacity['postproc_params']} params, no hidden layers), "
                                f"quantum MUST contribute for good performance.")
            else:
                verdict_detail = "Quantum gradients are negligible. Classical parts doing all the work."
        elif vanishing_pct > 50:
            verdict = "QUANTUM GRADIENTS VANISHING"
            verdict_detail = f"Quantum gradients vanish {vanishing_pct:.1f}% of the time."
        elif trend < -10:
            verdict = "QUANTUM CONTRIBUTION DECLINING"
            if postproc_minimal:
                verdict_detail = (f"Quantum contribution dropped {abs(trend):.1f}% over training. "
                                f"With minimal classical capacity (postproc: {self.network_capacity['postproc_params']} params, "
                                f"no hidden layers), this suggests quantum learned initial features but "
                                f"couldn't maintain specialized role. Performance may suffer.")
            else:
                verdict_detail = f"Quantum contribution dropped {abs(trend):.1f}% over training. Classical may be compensating."
        elif avg_quantum_contrib > 30:
            verdict = "QUANTUM CONTRIBUTING SIGNIFICANTLY"
            verdict_detail = f"Quantum provides {avg_quantum_contrib:.1f}% of gradients."
        else:
            verdict = "QUANTUM CONTRIBUTING MODERATELY"
            if postproc_minimal:
                verdict_detail = (f"Quantum provides {avg_quantum_contrib:.1f}% of gradients. "
                                f"Minimal postprocessing design forces quantum to carry decision logic.")
            else:
                verdict_detail = f"Quantum provides {avg_quantum_contrib:.1f}% of gradients."
        
        return {
            'verdict': verdict,
            'verdict_detail': verdict_detail,
            'metrics': {
                'avg_quantum_contribution_%': avg_quantum_contrib,
                'avg_classical_contribution_%': avg_classical_contrib,
                'contribution_trend': trend,
                'vanishing_gradient_%': vanishing_pct,
                'quantum_param_utilization_%': param_utilization,
                'total_updates_analyzed': len(quantum_contribs),
            },
            'component_breakdown': {
                'quantum_avg_norm': np.mean(quantum_norms),
                'encoder_avg_norm': np.mean(list(self.history['encoder_grad_norm'])),
                'postproc_avg_norm': np.mean(list(self.history['postproc_grad_norm'])),
            },
            'network_capacity': self.network_capacity
        }
    
    def print_summary(self):
        """Print formatted contribution summary."""
        summary = self.get_contribution_summary()
        
        if 'error' in summary:
            print(f"[WARN] {summary['error']}")
            return
        
        print("\n" + "="*70)
        print("QUANTUM CONTRIBUTION ANALYSIS")
        print("="*70)
        
        # Network capacity info
        if 'network_capacity' in summary:
            cap = summary['network_capacity']
            print(f"\n[BUILD]  NETWORK ARCHITECTURE:")
            print(f"   Quantum params:    {cap['quantum_params']:>4} ({cap['quantum_param_ratio']*100:.1f}% of total)")
            print(f"   Encoder params:    {cap['encoder_params']:>4} ({cap['encoder_hidden_layers']} hidden layers)")
            print(f"   Postproc params:   {cap['postproc_params']:>4} ({cap['postproc_hidden_layers']} hidden layers)")
            print(f"   Total params:      {cap['total_params']:>4}")
            if cap['postproc_is_minimal']:
                print(f"   [WARN]  Postprocessing is MINIMAL (no hidden layers)")
                print(f"       → Classical cannot 'compensate' for weak quantum contribution")
        
        print(f"\n🔮 VERDICT: {summary['verdict']}")
        print(f"   {summary['verdict_detail']}")
        
        metrics = summary['metrics']
        print(f"\n[DATA] CONTRIBUTION METRICS:")
        print(f"   Quantum contribution:  {metrics['avg_quantum_contribution_%']:.2f}%")
        print(f"   Classical contribution: {metrics['avg_classical_contribution_%']:.2f}%")
        print(f"   Contribution trend:    {metrics['contribution_trend']:+.2f}%")
        print(f"   Vanishing gradients:   {metrics['vanishing_gradient_%']:.1f}%")
        print(f"   Param utilization:     {metrics['quantum_param_utilization_%']:.1f}%")
        
        breakdown = summary['component_breakdown']
        print(f"\n[TEST] COMPONENT BREAKDOWN (avg gradient norms):")
        print(f"   Quantum circuit:  {breakdown['quantum_avg_norm']:.4e}")
        print(f"   Feature encoder:  {breakdown['encoder_avg_norm']:.4e}")
        print(f"   Postprocessing:   {breakdown['postproc_avg_norm']:.4e}")
        
        print("="*70 + "\n")
    
    def plot_gradient_analysis(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
        figsize: Tuple[int, int] = (16, 10)
    ):
        """
        Alias for plot_contribution_analysis for backward compatibility.
        
        Args:
            save_path: Path to save figure
            show: Whether to display plot
            figsize: Figure size
        """
        return self.plot_contribution_analysis(save_path=save_path, show=show, figsize=figsize)
    
    def plot_contribution_analysis(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
        figsize: Tuple[int, int] = (16, 10)
    ):
        """
        Create comprehensive visualization of quantum contribution.
        
        Args:
            save_path: Path to save figure
            show: Whether to display plot
            figsize: Figure size
        """
        if len(self.history['step']) < 2:
            print("[WARN] Insufficient data for visualization")
            return
        
        steps = np.array(list(self.history['step']))
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. Quantum vs Classical contribution over time
        ax = axes[0, 0]
        quantum_contrib = np.array(list(self.history['quantum_contrib_pct']))
        classical_contrib = 100 - quantum_contrib
        
        ax.stackplot(steps, quantum_contrib, classical_contrib,
                    labels=['Quantum', 'Classical'],
                    colors=['#FFD700', '#3498db'], alpha=0.8)
        ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% line')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Contribution (%)')
        ax.set_title('Quantum vs Classical Contribution')
        ax.set_ylim([0, 100])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Gradient norms comparison
        ax = axes[0, 1]
        quantum_norms = np.array(list(self.history['quantum_grad_norm']))
        classical_norms = np.array(list(self.history['classical_grad_norm']))
        
        ax.semilogy(steps, quantum_norms, label='Quantum', color='#FFD700', linewidth=2)
        ax.semilogy(steps, classical_norms, label='Classical', color='#3498db', linewidth=2)
        ax.axhline(self.vanishing_threshold, color='red', linestyle='--', 
                  alpha=0.5, label='Vanishing threshold')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Gradient Norm (log)')
        ax.set_title('Gradient Norms Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Quantum parameter change
        ax = axes[0, 2]
        param_change = np.array(list(self.history['quantum_param_change']))
        
        ax.plot(steps, param_change, color='purple', linewidth=2)
        ax.fill_between(steps, 0, param_change, alpha=0.3, color='purple')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Parameter Change (L2 norm)')
        ax.set_title('Quantum Parameter Evolution')
        ax.grid(True, alpha=0.3)
        
        # 4. Per-component gradient norms
        ax = axes[1, 0]
        encoder_norms = np.array(list(self.history['encoder_grad_norm']))
        postproc_norms = np.array(list(self.history['postproc_grad_norm']))
        
        ax.semilogy(steps, encoder_norms, label='Feature Encoder', color='#2ecc71', linewidth=2)
        ax.semilogy(steps, quantum_norms, label='Quantum Circuit', color='#FFD700', linewidth=2)
        ax.semilogy(steps, postproc_norms, label='Postprocessing', color='#e74c3c', linewidth=2)
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Gradient Norm (log)')
        ax.set_title('All Component Gradient Norms')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Contribution histogram
        ax = axes[1, 1]
        ax.hist(quantum_contrib, bins=50, color='#FFD700', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(quantum_contrib), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(quantum_contrib):.1f}%')
        ax.set_xlabel('Quantum Contribution (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Quantum Contribution Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Rolling average contribution
        ax = axes[1, 2]
        window = min(100, len(quantum_contrib) // 5)
        if window > 1:
            rolling_avg = np.convolve(quantum_contrib, np.ones(window)/window, mode='valid')
            rolling_steps = steps[:len(rolling_avg)]
            ax.plot(rolling_steps, rolling_avg, color='#FFD700', linewidth=2)
            ax.fill_between(rolling_steps, 0, rolling_avg, alpha=0.3, color='#FFD700')
        else:
            ax.plot(steps, quantum_contrib, color='#FFD700', linewidth=2)
        
        ax.axhline(50, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Contribution (%)')
        ax.set_title(f'Rolling Average Quantum Contribution (window={window})')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)
        
        # Add overall title with verdict
        summary = self.get_contribution_summary()
        if 'verdict' in summary:
            fig.suptitle(f"Quantum Contribution Analysis\n{summary['verdict']}", 
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Quantum contribution plot saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def save_analysis(self, filepath: str):
        """Save analysis results to JSON.
        
        Args:
            filepath: File path or directory. If directory, saves as 'quantum_analysis.json'
        
        Returns:
            Path: The path where the file was saved
        """
        from pathlib import Path
        path = Path(filepath)
        
        # If directory, create filename
        if path.is_dir() or not path.suffix:
            path = path / 'quantum_analysis.json'
            path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.get_contribution_summary()
        
        # Add history data (sampled if too large)
        history_data = {}
        max_points = 1000
        total_points = len(self.history['step'])
        
        if total_points > max_points:
            step = total_points // max_points
            indices = list(range(0, total_points, step))
        else:
            indices = list(range(total_points))
        
        for key, values in self.history.items():
            values_list = list(values)
            history_data[key] = [values_list[i] for i in indices]
        
        output = {
            'summary': summary,
            'history_sample': history_data,
            'total_updates': self.step_count
        }
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"[OK] Quantum analysis saved: {path}")
        return path


def analyze_model_quantum_contribution(
    checkpoint_path: str,
    model_factory_fn,
    test_states: Optional[List[Tuple[int, int, int]]] = None,
    n_samples: int = 100
) -> Dict:
    """
    Analyze quantum contribution for a single checkpoint.
    
    Performs forward passes and measures:
    - How much quantum layer affects outputs
    - Output variance from quantum vs classical
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_factory_fn: Function to create model instance
        test_states: Optional list of states to test
        n_samples: Number of test samples if test_states not provided
        
    Returns:
        Dictionary with contribution analysis
    """
    # Create and load model
    model = model_factory_fn()
    if isinstance(model, tuple):
        policy_net = model[0]
    else:
        policy_net = model
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'policy_net' in checkpoint:
        policy_net.load_state_dict(checkpoint['policy_net'])
    elif 'policy_state_dict' in checkpoint:
        policy_net.load_state_dict(checkpoint['policy_state_dict'])
    else:
        policy_net.load_state_dict(checkpoint)
    
    policy_net.eval()
    
    # Generate test states if not provided
    if test_states is None:
        test_states = []
        for _ in range(n_samples):
            player_sum = np.random.randint(4, 22)
            dealer_card = np.random.randint(1, 11)
            usable_ace = np.random.randint(0, 2)
            test_states.append((player_sum, dealer_card, usable_ace))
    
    # Check if model has forward_with_intermediates
    if not hasattr(policy_net, 'forward_with_intermediates'):
        return {
            'error': 'Model does not have forward_with_intermediates method',
            'suggestion': 'Add forward_with_intermediates() to hybrid network class'
        }
    
    # Analyze outputs
    results = {
        'quantum_output_variance': [],
        'classical_output_variance': [],
        'output_correlation': [],
    }
    
    with torch.no_grad():
        for state in test_states:
            intermediates = policy_net.forward_with_intermediates(state)
            
            if 'quantum_output' in intermediates:
                q_out = intermediates['quantum_output']
                if isinstance(q_out, torch.Tensor):
                    results['quantum_output_variance'].append(q_out.var().item())
            
            if 'encoder_output' in intermediates:
                c_out = intermediates['encoder_output']
                if isinstance(c_out, torch.Tensor):
                    results['classical_output_variance'].append(c_out.var().item())
    
    return {
        'avg_quantum_variance': np.mean(results['quantum_output_variance']) if results['quantum_output_variance'] else 0,
        'avg_classical_variance': np.mean(results['classical_output_variance']) if results['classical_output_variance'] else 0,
        'n_samples': len(test_states),
        'checkpoint': checkpoint_path
    }


def compare_quantum_classical_performance(
    hybrid_checkpoint: str,
    classical_checkpoint: str,
    hybrid_factory_fn,
    classical_factory_fn,
    n_eval_episodes: int = 100
) -> Dict:
    """
    Compare hybrid and classical model performance to assess quantum value.
    
    Args:
        hybrid_checkpoint: Path to hybrid model checkpoint
        classical_checkpoint: Path to classical model checkpoint
        hybrid_factory_fn: Function to create hybrid model
        classical_factory_fn: Function to create classical model
        n_eval_episodes: Number of evaluation episodes
        
    Returns:
        Comparison results dictionary
    """
    import gymnasium as gym
    
    # Load models
    hybrid_model = hybrid_factory_fn()
    classical_model = classical_factory_fn()
    
    if isinstance(hybrid_model, tuple):
        hybrid_model = hybrid_model[0]
    if isinstance(classical_model, tuple):
        classical_model = classical_model[0]
    
    # Load checkpoints
    for model, path in [(hybrid_model, hybrid_checkpoint), 
                        (classical_model, classical_checkpoint)]:
        ckpt = torch.load(path, map_location='cpu')
        if 'policy_net' in ckpt:
            model.load_state_dict(ckpt['policy_net'])
        elif 'policy_state_dict' in ckpt:
            model.load_state_dict(ckpt['policy_state_dict'])
        else:
            model.load_state_dict(ckpt)
        model.eval()
    
    # Evaluate both models
    env = gym.make('Blackjack-v1')
    
    results = {'hybrid': {'wins': 0, 'losses': 0, 'draws': 0},
               'classical': {'wins': 0, 'losses': 0, 'draws': 0}}
    
    for model_name, model in [('hybrid', hybrid_model), ('classical', classical_model)]:
        for _ in range(n_eval_episodes):
            state, _ = env.reset()
            done = False
            
            while not done:
                with torch.no_grad():
                    probs = model(state).squeeze()
                    action = torch.argmax(probs).item()
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            
            if reward > 0:
                results[model_name]['wins'] += 1
            elif reward < 0:
                results[model_name]['losses'] += 1
            else:
                results[model_name]['draws'] += 1
    
    env.close()
    
    # Compute metrics
    hybrid_wr = results['hybrid']['wins'] / n_eval_episodes * 100
    classical_wr = results['classical']['wins'] / n_eval_episodes * 100
    
    return {
        'hybrid_win_rate': hybrid_wr,
        'classical_win_rate': classical_wr,
        'quantum_advantage': hybrid_wr - classical_wr,
        'quantum_value': 'POSITIVE' if hybrid_wr > classical_wr else 'NEGATIVE' if hybrid_wr < classical_wr else 'NEUTRAL',
        'n_episodes': n_eval_episodes,
        'detailed_results': results
    }


# Aliases for backward compatibility
QuantumContributionAnalyzer = GradientFlowAnalyzer  # Old name from quantum.py
HybridGradientAnalyzer = GradientFlowAnalyzer  # Common alias
