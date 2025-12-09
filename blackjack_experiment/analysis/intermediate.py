"""
Intermediate output analysis for hybrid quantum-classical networks.

This module answers the core question:
"Is the quantum circuit making the decision, or is postprocessing doing all the work?"

Measures variance, entropy, and information content at each stage:
1. Feature encoder output (classical preprocessing)
2. Quantum circuit output (quantum processing)
3. Final action probabilities (postprocessing output)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from .utils import generate_all_blackjack_states


class IntermediateOutputAnalyzer:
    """
    Analyzes intermediate outputs at each stage of a hybrid network.
    
    Key metrics:
    - Output variance: How much does each stage's output vary across states?
    - Entropy: How much information is in each stage?
    - Decision correlation: Does quantum output predict final action?
    - Transformation magnitude: Is postprocessing passthrough or transformative?
    """
    
    def __init__(self, policy_net):
        """
        Initialize analyzer with a hybrid policy network.
        
        Args:
            policy_net: Network with forward_with_intermediates() method
        """
        self.policy_net = policy_net
        self.policy_net.eval()
        
        # Verify network has required method
        if not hasattr(policy_net, 'forward_with_intermediates'):
            raise ValueError("Network must have forward_with_intermediates() method")
    
    def _generate_all_states(self) -> List[Tuple[int, int, int]]:
        """Generate all valid Blackjack states."""
        return generate_all_blackjack_states()
    
    def analyze_single_state(self, state: Tuple[int, int, int]) -> Dict:
        """
        Analyze intermediate outputs for a single state.
        
        Args:
            state: (player_sum, dealer_card, usable_ace)
            
        Returns:
            Dictionary with outputs and metrics for each stage
        """
        with torch.no_grad():
            intermediates = self.policy_net.forward_with_intermediates(state)
        
        # Extract outputs
        encoder_out = intermediates['feature_encoder_output'].squeeze().numpy()
        quantum_out = intermediates['quantum_output'].squeeze().numpy()
        action_probs = intermediates['action_probs'].squeeze().numpy()
        
        # Compute per-stage metrics
        results = {
            'state': state,
            'encoder_output': {
                'values': encoder_out.tolist(),
                'mean': float(np.mean(encoder_out)),
                'std': float(np.std(encoder_out)),
                'min': float(np.min(encoder_out)),
                'max': float(np.max(encoder_out)),
                'l2_norm': float(np.linalg.norm(encoder_out))
            },
            'quantum_output': {
                'values': quantum_out.tolist(),
                'mean': float(np.mean(quantum_out)),
                'std': float(np.std(quantum_out)),
                'min': float(np.min(quantum_out)),
                'max': float(np.max(quantum_out)),
                'l2_norm': float(np.linalg.norm(quantum_out))
            },
            'action_probs': {
                'stick': float(action_probs[0]),
                'hit': float(action_probs[1]),
                'entropy': float(-np.sum(action_probs * np.log(action_probs + 1e-10))),
                'confidence': float(np.max(action_probs)),
                'decision': 'hit' if action_probs[1] > action_probs[0] else 'stick'
            }
        }
        
        return results
    
    def analyze_all_states(self) -> Dict:
        """
        Analyze intermediate outputs across all Blackjack states.
        
        Returns:
            Comprehensive analysis with per-state and aggregate metrics
        """
        states = self._generate_all_states()
        
        # Collect outputs for all states
        encoder_outputs = []
        quantum_outputs = []
        action_probs_list = []
        decisions = []
        
        for state in states:
            with torch.no_grad():
                intermediates = self.policy_net.forward_with_intermediates(state)
            
            encoder_outputs.append(intermediates['feature_encoder_output'].squeeze().numpy())
            quantum_outputs.append(intermediates['quantum_output'].squeeze().numpy())
            action_probs_list.append(intermediates['action_probs'].squeeze().numpy())
            decisions.append(torch.argmax(intermediates['action_probs']).item())
        
        encoder_outputs = np.array(encoder_outputs)
        quantum_outputs = np.array(quantum_outputs)
        action_probs = np.array(action_probs_list)
        
        # Aggregate metrics
        results = {
            'n_states': len(states),
            'encoder_analysis': self._analyze_stage_outputs(encoder_outputs, 'encoder'),
            'quantum_analysis': self._analyze_stage_outputs(quantum_outputs, 'quantum'),
            'action_analysis': self._analyze_action_probs(action_probs),
            'decision_attribution': self._analyze_decision_attribution(
                quantum_outputs, action_probs, decisions
            ),
            'transformation_analysis': self._analyze_transformation(
                encoder_outputs, quantum_outputs, action_probs
            )
        }
        
        return results
    
    def _analyze_stage_outputs(self, outputs: np.ndarray, stage_name: str) -> Dict:
        """Analyze outputs from a single stage across all states."""
        return {
            'shape': list(outputs.shape),
            'mean_per_dim': outputs.mean(axis=0).tolist(),
            'std_per_dim': outputs.std(axis=0).tolist(),
            'overall_mean': float(outputs.mean()),
            'overall_std': float(outputs.std()),
            'variance_across_states': float(outputs.var(axis=0).mean()),
            'active_dimensions': int(np.sum(outputs.std(axis=0) > 0.01)),
            'total_dimensions': outputs.shape[1],
            'utilization_pct': float(np.sum(outputs.std(axis=0) > 0.01) / outputs.shape[1] * 100)
        }
    
    def _analyze_action_probs(self, action_probs: np.ndarray) -> Dict:
        """Analyze action probability distribution."""
        entropies = -np.sum(action_probs * np.log(action_probs + 1e-10), axis=1)
        confidences = np.max(action_probs, axis=1)
        hit_probs = action_probs[:, 1]
        
        return {
            'mean_entropy': float(entropies.mean()),
            'std_entropy': float(entropies.std()),
            'mean_confidence': float(confidences.mean()),
            'std_confidence': float(confidences.std()),
            'hit_rate': float((hit_probs > 0.5).mean() * 100),
            'stick_rate': float((hit_probs <= 0.5).mean() * 100),
            'high_confidence_pct': float((confidences > 0.8).mean() * 100),
            'low_confidence_pct': float((confidences < 0.6).mean() * 100)
        }
    
    def _analyze_decision_attribution(
        self, 
        quantum_outputs: np.ndarray, 
        action_probs: np.ndarray,
        decisions: List[int]
    ) -> Dict:
        """
        Analyze how much quantum output predicts final decision.
        
        Key question: Can we predict the action from quantum output alone?
        """
        decisions = np.array(decisions)
        
        # Correlation between quantum output and hit probability
        hit_probs = action_probs[:, 1]
        quantum_mean = quantum_outputs.mean(axis=1)
        correlation = np.corrcoef(quantum_mean, hit_probs)[0, 1]
        
        # Per-dimension correlation with decision
        dim_correlations = []
        for dim in range(quantum_outputs.shape[1]):
            corr = np.corrcoef(quantum_outputs[:, dim], hit_probs)[0, 1]
            dim_correlations.append(float(corr) if not np.isnan(corr) else 0.0)
        
        # Find most predictive quantum dimensions
        sorted_dims = np.argsort(np.abs(dim_correlations))[::-1]
        
        return {
            'quantum_action_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'per_dim_correlations': dim_correlations,
            'most_predictive_dims': sorted_dims[:3].tolist(),
            'strongest_correlation': float(max(np.abs(dim_correlations))),
            'interpretation': self._interpret_correlation(correlation)
        }
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation value."""
        if np.isnan(corr):
            return "Unable to compute correlation"
        abs_corr = abs(corr)
        if abs_corr > 0.7:
            return "Strong: Quantum output strongly predicts action"
        elif abs_corr > 0.4:
            return "Moderate: Quantum output has some predictive power"
        elif abs_corr > 0.2:
            return "Weak: Postprocessing significantly transforms quantum output"
        else:
            return "Very weak: Quantum output may not drive decisions"
    
    def _analyze_transformation(
        self,
        encoder_outputs: np.ndarray,
        quantum_outputs: np.ndarray,
        action_probs: np.ndarray
    ) -> Dict:
        """
        Analyze how much each stage transforms the signal.
        
        Measures information compression/expansion at each stage.
        """
        # Variance ratios (how much variance is preserved/created)
        encoder_var = encoder_outputs.var()
        quantum_var = quantum_outputs.var()
        action_var = action_probs.var()
        
        # Effective dimensionality (via PCA-like analysis)
        def effective_dim(data):
            centered = data - data.mean(axis=0)
            try:
                _, s, _ = np.linalg.svd(centered, full_matrices=False)
                s_normalized = s / s.sum()
                return float(np.exp(-np.sum(s_normalized * np.log(s_normalized + 1e-10))))
            except:
                return float(data.shape[1])
        
        return {
            'encoder_variance': float(encoder_var),
            'quantum_variance': float(quantum_var),
            'action_variance': float(action_var),
            'encoder_to_quantum_ratio': float(quantum_var / (encoder_var + 1e-10)),
            'quantum_to_action_ratio': float(action_var / (quantum_var + 1e-10)),
            'encoder_effective_dim': effective_dim(encoder_outputs),
            'quantum_effective_dim': effective_dim(quantum_outputs),
            'compression_summary': self._summarize_transformation(
                encoder_var, quantum_var, action_var
            )
        }
    
    def _summarize_transformation(self, enc_var, q_var, act_var) -> str:
        """Summarize the transformation pattern."""
        if q_var > enc_var * 1.5:
            quantum_effect = "Quantum EXPANDS information"
        elif q_var < enc_var * 0.5:
            quantum_effect = "Quantum COMPRESSES information"
        else:
            quantum_effect = "Quantum PRESERVES information scale"
        
        if act_var < q_var * 0.3:
            post_effect = "Postprocessing strongly compresses to binary decision"
        else:
            post_effect = "Postprocessing preserves quantum signal diversity"
        
        return f"{quantum_effect}; {post_effect}"
    
    def save_analysis(self, output_path: str, analysis: Optional[Dict] = None):
        """Save analysis results to JSON."""
        if analysis is None:
            analysis = self.analyze_all_states()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Analysis saved to {output_path}")
        return analysis


def analyze_hybrid_intermediates(checkpoint_path: str) -> Dict:
    """
    Convenience function to analyze a hybrid checkpoint.
    
    Uses network loader to extract config from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Analysis dictionary
    """
    from ..networks.loader import load_policy_network, NetworkLoadError
    
    # Load model using network loader
    try:
        policy_net = load_policy_network(checkpoint_path, strict=False)
    except NetworkLoadError as e:
        return {'error': str(e)}
    
    if not hasattr(policy_net, 'quantum_circuit'):
        return {'error': 'Model is not a hybrid network'}
    
    # Analyze
    analyzer = IntermediateOutputAnalyzer(policy_net)
    return analyzer.analyze_all_states()
