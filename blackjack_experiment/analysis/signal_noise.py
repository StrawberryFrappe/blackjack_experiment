"""Signal vs noise analysis for hybrid quantum networks.

Tests the hypothesis: Is the quantum circuit making meaningful corrections
to the encoder output, or is it just adding noise?

Key analyses:
1. Linearity test: Is Q approx linear(Encoder)?
2. Correction patterns: Are Q corrections systematic or random noise?
3. State-conditional influence: Where does Q change decisions?
4. Signal vs noise: How much of Q's variance is structured vs random?

Note: Previously named quantum_contribution.py. Renamed to signal_noise.py
to distinguish from gradient_flow.py which analyzes gradient flow during training.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


from .utils import generate_all_blackjack_states


class SignalNoiseAnalyzer:
    """
    Analyzes whether the quantum circuit output is signal (structured) or noise.
    
    Determines if quantum makes meaningful state-dependent corrections
    or just adds random noise to the encoder output.
    
    Previously named QuantumContributionAnalyzer in quantum_contribution.py.
    """
    
    def __init__(self, policy_net):
        """
        Initialize with a hybrid policy network.
        
        Args:
            policy_net: Network with forward_with_intermediates() method
        """
        self.policy_net = policy_net
        self.policy_net.eval()
        
        if not hasattr(policy_net, 'forward_with_intermediates'):
            raise ValueError("Network must have forward_with_intermediates() method")
        
        # Cache all state outputs for analysis
        self._cache = None
    
    def _generate_all_states(self) -> List[Tuple[int, int, int]]:
        """Generate all valid Blackjack states."""
        return generate_all_blackjack_states()
    
    def _collect_outputs(self) -> Dict[str, np.ndarray]:
        """Collect all intermediate outputs for all states."""
        if self._cache is not None:
            return self._cache
        
        states = self._generate_all_states()
        
        encoder_outputs = []
        quantum_outputs = []
        action_probs_list = []
        state_features = []  # player_sum, dealer_card, usable_ace
        
        for state in states:
            with torch.no_grad():
                intermediates = self.policy_net.forward_with_intermediates(state)
            
            encoder_outputs.append(intermediates['feature_encoder_output'].squeeze().numpy())
            quantum_outputs.append(intermediates['quantum_output'].squeeze().numpy())
            action_probs_list.append(intermediates['action_probs'].squeeze().numpy())
            state_features.append(list(state))
        
        self._cache = {
            'states': np.array(state_features),
            'encoder': np.array(encoder_outputs),
            'quantum': np.array(quantum_outputs),
            'action_probs': np.array(action_probs_list),
            'hit_probs': np.array(action_probs_list)[:, 1],
            'decisions': (np.array(action_probs_list)[:, 1] > 0.5).astype(int)
        }
        return self._cache
    
    def analyze_linearity(self) -> Dict:
        """
        Test if quantum output is approximately a linear function of encoder output.
        
        High R^2 = quantum is near-linear transformation of encoder (not adding much)
        Low R^2 = quantum is doing nonlinear processing (potentially meaningful)
        """
        data = self._collect_outputs()
        encoder = data['encoder']
        quantum = data['quantum']
        
        # Fit linear regression: Q = W @ Encoder + b
        reg = LinearRegression()
        reg.fit(encoder, quantum)
        quantum_predicted = reg.predict(encoder)
        
        # Overall R^2 score
        r2_overall = r2_score(quantum, quantum_predicted)
        
        # Per-dimension R^2 scores
        r2_per_dim = []
        for dim in range(quantum.shape[1]):
            r2 = r2_score(quantum[:, dim], quantum_predicted[:, dim])
            r2_per_dim.append(float(r2))
        
        # Residuals (what linear model can't explain = nonlinear quantum contribution)
        residuals = quantum - quantum_predicted
        residual_variance = residuals.var()
        total_variance = quantum.var()
        nonlinear_fraction = residual_variance / (total_variance + 1e-10)
        
        return {
            'r2_overall': float(r2_overall),
            'r2_per_dim': r2_per_dim,
            'mean_r2_per_dim': float(np.mean(r2_per_dim)),
            'nonlinear_fraction': float(nonlinear_fraction),
            'residual_variance': float(residual_variance),
            'interpretation': self._interpret_linearity(r2_overall, nonlinear_fraction)
        }
    
    def _interpret_linearity(self, r2: float, nonlinear_frac: float) -> str:
        """Interpret linearity results."""
        if r2 > 0.9:
            return "NEAR-LINEAR: Quantum is mostly a linear transform of encoder (minimal added value)"
        elif r2 > 0.7:
            return "MODERATELY LINEAR: Quantum adds some nonlinear structure"
        elif r2 > 0.5:
            return "PARTIALLY NONLINEAR: Quantum does meaningful nonlinear processing"
        else:
            return "HIGHLY NONLINEAR: Quantum transformation is complex (could be useful OR noise)"
    
    def analyze_correction_patterns(self) -> Dict:
        """
        Analyze whether quantum makes systematic corrections or random noise.
        
        Systematic corrections = Q output correlates with game state features
        Random noise = Q output is uncorrelated with game state
        """
        data = self._collect_outputs()
        states = data['states']  # [player_sum, dealer_card, usable_ace]
        quantum = data['quantum']
        encoder = data['encoder']
        
        # Compute "residual" quantum contribution (what encoder alone doesn't predict)
        reg = LinearRegression()
        reg.fit(encoder, quantum)
        quantum_residual = quantum - reg.predict(encoder)
        
        # Correlation of residuals with game state features
        player_sum = states[:, 0]
        dealer_card = states[:, 1]
        usable_ace = states[:, 2]
        
        # Mean residual per dimension
        residual_mean = quantum_residual.mean(axis=1)
        
        # Correlations
        corr_player = np.corrcoef(residual_mean, player_sum)[0, 1]
        corr_dealer = np.corrcoef(residual_mean, dealer_card)[0, 1]
        corr_ace = np.corrcoef(residual_mean, usable_ace)[0, 1]
        
        # Per-dimension correlations with player sum (most important)
        dim_corr_player = []
        for dim in range(quantum_residual.shape[1]):
            c = np.corrcoef(quantum_residual[:, dim], player_sum)[0, 1]
            dim_corr_player.append(float(c) if not np.isnan(c) else 0.0)
        
        # Analyze residual structure by game state groups
        low_sum_residuals = quantum_residual[player_sum <= 11].mean(axis=0)
        mid_sum_residuals = quantum_residual[(player_sum > 11) & (player_sum <= 16)].mean(axis=0)
        high_sum_residuals = quantum_residual[player_sum > 16].mean(axis=0)
        
        # Residual variance (should be low if corrections are systematic)
        within_group_var = (
            quantum_residual[player_sum <= 11].var() +
            quantum_residual[(player_sum > 11) & (player_sum <= 16)].var() +
            quantum_residual[player_sum > 16].var()
        ) / 3
        total_var = quantum_residual.var()
        
        systematic_fraction = 1 - (within_group_var / (total_var + 1e-10))
        
        return {
            'correlation_with_player_sum': float(corr_player) if not np.isnan(corr_player) else 0.0,
            'correlation_with_dealer_card': float(corr_dealer) if not np.isnan(corr_dealer) else 0.0,
            'correlation_with_usable_ace': float(corr_ace) if not np.isnan(corr_ace) else 0.0,
            'per_dim_correlation_player_sum': dim_corr_player,
            'low_sum_mean_residual': low_sum_residuals.tolist(),
            'mid_sum_mean_residual': mid_sum_residuals.tolist(),
            'high_sum_mean_residual': high_sum_residuals.tolist(),
            'systematic_fraction': float(systematic_fraction),
            'interpretation': self._interpret_corrections(systematic_fraction, corr_player)
        }
    
    def _interpret_corrections(self, systematic_frac: float, player_corr: float) -> str:
        """Interpret correction pattern results."""
        if systematic_frac > 0.3 and abs(player_corr) > 0.2:
            return "SYSTEMATIC: Quantum makes state-dependent corrections (meaningful)"
        elif systematic_frac > 0.15:
            return "PARTIALLY SYSTEMATIC: Some structure in quantum corrections"
        else:
            return "NOISE-LIKE: Quantum residuals appear random (noise hypothesis supported)"
    
    def analyze_decision_influence(self) -> Dict:
        """
        Analyze where quantum output actually changes the decision.
        
        Compare: decision from encoder bypass vs decision with quantum
        """
        data = self._collect_outputs()
        states = data['states']
        encoder = data['encoder']
        quantum = data['quantum']
        hit_probs = data['hit_probs']
        
        # Get postprocessing layer
        postprocessing = self.policy_net.postprocessing
        softmax = self.policy_net.softmax
        
        # Compute "encoder bypass" decisions (what postprocessing does with zero quantum)
        with torch.no_grad():
            zero_input = torch.zeros(1, quantum.shape[1])
            encoder_bypass_logits = postprocessing(zero_input)
            encoder_bypass_probs = softmax(encoder_bypass_logits).numpy().squeeze()
        
        encoder_bypass_hit = encoder_bypass_probs[1]
        
        # Compute decisions from quantum output alone (no encoder context)
        quantum_only_decisions = []
        with torch.no_grad():
            for q in quantum:
                q_tensor = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
                logits = postprocessing(q_tensor)
                probs = softmax(logits).numpy().squeeze()
                quantum_only_decisions.append(probs[1])
        
        quantum_only_decisions = np.array(quantum_only_decisions)
        
        # Compare with actual decisions
        decision_diff = hit_probs - quantum_only_decisions
        
        # Analyze by game state region
        player_sum = states[:, 0]
        
        # Where does quantum change the decision?
        decision_changes = np.abs(hit_probs - encoder_bypass_hit) > 0.1
        
        results_by_region = {}
        for region_name, mask in [
            ('low_sum_4-11', player_sum <= 11),
            ('mid_sum_12-16', (player_sum > 11) & (player_sum <= 16)),
            ('high_sum_17-21', player_sum > 16)
        ]:
            region_quantum = quantum_only_decisions[mask]
            region_actual = hit_probs[mask]
            region_diff = region_actual - region_quantum
            
            results_by_region[region_name] = {
                'mean_quantum_hit_prob': float(region_quantum.mean()),
                'mean_actual_hit_prob': float(region_actual.mean()),
                'mean_difference': float(region_diff.mean()),
                'std_difference': float(region_diff.std()),
                'decisions_aligned_pct': float((np.abs(region_diff) < 0.1).mean() * 100)
            }
        
        return {
            'encoder_bypass_hit_prob': float(encoder_bypass_hit),
            'mean_quantum_hit_prob': float(quantum_only_decisions.mean()),
            'mean_actual_hit_prob': float(hit_probs.mean()),
            'correlation_quantum_actual': float(np.corrcoef(quantum_only_decisions, hit_probs)[0, 1]),
            'by_region': results_by_region,
            'interpretation': self._interpret_influence(
                results_by_region, np.corrcoef(quantum_only_decisions, hit_probs)[0, 1]
            )
        }
    
    def _interpret_influence(self, by_region: Dict, correlation: float) -> str:
        """Interpret decision influence results."""
        low_aligned = by_region['low_sum_4-11']['decisions_aligned_pct']
        mid_aligned = by_region['mid_sum_12-16']['decisions_aligned_pct']
        high_aligned = by_region['high_sum_17-21']['decisions_aligned_pct']
        
        insights = []
        
        if correlation > 0.7:
            insights.append("Quantum strongly determines final decision")
        elif correlation > 0.4:
            insights.append("Quantum moderately influences decision")
        else:
            insights.append("Postprocessing significantly transforms quantum signal")
        
        if low_aligned < 50:
            insights.append("Low sum: postprocessing overrides quantum")
        if high_aligned < 50:
            insights.append("High sum: postprocessing overrides quantum")
        if mid_aligned > 70:
            insights.append("Mid sum: quantum drives decision")
        
        return "; ".join(insights)
    
    def analyze_signal_vs_noise(self) -> Dict:
        """
        Decompose quantum output into signal (structured) vs noise (random).
        
        Uses: variance explained by game state features
        """
        data = self._collect_outputs()
        states = data['states']
        quantum = data['quantum']
        
        # Fit regression: Q = f(player_sum, dealer_card, usable_ace)
        reg = LinearRegression()
        reg.fit(states, quantum)
        quantum_predicted = reg.predict(states)
        
        # Signal = variance explained by state
        signal_variance = quantum_predicted.var()
        total_variance = quantum.var()
        noise_variance = (quantum - quantum_predicted).var()
        
        signal_ratio = signal_variance / (total_variance + 1e-10)
        noise_ratio = noise_variance / (total_variance + 1e-10)
        
        # Per-dimension analysis
        signal_per_dim = []
        for dim in range(quantum.shape[1]):
            reg_dim = LinearRegression()
            reg_dim.fit(states, quantum[:, dim])
            r2 = reg_dim.score(states, quantum[:, dim])
            signal_per_dim.append(float(r2))
        
        return {
            'signal_ratio': float(signal_ratio),
            'noise_ratio': float(noise_ratio),
            'signal_variance': float(signal_variance),
            'noise_variance': float(noise_variance),
            'total_variance': float(total_variance),
            'signal_per_dim': signal_per_dim,
            'mean_signal_per_dim': float(np.mean(signal_per_dim)),
            'high_signal_dims': int(np.sum(np.array(signal_per_dim) > 0.3)),
            'interpretation': self._interpret_signal_noise(signal_ratio)
        }
    
    def _interpret_signal_noise(self, signal_ratio: float) -> str:
        """Interpret signal vs noise results."""
        if signal_ratio > 0.5:
            return "HIGH SIGNAL: Quantum output is mostly structured by game state"
        elif signal_ratio > 0.2:
            return "MODERATE SIGNAL: Mix of state-dependent signal and noise"
        else:
            return "LOW SIGNAL: Quantum output appears mostly noise-like"
    
    def analyze_encoder_quantum_alignment(self) -> Dict:
        """
        Analyze how well encoder and quantum outputs are aligned.
        
        Tests: Is quantum just a rotated/scaled version of encoder?
        Or does it add orthogonal information?
        """
        data = self._collect_outputs()
        encoder = data['encoder']
        quantum = data['quantum']
        
        # Canonical correlation analysis (simplified)
        # Center the data
        encoder_centered = encoder - encoder.mean(axis=0)
        quantum_centered = quantum - quantum.mean(axis=0)
        
        # Cross-covariance matrix
        cross_cov = encoder_centered.T @ quantum_centered / len(encoder)
        
        # SVD to find principal directions of correlation
        U, s, Vt = np.linalg.svd(cross_cov)
        
        # Canonical correlations (normalized singular values)
        encoder_std = np.sqrt((encoder_centered ** 2).sum(axis=0).mean())
        quantum_std = np.sqrt((quantum_centered ** 2).sum(axis=0).mean())
        canonical_corrs = s / (encoder_std * quantum_std + 1e-10)
        
        # How much encoder variance is preserved in quantum?
        # Project encoder onto principal encoder directions
        encoder_projected = encoder_centered @ U[:, :min(5, len(s))]
        quantum_projected = quantum_centered @ Vt[:min(5, len(s)), :].T
        
        alignment_scores = []
        for i in range(min(5, len(s))):
            corr = np.corrcoef(encoder_projected[:, i], quantum_projected[:, i])[0, 1]
            alignment_scores.append(float(corr) if not np.isnan(corr) else 0.0)
        
        return {
            'canonical_correlations': canonical_corrs[:5].tolist(),
            'top_singular_value': float(s[0]),
            'explained_by_top_5': float(s[:5].sum() / s.sum()) if len(s) > 0 else 0.0,
            'alignment_scores': alignment_scores,
            'mean_alignment': float(np.mean(alignment_scores)),
            'interpretation': self._interpret_alignment(np.mean(alignment_scores), canonical_corrs[0])
        }
    
    def _interpret_alignment(self, mean_align: float, top_corr: float) -> str:
        """Interpret encoder-quantum alignment."""
        if mean_align > 0.7:
            return "HIGHLY ALIGNED: Quantum preserves encoder structure (continuous forward pass)"
        elif mean_align > 0.4:
            return "MODERATELY ALIGNED: Quantum partially preserves encoder info"
        else:
            return "WEAKLY ALIGNED: Quantum significantly transforms encoder representation"
    
    def run_full_analysis(self) -> Dict:
        """Run all analyses and return comprehensive results."""
        return {
            'linearity': self.analyze_linearity(),
            'correction_patterns': self.analyze_correction_patterns(),
            'decision_influence': self.analyze_decision_influence(),
            'signal_vs_noise': self.analyze_signal_vs_noise(),
            'encoder_quantum_alignment': self.analyze_encoder_quantum_alignment(),
            'summary': self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict:
        """Generate summary of all analyses."""
        linearity = self.analyze_linearity()
        corrections = self.analyze_correction_patterns()
        signal_noise = self.analyze_signal_vs_noise()
        alignment = self.analyze_encoder_quantum_alignment()
        
        # Verdict: Continuous forward pass OR Noise blur?
        evidence_for_continuous = 0
        evidence_for_noise = 0
        
        if linearity['r2_overall'] > 0.5:
            evidence_for_continuous += 1
        else:
            evidence_for_noise += 1
        
        if corrections['systematic_fraction'] > 0.2:
            evidence_for_continuous += 1
        else:
            evidence_for_noise += 1
        
        if signal_noise['signal_ratio'] > 0.3:
            evidence_for_continuous += 1
        else:
            evidence_for_noise += 1
        
        if alignment['mean_alignment'] > 0.4:
            evidence_for_continuous += 1
        else:
            evidence_for_noise += 1
        
        if evidence_for_continuous > evidence_for_noise:
            verdict = "CONTINUOUS FORWARD PASS: Quantum acts as structured continuation of encoder"
        elif evidence_for_noise > evidence_for_continuous:
            verdict = "NOISE BLUR: Quantum appears to add noise rather than meaningful corrections"
        else:
            verdict = "INCONCLUSIVE: Mixed evidence for both hypotheses"
        
        return {
            'evidence_for_continuous_forward': evidence_for_continuous,
            'evidence_for_noise_blur': evidence_for_noise,
            'verdict': verdict,
            'key_metrics': {
                'linearity_r2': linearity['r2_overall'],
                'systematic_corrections': corrections['systematic_fraction'],
                'signal_ratio': signal_noise['signal_ratio'],
                'encoder_alignment': alignment['mean_alignment']
            }
        }
    
    def plot_analysis(self, save_path: Optional[str] = None):
        """Create visualization of quantum contribution analysis."""
        data = self._collect_outputs()
        linearity = self.analyze_linearity()
        corrections = self.analyze_correction_patterns()
        signal_noise = self.analyze_signal_vs_noise()
        influence = self.analyze_decision_influence()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Quantum Contribution Analysis', fontsize=14, fontweight='bold')
        
        # 1. Linearity R^2 per dimension
        ax = axes[0, 0]
        dims = range(len(linearity['r2_per_dim']))
        ax.bar(dims, linearity['r2_per_dim'], color='steelblue', alpha=0.7)
        ax.axhline(y=linearity['r2_overall'], color='red', linestyle='--', 
                   label=f'Overall R^2={linearity["r2_overall"]:.3f}')
        ax.set_xlabel('Quantum Dimension')
        ax.set_ylabel('R^2 (Linearity)')
        ax.set_title('Encoder→Quantum Linearity')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # 2. Signal vs Noise per dimension
        ax = axes[0, 1]
        ax.bar(dims, signal_noise['signal_per_dim'], color='green', alpha=0.7, label='Signal')
        ax.bar(dims, [1 - s for s in signal_noise['signal_per_dim']], 
               bottom=signal_noise['signal_per_dim'], color='red', alpha=0.5, label='Noise')
        ax.set_xlabel('Quantum Dimension')
        ax.set_ylabel('Fraction')
        ax.set_title(f'Signal vs Noise (Signal={signal_noise["signal_ratio"]:.2f})')
        ax.legend()
        
        # 3. Correction patterns by player sum region
        ax = axes[0, 2]
        regions = ['Low (4-11)', 'Mid (12-16)', 'High (17-21)']
        low_res = np.array(corrections['low_sum_mean_residual'])
        mid_res = np.array(corrections['mid_sum_mean_residual'])
        high_res = np.array(corrections['high_sum_mean_residual'])
        
        x = np.arange(len(dims))
        width = 0.25
        ax.bar(x - width, low_res, width, label='Low sum', alpha=0.7)
        ax.bar(x, mid_res, width, label='Mid sum', alpha=0.7)
        ax.bar(x + width, high_res, width, label='High sum', alpha=0.7)
        ax.set_xlabel('Quantum Dimension')
        ax.set_ylabel('Mean Residual')
        ax.set_title('Quantum Corrections by Game State')
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 4. Decision influence by region
        ax = axes[1, 0]
        regions = list(influence['by_region'].keys())
        quantum_probs = [influence['by_region'][r]['mean_quantum_hit_prob'] for r in regions]
        actual_probs = [influence['by_region'][r]['mean_actual_hit_prob'] for r in regions]
        
        x = np.arange(len(regions))
        width = 0.35
        ax.bar(x - width/2, quantum_probs, width, label='Quantum-only', alpha=0.7)
        ax.bar(x + width/2, actual_probs, width, label='Actual', alpha=0.7)
        ax.set_xlabel('Game State Region')
        ax.set_ylabel('Mean Hit Probability')
        ax.set_title('Quantum vs Actual Decisions')
        ax.set_xticks(x)
        ax.set_xticklabels(['Low\n(4-11)', 'Mid\n(12-16)', 'High\n(17-21)'])
        ax.legend()
        
        # 5. Summary metrics
        ax = axes[1, 1]
        summary = self._generate_summary()
        metrics = summary['key_metrics']
        
        metric_names = ['Linearity\nR^2', 'Systematic\nCorrections', 'Signal\nRatio', 'Encoder\nAlignment']
        metric_values = [metrics['linearity_r2'], metrics['systematic_corrections'],
                        metrics['signal_ratio'], metrics['encoder_alignment']]
        
        colors = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in metric_values]
        ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax.set_ylabel('Value')
        ax.set_title('Key Metrics Summary')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # 6. Verdict
        ax = axes[1, 2]
        ax.axis('off')
        verdict_text = f"""
        VERDICT
        -------
        Evidence for Continuous Forward Pass: {summary['evidence_for_continuous_forward']}/4
        Evidence for Noise Blur: {summary['evidence_for_noise_blur']}/4
        
        {summary['verdict']}
        
        Key Findings:
        • Linearity R^2: {metrics['linearity_r2']:.3f}
        • Systematic corrections: {metrics['systematic_corrections']:.3f}
        • Signal ratio: {metrics['signal_ratio']:.3f}
        • Encoder alignment: {metrics['encoder_alignment']:.3f}
        """
        ax.text(0.1, 0.5, verdict_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved analysis plot to {save_path}")
        
        plt.close()
        return fig
    
    def save_analysis(self, output_path: str):
        """Save full analysis to JSON."""
        analysis = self.run_full_analysis()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Saved analysis to {output_path}")
        return analysis


def analyze_checkpoint(checkpoint_path: str, output_dir: Optional[str] = None) -> Dict:
    """
    Analyze a hybrid checkpoint for quantum contribution.
    
    Uses network loader to extract config from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        output_dir: Optional directory to save results
        
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
    analyzer = SignalNoiseAnalyzer(policy_net)
    analysis = analyzer.run_full_analysis()
    
    # Save if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analyzer.save_analysis(output_dir / 'quantum_contribution_analysis.json')
        analyzer.plot_analysis(output_dir / 'quantum_contribution_analysis.png')
    
    return analysis


# Backward compatibility alias
QuantumContributionAnalyzer = SignalNoiseAnalyzer
