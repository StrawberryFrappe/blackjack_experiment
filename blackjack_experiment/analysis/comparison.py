"""
Comparison analysis for multiple Blackjack models.

This module provides tools to compare behavior and strategy between different models:
1. Decision agreement analysis
2. Strategy difference heatmaps
3. Input sensitivity comparison
4. Win rate vs strategy correlation
5. Behavioral profile comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import json

from .strategy import DecisionAnalyzer, get_basic_strategy_action


class ComparisonAnalyzer:
    """
    Compares behavior and strategy between multiple Blackjack models.
    
    Provides:
    - Side-by-side decision heatmaps
    - Decision agreement analysis
    - Strategy profile comparison
    - Input sensitivity comparison
    - Behavioral difference analysis
    """
    
    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        model_names: Optional[List[str]] = None
    ):
        """
        Initialize the comparison analyzer.
        
        Args:
            models: Dictionary of {name: policy_net}
            model_names: Optional list of model names for display
        """
        self.models = models
        self.model_names = model_names or list(models.keys())
        
        # Create individual analyzers
        self.analyzers = {
            name: DecisionAnalyzer(net)
            for name, net in models.items()
        }
        
        # Pre-compute decision maps
        self.decision_maps = {}
        for name, analyzer in self.analyzers.items():
            self.decision_maps[name] = {
                'no_ace': analyzer.create_decision_map(0),
                'usable_ace': analyzer.create_decision_map(1)
            }
    
    def compute_agreement_matrix(self, usable_ace: int = 0) -> Dict:
        """
        Compute pairwise decision agreement between models.
        
        Returns:
            Dictionary with agreement percentages and detailed breakdown
        """
        names = list(self.models.keys())
        n_models = len(names)
        
        # Agreement matrix
        agreement_matrix = np.zeros((n_models, n_models))
        
        # Detailed agreement per state
        ace_key = 'usable_ace' if usable_ace else 'no_ace'
        
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i == j:
                    agreement_matrix[i, j] = 100.0
                else:
                    actions1 = self.decision_maps[name1][ace_key]['actions']
                    actions2 = self.decision_maps[name2][ace_key]['actions']
                    agreement = np.mean(actions1 == actions2) * 100
                    agreement_matrix[i, j] = agreement
        
        return {
            'agreement_matrix': agreement_matrix,
            'model_names': names,
            'usable_ace': usable_ace
        }
    
    def compute_behavioral_profiles(self) -> Dict:
        """
        Compute behavioral profiles for all models.
        
        Returns metrics like:
        - Overall aggression (hit rate)
        - Dealer sensitivity
        - Ace sensitivity
        - Basic strategy adherence
        """
        profiles = {}
        
        for name, analyzer in self.analyzers.items():
            # Compute metrics
            no_ace_map = self.decision_maps[name]['no_ace']
            usable_ace_map = self.decision_maps[name]['usable_ace']
            
            # Overall aggression (average hit probability)
            overall_hit_rate = np.mean(no_ace_map['hit_probs'])
            
            # Sensitivity analysis
            sensitivity = analyzer.analyze_input_sensitivity()
            
            # Basic strategy accuracy
            cm_data = analyzer.generate_confusion_matrix()
            
            # Dealer sensitivity: variance in hit prob across dealer cards
            dealer_sensitivity = sensitivity['dealer_card']['mean_variance']
            
            # Ace sensitivity: difference in behavior with/without ace
            ace_sensitivity = sensitivity['usable_ace']['mean_diff']
            
            # Hit rate by player sum ranges
            hit_rate_low = np.mean(no_ace_map['hit_probs'][:7, :])  # 4-10
            hit_rate_mid = np.mean(no_ace_map['hit_probs'][7:13, :])  # 11-16
            hit_rate_high = np.mean(no_ace_map['hit_probs'][13:, :])  # 17-21
            
            profiles[name] = {
                'overall_hit_rate': overall_hit_rate * 100,
                'dealer_sensitivity': dealer_sensitivity,
                'ace_sensitivity': ace_sensitivity,
                'basic_strategy_accuracy': cm_data['accuracy'],
                'hit_rate_low_sum': hit_rate_low * 100,
                'hit_rate_mid_sum': hit_rate_mid * 100,
                'hit_rate_high_sum': hit_rate_high * 100,
                'confusion_matrix': cm_data
            }
        
        return profiles
    
    def compute_decision_differences(self, usable_ace: int = 0) -> Dict:
        """
        Compute where models disagree and identify patterns.
        """
        names = list(self.models.keys())
        if len(names) < 2:
            return {'error': 'Need at least 2 models for comparison'}
        
        ace_key = 'usable_ace' if usable_ace else 'no_ace'
        
        # For simplicity, compare first two models
        name1, name2 = names[0], names[1]
        actions1 = self.decision_maps[name1][ace_key]['actions']
        actions2 = self.decision_maps[name2][ace_key]['actions']
        
        # Difference map: 0 = agree, 1 = model1 hits & model2 stands, -1 = opposite
        diff_map = actions1 - actions2
        
        # Find disagreement states
        disagreements = []
        for i, player_sum in enumerate(range(4, 22)):
            for j, dealer_card in enumerate(range(1, 11)):
                if diff_map[i, j] != 0:
                    disagreements.append({
                        'state': (player_sum, dealer_card, usable_ace),
                        f'{name1}_action': 'Hit' if actions1[i, j] == 1 else 'Stand',
                        f'{name2}_action': 'Hit' if actions2[i, j] == 1 else 'Stand',
                        'optimal_action': 'Hit' if get_basic_strategy_action(player_sum, dealer_card, usable_ace) == 1 else 'Stand'
                    })
        
        return {
            'difference_map': diff_map,
            'disagreement_count': len(disagreements),
            'disagreement_rate': len(disagreements) / (18 * 10) * 100,
            'disagreements': disagreements,
            'model_names': (name1, name2)
        }
    
    def plot_comparison_heatmaps(
        self,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot side-by-side decision heatmaps for all models.
        """
        from matplotlib.colors import LinearSegmentedColormap
        
        names = list(self.models.keys())
        n_models = len(names)
        
        # Custom colormap: Red=Stand, Green=Hit
        colors = ['#e74c3c', '#f39c12', '#2ecc71']
        action_cmap = LinearSegmentedColormap.from_list('StandHit', colors, N=256)
        
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for col, name in enumerate(names):
            for row, usable_ace in enumerate([0, 1]):
                ace_key = 'usable_ace' if usable_ace else 'no_ace'
                decision_map = self.decision_maps[name][ace_key]
                
                ax = axes[row, col]
                im = ax.imshow(
                    decision_map['actions'],
                    aspect='auto',
                    cmap=action_cmap,
                    origin='lower',
                    extent=[0.5, 10.5, 3.5, 21.5],
                    vmin=0, vmax=1
                )
                ax.set_xlabel('Dealer Showing')
                ax.set_ylabel('Player Sum')
                ace_str = 'With Ace' if usable_ace else 'No Ace'
                ax.set_title(f'{name}\n({ace_str})')
                ax.set_xticks(range(1, 11))
                ax.set_yticks(range(4, 22, 2))
                
                if col == n_models - 1:
                    cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
                    cbar.ax.set_yticklabels(['Stand', 'Hit'])
        
        plt.suptitle('Decision Comparison (Green=Hit, Red=Stand)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Comparison heatmaps saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_behavioral_comparison(
        self,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot behavioral profile comparison across models.
        """
        profiles = self.compute_behavioral_profiles()
        names = list(profiles.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Plot 1: Basic Strategy Accuracy
        ax = axes[0, 0]
        accuracies = [profiles[n]['basic_strategy_accuracy'] for n in names]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'][:len(names)]
        bars = ax.bar(names, accuracies, color=colors, edgecolor='black', alpha=0.8)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Basic Strategy Adherence')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Hit Rate by Sum Range
        ax = axes[0, 1]
        x = np.arange(len(names))
        width = 0.25
        
        low_rates = [profiles[n]['hit_rate_low_sum'] for n in names]
        mid_rates = [profiles[n]['hit_rate_mid_sum'] for n in names]
        high_rates = [profiles[n]['hit_rate_high_sum'] for n in names]
        
        ax.bar(x - width, low_rates, width, label='Low (4-10)', color='#2ecc71', alpha=0.8)
        ax.bar(x, mid_rates, width, label='Mid (11-16)', color='#f39c12', alpha=0.8)
        ax.bar(x + width, high_rates, width, label='High (17-21)', color='#e74c3c', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title('Hit Rate by Player Sum Range')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Input Sensitivity Comparison
        ax = axes[0, 2]
        dealer_sens = [profiles[n]['dealer_sensitivity'] for n in names]
        ace_sens = [profiles[n]['ace_sensitivity'] for n in names]
        
        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width/2, dealer_sens, width, label='Dealer Sensitivity', color='#3498db')
        ax.bar(x + width/2, ace_sens, width, label='Ace Sensitivity', color='#9b59b6')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel('Sensitivity Score')
        ax.set_title('Input Sensitivity')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Agreement Matrix
        ax = axes[1, 0]
        agreement_data = self.compute_agreement_matrix(0)
        im = ax.imshow(agreement_data['agreement_matrix'], cmap='RdYlGn', vmin=50, vmax=100)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticklabels(names)
        ax.set_title('Decision Agreement (%)\n(No Usable Ace)')
        # Add text annotations
        for i in range(len(names)):
            for j in range(len(names)):
                val = agreement_data['agreement_matrix'][i, j]
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontweight='bold')
        plt.colorbar(im, ax=ax)
        
        # Plot 5: Disagreement Analysis
        ax = axes[1, 1]
        if len(names) >= 2:
            diff_data = self.compute_decision_differences(0)
            diff_map = diff_data['difference_map']
            
            # Custom colormap: -1 = blue, 0 = white, 1 = red
            from matplotlib.colors import LinearSegmentedColormap
            diff_cmap = LinearSegmentedColormap.from_list('diff', ['#3498db', 'white', '#e74c3c'])
            
            im = ax.imshow(
                diff_map,
                aspect='auto',
                cmap=diff_cmap,
                origin='lower',
                extent=[0.5, 10.5, 3.5, 21.5],
                vmin=-1, vmax=1
            )
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ax.set_title(f'Decision Differences\n({names[0]} vs {names[1]})\nDisagreement: {diff_data["disagreement_rate"]:.1f}%')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
            cbar.ax.set_yticklabels([f'{names[0]} hits', 'Agree', f'{names[1]} hits'])
        else:
            ax.text(0.5, 0.5, 'Need 2+ models', ha='center', va='center', transform=ax.transAxes)
        
        # Plot 6: Overall Profile Radar-like comparison
        ax = axes[1, 2]
        metrics = ['Accuracy', 'Hit Low', 'Hit Mid', 'Hit High', 'Dealer Sens', 'Ace Sens']
        
        for i, name in enumerate(names):
            p = profiles[name]
            values = [
                p['basic_strategy_accuracy'] / 100,
                p['hit_rate_low_sum'] / 100,
                p['hit_rate_mid_sum'] / 100,
                p['hit_rate_high_sum'] / 100,
                min(p['dealer_sensitivity'] * 10, 1),  # Scale for visibility
                min(p['ace_sensitivity'] * 5, 1)
            ]
            ax.plot(metrics, values, marker='o', linewidth=2, label=name, alpha=0.7)
        
        ax.set_ylim([0, 1])
        ax.set_ylabel('Normalized Score')
        ax.set_title('Behavioral Profile Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.suptitle('Model Behavior Comparison Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Behavioral comparison saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, profiles
    
    def generate_comparison_report(
        self,
        output_dir: str,
        show: bool = False
    ) -> Dict:
        """
        Generate a complete comparison report.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("COMPARISON ANALYSIS REPORT")
        print("="*70)
        
        # Generate all plots
        self.plot_comparison_heatmaps(
            save_path=str(output_path / 'comparison_heatmaps.png'),
            show=show
        )
        
        _, profiles = self.plot_behavioral_comparison(
            save_path=str(output_path / 'behavioral_comparison.png'),
            show=show
        )
        
        # Compute summary statistics
        agreement_no_ace = self.compute_agreement_matrix(0)
        agreement_with_ace = self.compute_agreement_matrix(1)
        
        diff_analysis = self.compute_decision_differences(0)
        
        summary = {
            'models_compared': self.model_names,
            'behavioral_profiles': {},
            'agreement_analysis': {
                'no_ace': agreement_no_ace['agreement_matrix'].tolist(),
                'with_ace': agreement_with_ace['agreement_matrix'].tolist()
            },
            'disagreement_analysis': {
                'count': diff_analysis.get('disagreement_count', 0),
                'rate': diff_analysis.get('disagreement_rate', 0)
            }
        }
        
        # Add simplified profiles
        for name, profile in profiles.items():
            summary['behavioral_profiles'][name] = {
                'basic_strategy_accuracy': profile['basic_strategy_accuracy'],
                'overall_hit_rate': profile['overall_hit_rate'],
                'dealer_sensitivity': profile['dealer_sensitivity'],
                'ace_sensitivity': profile['ace_sensitivity']
            }
        
        # Save summary
        with open(output_path / 'comparison_analysis.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[DATA] Comparison report saved to: {output_path}")
        print("="*70)
        
        return summary


def compare_models_from_checkpoints(
    checkpoint_paths: Dict[str, str],
    model_factory_fn: Callable,
    output_dir: str,
    show: bool = False
) -> Dict:
    """
    Compare models loaded from checkpoint files.
    
    Args:
        checkpoint_paths: Dictionary of {name: checkpoint_path}
        model_factory_fn: Function to create model instances
        output_dir: Directory to save comparison results
        show: Whether to display plots
    
    Returns:
        Comparison summary dictionary
    """
    # Load models
    models = {}
    for name, path in checkpoint_paths.items():
        model = model_factory_fn()
        if isinstance(model, tuple):
            policy_net = model[0]
        else:
            policy_net = model
        
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'policy_net' in checkpoint:
            policy_net.load_state_dict(checkpoint['policy_net'])
        elif 'policy_state_dict' in checkpoint:
            policy_net.load_state_dict(checkpoint['policy_state_dict'])
        else:
            policy_net.load_state_dict(checkpoint)
        
        policy_net.eval()
        models[name] = policy_net
    
    # Create analyzer and generate report
    analyzer = ComparisonAnalyzer(models)
    return analyzer.generate_comparison_report(output_dir, show=show)
