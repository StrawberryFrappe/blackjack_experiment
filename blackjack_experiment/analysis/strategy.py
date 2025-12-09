"""Strategy analysis and visualization for Blackjack models.

This module provides tools to understand model behavior and strategy:
1. Decision heatmaps (hit/stick probabilities)
2. Input sensitivity analysis (what inputs matter most)
3. Basic strategy comparison and deviation analysis
4. Confusion matrix vs basic strategy
5. Per-checkpoint behavior progression

Note: Previously named decisions.py. Renamed to strategy.py for clarity.
Basic strategy functions moved to utils.py but re-exported here for compatibility.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import json
import seaborn as sns

# Re-export from utils for backward compatibility
from .utils import get_basic_strategy_action, create_blackjack_test_states


class BehaviorAnalyzer:
    """
    Analyzes and visualizes model behavior and decision-making patterns.
    
    Provides:
    - Decision heatmaps across all possible states
    - Input sensitivity (which state features matter most)
    - Failure analysis (where model makes wrong decisions)
    - Comparison with basic strategy
    """
    
    def __init__(self, policy_net, encode_fn: Optional[Callable] = None, value_net=None, **kwargs):
        """
        Initialize the analyzer.
        
        Args:
            policy_net: The policy network to analyze
            encode_fn: Optional function to encode state tuples
            value_net: Optional value network (currently unused, accepted for API compatibility)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.policy_net = policy_net
        self.encode_fn = encode_fn
        self.value_net = value_net  # Store but not currently used
        self.policy_net.eval()
    
    def _get_action_probs(self, state: Tuple[int, int, int]) -> torch.Tensor:
        """Get action probabilities for a state."""
        with torch.no_grad():
            if self.encode_fn:
                encoded = self.encode_fn(state)
                if not isinstance(encoded, torch.Tensor):
                    encoded = torch.tensor(encoded, dtype=torch.float32)
                probs = self.policy_net(encoded)
            else:
                probs = self.policy_net(state)
            return probs.squeeze()
    
    def create_decision_map(self, usable_ace: int = 0) -> Dict:
        """
        Create decision maps showing hit/stick probabilities.
        
        Args:
            usable_ace: Whether the player has a usable ace (0 or 1)
            
        Returns:
            Dictionary containing hit_probs, stick_probs, and actions matrices
        """
        player_range = range(4, 22)
        dealer_range = range(1, 11)
        
        hit_probs = np.zeros((len(player_range), len(dealer_range)))
        stick_probs = np.zeros((len(player_range), len(dealer_range)))
        actions = np.zeros((len(player_range), len(dealer_range)))
        confidence = np.zeros((len(player_range), len(dealer_range)))
        
        for i, player_sum in enumerate(player_range):
            for j, dealer_card in enumerate(dealer_range):
                probs = self._get_action_probs((player_sum, dealer_card, usable_ace))
                
                if len(probs) >= 2:
                    stick_probs[i, j] = probs[0].item()
                    hit_probs[i, j] = probs[1].item()
                    actions[i, j] = torch.argmax(probs).item()
                    confidence[i, j] = torch.max(probs).item()
        
        return {
            'hit_probs': hit_probs,
            'stick_probs': stick_probs,
            'actions': actions,
            'confidence': confidence,
            'player_range': list(player_range),
            'dealer_range': list(dealer_range),
            'usable_ace': usable_ace
        }
    
    def analyze_input_sensitivity(self, test_states=None) -> Dict:
        """
        Analyze which inputs have the most influence on decisions.
        
        Tests each input dimension by measuring how much decisions
        change when that input varies.
        
        Args:
            test_states: Optional test states (not used, but accepted for API compatibility)
        
        Returns:
            Dictionary with sensitivity metrics for each input factor
        """
        sensitivity = {
            'player_sum': [],
            'dealer_card': [],
            'usable_ace': []
        }
        
        # Player sum sensitivity (fixing other inputs)
        for dealer_card in range(1, 11):
            hit_probs = []
            for player_sum in range(4, 22):
                probs = self._get_action_probs((player_sum, dealer_card, 0))
                hit_probs.append(probs[1].item())
            sensitivity['player_sum'].append(np.var(hit_probs))
        
        # Dealer card sensitivity
        for player_sum in range(4, 22):
            hit_probs = []
            for dealer_card in range(1, 11):
                probs = self._get_action_probs((player_sum, dealer_card, 0))
                hit_probs.append(probs[1].item())
            sensitivity['dealer_card'].append(np.var(hit_probs))
        
        # Usable ace sensitivity
        for player_sum in range(12, 21):
            for dealer_card in range(1, 11):
                probs_no_ace = self._get_action_probs((player_sum, dealer_card, 0))
                probs_ace = self._get_action_probs((player_sum, dealer_card, 1))
                diff = abs(probs_no_ace[1].item() - probs_ace[1].item())
                sensitivity['usable_ace'].append(diff)
        
        return {
            'player_sum': {
                'mean_variance': np.mean(sensitivity['player_sum']),
                'max_variance': np.max(sensitivity['player_sum']),
                'importance': 'HIGH' if np.mean(sensitivity['player_sum']) > 0.1 else 'MEDIUM'
            },
            'dealer_card': {
                'mean_variance': np.mean(sensitivity['dealer_card']),
                'max_variance': np.max(sensitivity['dealer_card']),
                'importance': 'HIGH' if np.mean(sensitivity['dealer_card']) > 0.05 else 'MEDIUM'
            },
            'usable_ace': {
                'mean_diff': np.mean(sensitivity['usable_ace']),
                'max_diff': np.max(sensitivity['usable_ace']),
                'importance': 'HIGH' if np.mean(sensitivity['usable_ace']) > 0.1 else 'LOW'
            }
        }
    
    def analyze_layer_contributions(self, test_states=None) -> Dict:
        """
        Analyze layer contributions to the model output.
        
        This is a stub method for API compatibility with the trainer.
        For detailed analysis, use generate_confusion_matrix() instead.
        
        Args:
            test_states: Optional test states (not used in this implementation)
        
        Returns:
            Empty dictionary - layer analysis is done via other methods
        """
        # Layer contribution analysis is integrated into sensitivity analysis
        # This method exists for API compatibility
        return {}
    
    def save_analysis_results(self, filepath: str) -> None:
        """
        Save analysis results to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        import json
        
        results = {
            'sensitivity': self.analyze_input_sensitivity(),
            'confusion_matrix': self.generate_confusion_matrix(),
            'decision_map': self.create_decision_map()
        }
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def plot_comprehensive_analysis(self, save_path: str = None, show: bool = True) -> None:
        """
        Generate a comprehensive analysis plot.
        
        Args:
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        if save_path:
            from pathlib import Path
            output_dir = str(Path(save_path).parent)
            self.generate_full_report(
                output_dir=output_dir,
                show=show
            )
        else:
            # Generate to temp dir if no save_path provided
            import tempfile
            self.generate_full_report(
                output_dir=tempfile.gettempdir(),
                show=show
            )
    

    
    def generate_confusion_matrix(self) -> Dict:
        """
        Generate a confusion matrix comparing model decisions against basic strategy.
        
        Returns:
            Dictionary containing:
            - confusion_matrix: 2-by-2 matrix [predicted][optimal]
            - accuracy: Overall accuracy percentage
            - precision: Precision for each action
            - recall: Recall for each action
            - per_state_results: Detailed breakdown per state
        """
        # Initialize confusion matrix: [predicted_action][optimal_action]
        # confusion[0][0] = True Negative (correctly stand)
        # confusion[0][1] = False Negative (model stands, should hit)
        # confusion[1][0] = False Positive (model hits, should stand)
        # confusion[1][1] = True Positive (correctly hit)
        confusion = np.zeros((2, 2), dtype=int)
        
        per_state_results = []
        
        # Iterate all valid blackjack states
        for player_sum in range(4, 22):
            for dealer_card in range(1, 11):
                for usable_ace in [0, 1]:
                    # Skip invalid states (usable ace requires sum >= 12)
                    if usable_ace and player_sum < 12:
                        continue
                    
                    # Get model prediction
                    probs = self._get_action_probs((player_sum, dealer_card, usable_ace))
                    model_action = torch.argmax(probs).item()
                    model_confidence = probs[model_action].item()
                    
                    # Get optimal action from basic strategy
                    optimal_action = get_basic_strategy_action(player_sum, dealer_card, usable_ace)
                    
                    # Update confusion matrix
                    confusion[model_action][optimal_action] += 1
                    
                    # Store detailed result
                    per_state_results.append({
                        'state': (player_sum, dealer_card, usable_ace),
                        'model_action': model_action,
                        'optimal_action': optimal_action,
                        'correct': model_action == optimal_action,
                        'confidence': model_confidence
                    })
        
        # Calculate metrics
        total = confusion.sum()
        accuracy = (confusion[0][0] + confusion[1][1]) / total * 100
        
        # Precision and recall
        # For "Hit" action (1)
        hit_precision = confusion[1][1] / (confusion[1][0] + confusion[1][1]) * 100 if (confusion[1][0] + confusion[1][1]) > 0 else 0
        hit_recall = confusion[1][1] / (confusion[0][1] + confusion[1][1]) * 100 if (confusion[0][1] + confusion[1][1]) > 0 else 0
        
        # For "Stand" action (0)
        stand_precision = confusion[0][0] / (confusion[0][0] + confusion[0][1]) * 100 if (confusion[0][0] + confusion[0][1]) > 0 else 0
        stand_recall = confusion[0][0] / (confusion[0][0] + confusion[1][0]) * 100 if (confusion[0][0] + confusion[1][0]) > 0 else 0
        
        return {
            'confusion_matrix': confusion.tolist(),
            'accuracy': accuracy,
            'metrics': {
                'hit': {'precision': hit_precision, 'recall': hit_recall, 'f1': 2 * hit_precision * hit_recall / (hit_precision + hit_recall) if (hit_precision + hit_recall) > 0 else 0},
                'stand': {'precision': stand_precision, 'recall': stand_recall, 'f1': 2 * stand_precision * stand_recall / (stand_precision + stand_recall) if (stand_precision + stand_recall) > 0 else 0}
            },
            'counts': {
                'true_stand': int(confusion[0][0]),
                'false_stand': int(confusion[0][1]),  # Should have hit
                'false_hit': int(confusion[1][0]),    # Should have stood
                'true_hit': int(confusion[1][1])
            },
            'total_states': total,
            'per_state_results': per_state_results
        }
    
    def _create_basic_strategy_map(self, usable_ace: int = 0) -> np.ndarray:
        """
        Create a heatmap of basic strategy optimal actions.
        
        Returns:
            18x10 array where 0 = Stand, 1 = Hit
        """
        strategy_map = np.zeros((18, 10))  # player_sum 4-21, dealer 1-10
        
        for i, player_sum in enumerate(range(4, 22)):
            for j, dealer_card in enumerate(range(1, 11)):
                strategy_map[i, j] = get_basic_strategy_action(player_sum, dealer_card, usable_ace)
        
        return strategy_map
    
    def _create_deviation_map(self, usable_ace: int = 0) -> Tuple[np.ndarray, int, int]:
        """
        Create a map showing where model deviates from basic strategy.
        
        Returns:
            Tuple of (deviation_map, n_correct, n_wrong)
            deviation_map: 0 = agree, 1 = model hits when should stand, -1 = model stands when should hit
        """
        decision_map = self.create_decision_map(usable_ace)
        model_actions = decision_map['actions']
        strategy_map = self._create_basic_strategy_map(usable_ace)
        
        deviation_map = np.zeros_like(model_actions)
        n_correct = 0
        n_wrong = 0
        
        for i in range(model_actions.shape[0]):
            for j in range(model_actions.shape[1]):
                model = model_actions[i, j]
                optimal = strategy_map[i, j]
                
                if model == optimal:
                    deviation_map[i, j] = 0  # Correct
                    n_correct += 1
                elif model == 1 and optimal == 0:
                    deviation_map[i, j] = 1  # Model hits when should stand (too aggressive)
                    n_wrong += 1
                else:
                    deviation_map[i, j] = -1  # Model stands when should hit (too passive)
                    n_wrong += 1
        
        return deviation_map, n_correct, n_wrong

    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot enhanced confusion matrix with basic strategy visualization.
        
        Shows:
        - Basic strategy heatmap (what optimal play looks like)
        - Model vs basic strategy deviation map (where model differs)
        - Traditional confusion matrix
        - Classification metrics
        """
        cm_data = self.generate_confusion_matrix()
        confusion = np.array(cm_data['confusion_matrix'])
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)
        
        action_cmap = self._create_custom_colormap()
        
        # =================================================================
        # ROW 1: Basic Strategy (no ace), Basic Strategy (ace), Deviation Map
        # =================================================================
        
        # Plot 1: Basic Strategy - No Usable Ace
        ax = fig.add_subplot(gs[0, 0])
        strategy_no_ace = self._create_basic_strategy_map(0)
        im = ax.imshow(
            strategy_no_ace,
            aspect='auto',
            cmap=action_cmap,
            origin='lower',
            extent=[0.5, 10.5, 3.5, 21.5],
            vmin=0, vmax=1
        )
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_title('Basic Strategy (No Ace)\nGreen=Hit, Red=Stand', fontweight='bold')
        ax.set_xticks(range(1, 11))
        ax.set_yticks(range(4, 22, 2))
        cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75], fraction=0.046)
        cbar.ax.set_yticklabels(['Stand', 'Hit'])
        
        # Plot 2: Basic Strategy - Usable Ace
        ax = fig.add_subplot(gs[0, 1])
        strategy_ace = self._create_basic_strategy_map(1)
        im = ax.imshow(
            strategy_ace,
            aspect='auto',
            cmap=action_cmap,
            origin='lower',
            extent=[0.5, 10.5, 3.5, 21.5],
            vmin=0, vmax=1
        )
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_title('Basic Strategy (Usable Ace)\nGreen=Hit, Red=Stand', fontweight='bold')
        ax.set_xticks(range(1, 11))
        ax.set_yticks(range(4, 22, 2))
        cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75], fraction=0.046)
        cbar.ax.set_yticklabels(['Stand', 'Hit'])
        
        # Plot 3: Combined Deviation Map
        ax = fig.add_subplot(gs[0, 2])
        dev_no_ace, correct_no_ace, wrong_no_ace = self._create_deviation_map(0)
        dev_ace, correct_ace, wrong_ace = self._create_deviation_map(1)
        
        # Combine: use no ace as base, but highlight ace deviations too
        combined_dev = dev_no_ace.copy()
        # Mark ace deviations with higher intensity where applicable
        for i in range(8, 18):  # player_sum 12-21 (where ace matters)
            for j in range(10):
                if dev_ace[i, j] != 0:
                    combined_dev[i, j] = dev_ace[i, j] * 0.5 + combined_dev[i, j] * 0.5
        
        # Custom colormap: Blue=too passive, White=correct, Orange=too aggressive
        from matplotlib.colors import LinearSegmentedColormap
        deviation_colors = ['#3498db', '#ffffff', '#e74c3c']  # Blue, White, Red
        deviation_cmap = LinearSegmentedColormap.from_list('deviation', deviation_colors, N=256)
        
        im = ax.imshow(
            combined_dev,
            aspect='auto',
            cmap=deviation_cmap,
            origin='lower',
            extent=[0.5, 10.5, 3.5, 21.5],
            vmin=-1, vmax=1
        )
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        total_correct = correct_no_ace + correct_ace
        total_wrong = wrong_no_ace + wrong_ace
        ax.set_title(f'Model Deviations from Basic Strategy\n{total_correct} correct, {total_wrong} wrong', fontweight='bold')
        ax.set_xticks(range(1, 11))
        ax.set_yticks(range(4, 22, 2))
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_ticks([-1, 0, 1])
        cbar.ax.set_yticklabels(['Too Passive', 'Correct', 'Too Aggressive'])
        
        # =================================================================
        # ROW 2: Confusion Matrix, Metrics, Detailed Deviation Stats
        # =================================================================
        
        # Plot 4: Confusion Matrix
        ax = fig.add_subplot(gs[1, 0])
        labels = ['Stand', 'Hit']
        sns.heatmap(
            confusion,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        ax.set_xlabel('Optimal Action (Basic Strategy)', fontsize=11)
        ax.set_ylabel('Model Prediction', fontsize=11)
        ax.set_title(f'Confusion Matrix\nAccuracy: {cm_data["accuracy"]:.1f}%', fontsize=12, fontweight='bold')
        
        # Plot 5: Metrics bar chart
        ax = fig.add_subplot(gs[1, 1])
        metrics = cm_data['metrics']
        x = np.arange(2)
        width = 0.25
        
        precision_vals = [metrics['stand']['precision'], metrics['hit']['precision']]
        recall_vals = [metrics['stand']['recall'], metrics['hit']['recall']]
        f1_vals = [metrics['stand']['f1'], metrics['hit']['f1']]
        
        bars1 = ax.bar(x - width, precision_vals, width, label='Precision', color='#3498db')
        bars2 = ax.bar(x, recall_vals, width, label='Recall', color='#e74c3c')
        bars3 = ax.bar(x + width, f1_vals, width, label='F1 Score', color='#2ecc71')
        
        ax.set_xlabel('Action', fontsize=11)
        ax.set_ylabel('Score (%)', fontsize=11)
        ax.set_title('Classification Metrics', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Stand', 'Hit'])
        ax.set_ylim([0, 100])
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        # Plot 6: Deviation Summary (text box)
        ax = fig.add_subplot(gs[1, 2])
        ax.axis('off')
        
        summary_text = "DEVIATION ANALYSIS\n" + "="*35 + "\n\n"
        summary_text += f"Overall Accuracy: {cm_data['accuracy']:.1f}%\n\n"
        
        summary_text += "Without Usable Ace:\n"
        summary_text += f"  [OK] Correct: {correct_no_ace} states\n"
        summary_text += f"  [X] Wrong:   {wrong_no_ace} states\n\n"
        
        summary_text += "With Usable Ace:\n"
        summary_text += f"  [OK] Correct: {correct_ace} states\n"
        summary_text += f"  [X] Wrong:   {wrong_ace} states\n\n"
        
        # Count error types
        too_aggressive = cm_data['counts']['false_hit']
        too_passive = cm_data['counts']['false_stand']
        
        summary_text += "Error Breakdown:\n"
        summary_text += f"  🔴 Too Aggressive: {too_aggressive}\n"
        summary_text += f"     (hits when should stand)\n"
        summary_text += f"  🔵 Too Passive: {too_passive}\n"
        summary_text += f"     (stands when should hit)"
        
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='center', horizontalalignment='center',
               fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))
        ax.set_title('Deviation Summary', fontweight='bold')
        
        plt.suptitle('Model vs Basic Strategy - Complete Analysis', fontsize=14, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Confusion matrix saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, cm_data
    
    def _create_custom_colormap(self):
        """Get colormap: Red=Stand(0), Green=Hit(1)."""
        import matplotlib.pyplot as plt
        return plt.cm.RdYlGn
    
    def plot_decision_heatmaps(
        self, 
        save_path: Optional[str] = None, 
        show: bool = False,
        figsize: Tuple[int, int] = (16, 8)
    ):
        """
        Plot decision heatmaps for both usable ace states.
        
        Shows:
        - Hit probability heatmap
        - Discrete action map
        - Confidence map
        
        Color scheme: Green = Hit, Red = Stand
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Custom colormap: Red(Stand) to Green(Hit)
        action_cmap = self._create_custom_colormap()
        
        for row, usable_ace in enumerate([0, 1]):
            decision_map = self.create_decision_map(usable_ace)
            
            player_range = decision_map['player_range']
            dealer_range = decision_map['dealer_range']
            
            # Hit probability heatmap (green = high hit prob, red = low)
            ax = axes[row, 0]
            im = ax.imshow(
                decision_map['hit_probs'], 
                aspect='auto', 
                cmap=action_cmap,
                origin='lower',
                extent=[0.5, 10.5, 3.5, 21.5],
                vmin=0, vmax=1
            )
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ace_str = 'With' if usable_ace else 'Without'
            ax.set_title(f'Hit Probability ({ace_str} Usable Ace)')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('P(Hit)', fontsize=9)
            cbar.ax.set_ylabel('Stand ← → Hit', fontsize=8)
            
            # Discrete action map (green = hit, red = stand)
            ax = axes[row, 1]
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
            ax.set_title(f'Actions ({ace_str} Usable Ace)')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            # Add colorbar with discrete labels
            cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
            cbar.ax.set_yticklabels(['Stand (Red)', 'Hit (Green)'])
            
            # Confidence map
            ax = axes[row, 2]
            im = ax.imshow(
                decision_map['confidence'], 
                aspect='auto', 
                cmap='viridis',
                origin='lower',
                extent=[0.5, 10.5, 3.5, 21.5],
                vmin=0.5, vmax=1
            )
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ax.set_title(f'Decision Confidence ({ace_str} Usable Ace)')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            plt.colorbar(im, ax=ax, label='Confidence')
        
        plt.suptitle('Blackjack Decision Analysis (Green=Hit, Red=Stand)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Decision heatmaps saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_comprehensive_decision_analysis(
        self,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Create comprehensive 3-by-3 decision analysis plot combining:
        - Row 1: Player sum influence, Dealer card influence, Input sensitivity (all 3 axes)
        - Row 2: Decision heatmap without usable ace
        - Row 3: Decision heatmap with usable ace
        
        Color scheme: Green = Hit, Red = Stand
        """
        fig = plt.figure(figsize=(18, 16))
        
        # Create GridSpec for flexible layout
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.2, 1.2], hspace=0.3, wspace=0.25)
        
        # Custom colormap
        action_cmap = self._create_custom_colormap()
        
        # Get sensitivity data
        sensitivity = self.analyze_input_sensitivity()
        
        # =====================================================================
        # ROW 1: Marginal decision profiles and combined sensitivity
        # =====================================================================
        
        # Plot 1: Player Sum Marginal (average hit prob across all dealer cards)
        ax = fig.add_subplot(gs[0, 0])
        player_hit_probs = []
        for player_sum in range(4, 22):
            probs = []
            for dealer_card in range(1, 11):
                p = self._get_action_probs((player_sum, dealer_card, 0))
                probs.append(p[1].item())
            player_hit_probs.append(np.mean(probs))
        
        colors = [action_cmap(p) for p in player_hit_probs]
        bars = ax.bar(range(4, 22), player_hit_probs, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Player Sum', fontsize=10)
        ax.set_ylabel('Avg Hit Probability', fontsize=10)
        ax.set_title('Decision by Player Sum\n(averaged over dealer cards)', fontsize=11)
        ax.set_ylim([0, 1])
        ax.set_xticks(range(4, 22, 2))
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Dealer Card Marginal (average hit prob across player sums 12-19)
        ax = fig.add_subplot(gs[0, 1])
        dealer_hit_probs = []
        for dealer_card in range(1, 11):
            probs = []
            for player_sum in range(12, 20):  # Focus on interesting range
                p = self._get_action_probs((player_sum, dealer_card, 0))
                probs.append(p[1].item())
            dealer_hit_probs.append(np.mean(probs))
        
        dealer_labels = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        colors = [action_cmap(p) for p in dealer_hit_probs]
        bars = ax.bar(dealer_labels, dealer_hit_probs, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Dealer Card', fontsize=10)
        ax.set_ylabel('Avg Hit Probability', fontsize=10)
        ax.set_title('Decision by Dealer Card\n(player sum 12-19)', fontsize=11)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Combined Input Sensitivity
        ax = fig.add_subplot(gs[0, 2])
        factors = ['Player Sum', 'Dealer Card', 'Usable Ace']
        
        # Get importance scores (normalized)
        player_importance = sensitivity['player_sum']['mean_variance']
        dealer_importance = sensitivity['dealer_card']['mean_variance']
        ace_importance = sensitivity['usable_ace']['mean_diff']
        
        # Normalize to percentage
        total = player_importance + dealer_importance + ace_importance
        if total > 0:
            scores = [
                player_importance / total * 100,
                dealer_importance / total * 100,
                ace_importance / total * 100
            ]
        else:
            scores = [33.3, 33.3, 33.3]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.bar(factors, scores, color=colors, edgecolor='black', alpha=0.8)
        ax.set_ylabel('Relative Importance (%)', fontsize=10)
        ax.set_title('Input Sensitivity\n(contribution to decisions)', fontsize=11)
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add importance labels
        importance_labels = [
            sensitivity['player_sum']['importance'],
            sensitivity['dealer_card']['importance'],
            sensitivity['usable_ace']['importance']
        ]
        for bar, score, imp in zip(bars, scores, importance_labels):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{score:.1f}%\n({imp})', ha='center', va='bottom', fontsize=9)
        
        # =====================================================================
        # ROW 2 & 3: Full decision heatmaps
        # =====================================================================
        
        for row, usable_ace in enumerate([0, 1]):
            decision_map = self.create_decision_map(usable_ace)
            ace_str = 'Without' if usable_ace == 0 else 'With'
            grid_row = row + 1
            
            # Hit probability heatmap
            ax = fig.add_subplot(gs[grid_row, 0])
            im = ax.imshow(
                decision_map['hit_probs'],
                aspect='auto',
                cmap=action_cmap,
                origin='lower',
                extent=[0.5, 10.5, 3.5, 21.5],
                vmin=0, vmax=1
            )
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ax.set_title(f'Hit Probability ({ace_str} Ace)')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('Stand ← → Hit', fontsize=8)
            
            # Discrete action map
            ax = fig.add_subplot(gs[grid_row, 1])
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
            ax.set_title(f'Actions ({ace_str} Ace)')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75], fraction=0.046, pad=0.04)
            cbar.ax.set_yticklabels(['Stand', 'Hit'])
            
            # Confidence map
            ax = fig.add_subplot(gs[grid_row, 2])
            im = ax.imshow(
                decision_map['confidence'],
                aspect='auto',
                cmap='viridis',
                origin='lower',
                extent=[0.5, 10.5, 3.5, 21.5],
                vmin=0.5, vmax=1
            )
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ax.set_title(f'Confidence ({ace_str} Ace)')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Confidence')
        
        plt.suptitle('Comprehensive Decision Analysis\nGreen = Hit, Red = Stand', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Comprehensive decision analysis saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_sensitivity_analysis(
        self, 
        save_path: Optional[str] = None, 
        show: bool = False
    ):
        """Plot input sensitivity analysis results."""
        sensitivity = self.analyze_input_sensitivity()
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        factors = ['player_sum', 'dealer_card', 'usable_ace']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, (factor, color) in enumerate(zip(factors, colors)):
            ax = axes[i]
            data = sensitivity[factor]
            
            if 'mean_variance' in data:
                bars = [data['mean_variance'], data['max_variance']]
                labels = ['Mean Var', 'Max Var']
            else:
                bars = [data['mean_diff'], data['max_diff']]
                labels = ['Mean Diff', 'Max Diff']
            
            ax.bar(labels, bars, color=color, alpha=0.7, edgecolor='black')
            ax.set_title(f'{factor.replace("_", " ").title()}\n({data["importance"]})')
            ax.set_ylabel('Sensitivity')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Input Sensitivity Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Sensitivity analysis saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    

    
    def generate_full_report(
        self, 
        output_dir: str,
        show: bool = False
    ) -> Dict:
        """
        Generate a complete decision analysis report.
        
        Creates:
        - Comprehensive decision analysis (3-by-3 grid with sensitivity + heatmaps)
        - Confusion matrix vs basic strategy
        - Failure analysis plot
        - JSON summary with all metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("DECISION ANALYSIS REPORT")
        print("="*60)
        
        # Generate comprehensive 3-by-3 decision analysis (combines sensitivity + heatmaps)
        self.plot_comprehensive_decision_analysis(
            save_path=str(output_path / 'decision_analysis.png'),
            show=show
        )
        
        # Generate confusion matrix
        _, cm_data = self.plot_confusion_matrix(
            save_path=str(output_path / 'confusion_matrix.png'),
            show=show
        )
        
        # Generate summary
        sensitivity = self.analyze_input_sensitivity()
        
        summary = {
            'sensitivity': sensitivity,
            'confusion_matrix': {
                'accuracy': cm_data['accuracy'],
                'matrix': cm_data['confusion_matrix'],
                'metrics': cm_data['metrics'],
                'counts': cm_data['counts']
            },
            'decision_maps': {
                'no_ace': self.create_decision_map(0),
                'usable_ace': self.create_decision_map(1)
            }
        }
        
        # Convert numpy arrays to lists for JSON
        for key in summary['decision_maps']:
            for subkey in summary['decision_maps'][key]:
                if isinstance(summary['decision_maps'][key][subkey], np.ndarray):
                    summary['decision_maps'][key][subkey] = summary['decision_maps'][key][subkey].tolist()
        
        with open(output_path / 'decision_analysis.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[DATA] Report saved to: {output_path}")
        print(f"   - decision_analysis.png (comprehensive 3-by-3 grid)")
        print(f"   - confusion_matrix.png (vs basic strategy)")
        print(f"   - decision_analysis.json")
        print("="*60)
        
        return summary


def compare_decisions(
    models: Dict[str, 'torch.nn.Module'],
    encode_fns: Optional[Dict[str, Callable]] = None,
    save_path: Optional[str] = None,
    show: bool = False
) -> Dict:
    """
    Compare decisions between multiple models.
    
    Args:
        models: Dictionary of {name: policy_net}
        encode_fns: Optional dictionary of {name: encode_fn}
        save_path: Path to save comparison plot
        show: Whether to display the plot
        
    Returns:
        Dictionary with comparison metrics
    """
    if encode_fns is None:
        encode_fns = {}
    
    # Create analyzers
    analyzers = {
        name: DecisionAnalyzer(net, encode_fns.get(name))
        for name, net in models.items()
    }
    
    # Create decision maps
    decision_maps = {
        name: analyzer.create_decision_map(0)
        for name, analyzer in analyzers.items()
    }
    
    # Plot comparison
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, dmap) in zip(axes, decision_maps.items()):
        im = ax.imshow(
            dmap['hit_probs'],
            aspect='auto',
            cmap='RdYlGn_r',
            origin='lower',
            extent=[0.5, 10.5, 3.5, 21.5],
            vmin=0, vmax=1
        )
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_title(f'{name}\nHit Probability')
        ax.set_xticks(range(1, 11))
        ax.set_yticks(range(4, 22, 2))
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Model Decision Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Comparison saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Compute agreement metrics
    if n_models >= 2:
        names = list(decision_maps.keys())
        agreements = {}
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                actions1 = decision_maps[name1]['actions']
                actions2 = decision_maps[name2]['actions']
                agreement = np.mean(actions1 == actions2) * 100
                agreements[f'{name1}_vs_{name2}'] = agreement
        return {'decision_maps': decision_maps, 'agreements': agreements}
    
    return {'decision_maps': decision_maps}


# Aliases for backward compatibility
ModelAnalyzer = BehaviorAnalyzer
DecisionAnalyzer = BehaviorAnalyzer
