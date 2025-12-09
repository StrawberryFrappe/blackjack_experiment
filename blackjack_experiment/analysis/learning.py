"""
Learning progression analysis for Blackjack models.

This module provides tools to analyze WHEN and HOW models learn to play Blackjack,
tracking the "moment of realization" when they discover key strategic insights.

Key analyses:
1. Basic strategy adherence (stick on 20-21, hit on low sums)
2. Dealer awareness (adapting to dealer's card)
3. Confidence development over training
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import json
import re


class LearningAnalyzer:
    """
    Analyzes the learning progression of Blackjack models through checkpoints.
    
    Tracks:
    - When models learn basic strategy
    - When models differentiate based on dealer's card
    - When models learn to handle usable aces
    - Strategic breakthrough moments
    """
    
    def __init__(self, checkpoint_dir: str, model_factory_fn: Callable):
        """
        Initialize the analyzer.
        
        Args:
            checkpoint_dir: Directory containing model checkpoints (or parent of checkpoints/)
            model_factory_fn: Function that creates a fresh model instance
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_factory_fn = model_factory_fn
        
        # Check for checkpoints in new location (checkpoints/ subfolder)
        checkpoints_subdir = self.checkpoint_dir / 'checkpoints'
        if checkpoints_subdir.exists() and any(checkpoints_subdir.glob('checkpoint_*.pth')):
            self.checkpoint_dir = checkpoints_subdir
        
        self.checkpoints = self._discover_checkpoints()
        self.analysis_results = []
        
    def _discover_checkpoints(self) -> List[Dict]:
        """Discover and sort all checkpoint files."""
        checkpoints = []
        pattern = re.compile(r'checkpoint_ep(\d+)(?:_wr([\d.]+))?\.pth')
        
        for ckpt_path in self.checkpoint_dir.glob('checkpoint_*.pth'):
            match = pattern.match(ckpt_path.name)
            if match:
                episode = int(match.group(1))
                win_rate = float(match.group(2)) if match.group(2) else None
                checkpoints.append({
                    'path': ckpt_path,
                    'episode': episode,
                    'win_rate': win_rate,
                    'name': ckpt_path.stem
                })
        
        checkpoints.sort(key=lambda x: x['episode'])
        print(f"Discovered {len(checkpoints)} checkpoints in {self.checkpoint_dir}")
        return checkpoints
    
    def _load_checkpoint(self, checkpoint_info: Dict):
        """
        Load a model from checkpoint with correct architecture.
        
        Uses the network loader to extract configuration from the checkpoint
        and create a network with matching architecture, rather than relying
        on the model_factory_fn (which uses current defaults).
        """
        from ..networks.loader import load_policy_network, NetworkLoadError
        
        try:
            # Try to load using the proper loader (extracts config from checkpoint)
            policy_net = load_policy_network(checkpoint_info['path'], strict=True)
            return policy_net
        except NetworkLoadError:
            # Fall back to factory for old checkpoints without network_config
            # This may fail if architecture doesn't match
            print(f"  [WARN] Checkpoint lacks network_config, using factory (may fail)")
            model = self.model_factory_fn()
            
            if isinstance(model, tuple):
                policy_net = model[0]
            else:
                policy_net = model
            
            checkpoint = torch.load(checkpoint_info['path'], map_location='cpu', weights_only=False)
            
            if 'policy_net' in checkpoint:
                policy_net.load_state_dict(checkpoint['policy_net'])
            elif 'policy_state_dict' in checkpoint:
                policy_net.load_state_dict(checkpoint['policy_state_dict'])
            elif 'model_state_dict' in checkpoint:
                policy_net.load_state_dict(checkpoint['model_state_dict'])
            else:
                policy_net.load_state_dict(checkpoint)
            
            policy_net.eval()
            return policy_net
    
    def _evaluate_basic_strategy(self, policy_net) -> Dict:
        """Evaluate how well the model follows basic Blackjack strategy."""
        with torch.no_grad():
            metrics = {}
            
            # Test 1: Stick on 20-21
            stick_on_high = []
            for player_sum in [20, 21]:
                for dealer_card in range(1, 11):
                    probs = policy_net((player_sum, dealer_card, 0)).squeeze()
                    action = torch.argmax(probs).item()
                    stick_on_high.append(1 if action == 0 else 0)
            metrics['stick_on_20_21'] = np.mean(stick_on_high) * 100
            
            # Test 2: Hit on low sums (4-11)
            hit_on_low = []
            for player_sum in range(4, 12):
                for dealer_card in range(1, 11):
                    probs = policy_net((player_sum, dealer_card, 0)).squeeze()
                    action = torch.argmax(probs).item()
                    hit_on_low.append(1 if action == 1 else 0)
            metrics['hit_on_low_sums'] = np.mean(hit_on_low) * 100
            
            # Test 3: Rational behavior on 12-16
            rational_decisions = []
            for player_sum in range(12, 17):
                for dealer_card in range(7, 11):  # Strong dealer
                    probs = policy_net((player_sum, dealer_card, 0)).squeeze()
                    action = torch.argmax(probs).item()
                    rational_decisions.append(1 if action == 1 else 0)
                for dealer_card in range(2, 7):  # Weak dealer
                    probs = policy_net((player_sum, dealer_card, 0)).squeeze()
                    action = torch.argmax(probs).item()
                    rational_decisions.append(1 if action == 0 else 0)
            metrics['rational_12_16'] = np.mean(rational_decisions) * 100
            
            # Test 4: Stick on 17-19
            stick_on_17_19 = []
            for player_sum in range(17, 20):
                for dealer_card in range(1, 11):
                    probs = policy_net((player_sum, dealer_card, 0)).squeeze()
                    action = torch.argmax(probs).item()
                    stick_on_17_19.append(1 if action == 0 else 0)
            metrics['stick_on_17_19'] = np.mean(stick_on_17_19) * 100
            
            # Test 5: Usable ace aggression
            hit_rate_no_ace = []
            hit_rate_usable_ace = []
            for player_sum in range(12, 20):
                for dealer_card in range(1, 11):
                    probs_no_ace = policy_net((player_sum, dealer_card, 0)).squeeze()
                    hit_rate_no_ace.append(1 if torch.argmax(probs_no_ace).item() == 1 else 0)
                    probs_ace = policy_net((player_sum, dealer_card, 1)).squeeze()
                    hit_rate_usable_ace.append(1 if torch.argmax(probs_ace).item() == 1 else 0)
            
            metrics['usable_ace_aggression'] = (np.mean(hit_rate_usable_ace) - np.mean(hit_rate_no_ace)) * 100
            
            # Overall strategy score
            metrics['overall_strategy_score'] = (
                metrics['stick_on_20_21'] * 0.3 +
                metrics['hit_on_low_sums'] * 0.25 +
                metrics['rational_12_16'] * 0.25 +
                metrics['stick_on_17_19'] * 0.15 +
                (min(metrics['usable_ace_aggression'], 20) / 20 * 100) * 0.05
            )
            
            return metrics
    
    def _evaluate_dealer_awareness(self, policy_net) -> Dict:
        """Evaluate how much the model adapts to the dealer's card."""
        with torch.no_grad():
            # Check decision variance across dealer cards
            decision_variance = []
            for player_sum in range(12, 20):
                hit_probs = []
                for dealer_card in range(1, 11):
                    probs = policy_net((player_sum, dealer_card, 0)).squeeze()
                    hit_probs.append(probs[1].item())
                decision_variance.append(np.var(hit_probs))
            
            # Conservative against weak dealer, aggressive against strong
            stick_rate_weak = []
            hit_rate_strong = []
            for player_sum in range(12, 17):
                for dealer_card in [4, 5, 6]:  # Weak
                    probs = policy_net((player_sum, dealer_card, 0)).squeeze()
                    stick_rate_weak.append(1 if torch.argmax(probs).item() == 0 else 0)
                for dealer_card in [9, 10]:  # Strong
                    probs = policy_net((player_sum, dealer_card, 0)).squeeze()
                    hit_rate_strong.append(1 if torch.argmax(probs).item() == 1 else 0)
            
            return {
                'decision_variance': np.mean(decision_variance),
                'weak_dealer_conservatism': np.mean(stick_rate_weak) * 100,
                'strong_dealer_aggression': np.mean(hit_rate_strong) * 100,
                'dealer_awareness_score': (np.mean(stick_rate_weak) + np.mean(hit_rate_strong)) * 50
            }
    
    def _evaluate_confidence(self, policy_net) -> Dict:
        """Evaluate the model's confidence in its decisions."""
        with torch.no_grad():
            confidences = []
            for player_sum in range(4, 22):
                for dealer_card in range(1, 11):
                    for usable_ace in [0, 1]:
                        probs = policy_net((player_sum, dealer_card, usable_ace)).squeeze()
                        confidences.append(torch.max(probs).item())
            
            return {
                'mean_confidence': np.mean(confidences) * 100,
                'min_confidence': np.min(confidences) * 100,
                'std_confidence': np.std(confidences) * 100
            }
    
    def analyze_learning_progression(self, max_checkpoints: Optional[int] = None) -> List[Dict]:
        """Analyze learning progression across checkpoints."""
        checkpoints_to_analyze = self.checkpoints
        if max_checkpoints and len(self.checkpoints) > max_checkpoints:
            step = len(self.checkpoints) // max_checkpoints
            checkpoints_to_analyze = self.checkpoints[::step]
        
        print("\n" + "="*80)
        print("LEARNING PROGRESSION ANALYSIS")
        print("="*80)
        print(f"Analyzing {len(checkpoints_to_analyze)} checkpoints...")
        
        self.analysis_results = []
        
        for i, ckpt in enumerate(checkpoints_to_analyze):
            print(f"[{i+1}/{len(checkpoints_to_analyze)}] Episode {ckpt['episode']}...", end=" ")
            
            policy_net = self._load_checkpoint(ckpt)
            
            result = {
                'episode': ckpt['episode'],
                'win_rate': ckpt['win_rate'],
                'checkpoint_name': ckpt['name'],
                'basic_strategy': self._evaluate_basic_strategy(policy_net),
                'dealer_awareness': self._evaluate_dealer_awareness(policy_net),
                'confidence': self._evaluate_confidence(policy_net)
            }
            
            self.analysis_results.append(result)
            print("[OK]")
        
        self._identify_breakthroughs()
        return self.analysis_results
    
    def _identify_breakthroughs(self):
        """Identify moments of significant strategic improvement."""
        if len(self.analysis_results) < 2:
            return
        
        print("\n" + "="*80)
        print("BREAKTHROUGH MOMENTS")
        print("="*80)
        
        breakthroughs = []
        
        for i in range(1, len(self.analysis_results)):
            prev = self.analysis_results[i-1]
            curr = self.analysis_results[i]
            
            strategy_improvement = (
                curr['basic_strategy']['overall_strategy_score'] - 
                prev['basic_strategy']['overall_strategy_score']
            )
            
            if strategy_improvement > 10:
                breakthroughs.append({
                    'episode': curr['episode'],
                    'description': f"Major strategic breakthrough (+{strategy_improvement:.1f}%)"
                })
        
        if breakthroughs:
            for bt in breakthroughs:
                print(f"⚡ Episode {bt['episode']:>6}: {bt['description']}")
        else:
            print("No major breakthroughs detected (gradual learning)")
        
        print("="*80)
        
        # Also detect forgetting
        self._detect_forgetting()
    
    def _detect_forgetting(self):
        """
        Detect catastrophic forgetting events.
        
        Forgetting patterns include:
        - Loss of dealer card sensitivity (decisions no longer vary by dealer)
        - Loss of usable ace awareness
        - Regression in basic strategy adherence
        - Overall strategy score drops
        """
        if len(self.analysis_results) < 2:
            return
        
        print("\n" + "="*80)
        print("FORGETTING DETECTION")
        print("="*80)
        
        forgetting_events = []
        
        # Track peak values
        peak_strategy_score = self.analysis_results[0]['basic_strategy']['overall_strategy_score']
        peak_dealer_awareness = self.analysis_results[0]['dealer_awareness']['dealer_awareness_score']
        peak_dealer_variance = self.analysis_results[0]['dealer_awareness']['decision_variance']
        
        for i in range(1, len(self.analysis_results)):
            prev = self.analysis_results[i-1]
            curr = self.analysis_results[i]
            
            # Update peaks
            peak_strategy_score = max(peak_strategy_score, prev['basic_strategy']['overall_strategy_score'])
            peak_dealer_awareness = max(peak_dealer_awareness, prev['dealer_awareness']['dealer_awareness_score'])
            peak_dealer_variance = max(peak_dealer_variance, prev['dealer_awareness']['decision_variance'])
            
            events_at_episode = []
            
            # Check for strategy score regression (drop from peak)
            current_score = curr['basic_strategy']['overall_strategy_score']
            if peak_strategy_score - current_score > 10:
                events_at_episode.append({
                    'type': 'strategy_regression',
                    'description': f'Strategy score dropped from peak {peak_strategy_score:.1f}% to {current_score:.1f}%',
                    'severity': 'HIGH' if peak_strategy_score - current_score > 20 else 'MEDIUM'
                })
            
            # Check for dealer awareness loss
            current_awareness = curr['dealer_awareness']['dealer_awareness_score']
            prev_awareness = prev['dealer_awareness']['dealer_awareness_score']
            if prev_awareness - current_awareness > 15:
                events_at_episode.append({
                    'type': 'dealer_awareness_loss',
                    'description': f'Dealer awareness dropped from {prev_awareness:.1f}% to {current_awareness:.1f}%',
                    'severity': 'HIGH'
                })
            
            # Check for dealer variance collapse (model ignoring dealer card)
            current_variance = curr['dealer_awareness']['decision_variance']
            if peak_dealer_variance > 0.01 and current_variance < peak_dealer_variance * 0.3:
                events_at_episode.append({
                    'type': 'dealer_sensitivity_collapse',
                    'description': f'Decision variance by dealer card collapsed (was {peak_dealer_variance:.3f}, now {current_variance:.3f})',
                    'severity': 'HIGH'
                })
            
            # Check for stick_on_20_21 regression (critical)
            curr_stick = curr['basic_strategy']['stick_on_20_21']
            prev_stick = prev['basic_strategy']['stick_on_20_21']
            if prev_stick - curr_stick > 10:
                events_at_episode.append({
                    'type': 'critical_regression',
                    'description': f'Stick on 20-21 dropped from {prev_stick:.1f}% to {curr_stick:.1f}%',
                    'severity': 'CRITICAL'
                })
            
            # Check for usable ace handling regression
            curr_ace_agg = curr['basic_strategy']['usable_ace_aggression']
            prev_ace_agg = prev['basic_strategy']['usable_ace_aggression']
            # Good ace handling should be positive (more aggressive with ace)
            if prev_ace_agg > 5 and curr_ace_agg < 2:
                events_at_episode.append({
                    'type': 'ace_awareness_loss',
                    'description': f'Lost usable ace awareness (aggression: {prev_ace_agg:.1f}% → {curr_ace_agg:.1f}%)',
                    'severity': 'MEDIUM'
                })
            
            if events_at_episode:
                forgetting_events.append({
                    'episode': curr['episode'],
                    'events': events_at_episode
                })
        
        # Report findings
        if forgetting_events:
            print(f"[WARN]  FORGETTING DETECTED ({len(forgetting_events)} episodes with regression)")
            for fe in forgetting_events:
                print(f"\n  Episode {fe['episode']}:")
                for event in fe['events']:
                    severity_icon = '🔴' if event['severity'] == 'CRITICAL' else ('🟡' if event['severity'] == 'HIGH' else '🟢')
                    print(f"    {severity_icon} [{event['type']}] {event['description']}")
        else:
            print("✅ No significant forgetting detected")
        
        print("="*80)
        
        self.forgetting_events = forgetting_events
        return forgetting_events
    
    def plot_learning_progression(self, save_path: Optional[str] = None, show: bool = False):
        """
        Create unified visualization of learning progression with forgetting detection.
        
        Combines learning timeline and forgetting analysis into a single 3x3 grid:
        - Row 1: Win Rate, Strategy Score (with forgetting markers), Basic Skills
        - Row 2: Dealer Awareness, Confidence, Usable Ace Handling
        - Row 3: Critical Skills Stability, Forgetting Events Summary, Strategy Correlation
        """
        if not self.analysis_results:
            print("[WARN] Run analyze_learning_progression() first!")
            return
        
        episodes = [r['episode'] for r in self.analysis_results]
        overall_scores = [r['basic_strategy']['overall_strategy_score'] for r in self.analysis_results]
        stick_20_21 = [r['basic_strategy']['stick_on_20_21'] for r in self.analysis_results]
        hit_low = [r['basic_strategy']['hit_on_low_sums'] for r in self.analysis_results]
        dealer_awareness = [r['dealer_awareness']['dealer_awareness_score'] for r in self.analysis_results]
        dealer_variance = [r['dealer_awareness']['decision_variance'] for r in self.analysis_results]
        confidence_mean = [r['confidence']['mean_confidence'] for r in self.analysis_results]
        ace_agg = [r['basic_strategy']['usable_ace_aggression'] for r in self.analysis_results]
        win_rates = [r.get('win_rate') for r in self.analysis_results]
        
        # Filter None values from win_rates
        win_rate_episodes = [e for e, wr in zip(episodes, win_rates) if wr is not None]
        win_rates_filtered = [wr for wr in win_rates if wr is not None]
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # =================================================================
        # ROW 1: Win Rate, Strategy Score with Forgetting, Basic Skills
        # =================================================================
        
        # Plot 1: Win Rate
        ax = fig.add_subplot(gs[0, 0])
        if win_rates_filtered:
            ax.plot(win_rate_episodes, win_rates_filtered, linewidth=2.5, color='#9b59b6', marker='D', markersize=6)
            ax.fill_between(win_rate_episodes, 0, win_rates_filtered, alpha=0.2, color='#9b59b6')
            best_idx = np.argmax(win_rates_filtered)
            worst_idx = np.argmin(win_rates_filtered)
            ax.scatter([win_rate_episodes[best_idx]], [win_rates_filtered[best_idx]], 
                      color='#2ecc71', s=100, zorder=5, marker='*', label=f'Best: {win_rates_filtered[best_idx]:.1f}%')
            ax.scatter([win_rate_episodes[worst_idx]], [win_rates_filtered[worst_idx]], 
                      color='#e74c3c', s=80, zorder=5, marker='v', label=f'Worst: {win_rates_filtered[worst_idx]:.1f}%')
            ax.legend(loc='lower right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No win rate data', ha='center', va='center', transform=ax.transAxes)
        ax.axhline(y=42, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate Over Training', fontweight='bold')
        ax.set_ylim([0, 70])
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Strategy Score with Forgetting Markers
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(episodes, overall_scores, linewidth=2, color='#2c3e50', marker='o', markersize=4)
        ax.fill_between(episodes, 0, overall_scores, alpha=0.15, color='#3498db')
        
        # Mark peak
        peak_idx = np.argmax(overall_scores)
        ax.scatter([episodes[peak_idx]], [overall_scores[peak_idx]], 
                  color='#2ecc71', s=120, marker='*', zorder=5, label=f'Peak: {overall_scores[peak_idx]:.1f}%')
        
        # Mark forgetting events
        if hasattr(self, 'forgetting_events') and self.forgetting_events:
            for fe in self.forgetting_events:
                ep_idx = next((i for i, r in enumerate(self.analysis_results) if r['episode'] == fe['episode']), None)
                if ep_idx is not None:
                    ax.axvline(x=fe['episode'], color='red', linestyle='--', alpha=0.4)
                    ax.scatter([fe['episode']], [overall_scores[ep_idx]], 
                              color='#e74c3c', s=80, marker='v', zorder=5)
            ax.scatter([], [], color='#e74c3c', marker='v', label=f'Forgetting ({len(self.forgetting_events)})')
        
        ax.axhline(y=80, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Strategy Score (%)')
        ax.set_title('Strategy Score + Forgetting Events', fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
        
        # Plot 3: Basic Skills
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(episodes, stick_20_21, linewidth=2, color='#e74c3c', marker='s', markersize=4, label='Stick 20-21')
        ax.plot(episodes, hit_low, linewidth=2, color='#2ecc71', marker='^', markersize=4, label='Hit 4-11')
        ax.axhline(y=90, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Correctness (%)')
        ax.set_title('Basic Skills Learning', fontweight='bold')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # =================================================================
        # ROW 2: Dealer Awareness (dual axis), Confidence, Usable Ace
        # =================================================================
        
        # Plot 4: Dealer Card Sensitivity (dual axis)
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(episodes, dealer_awareness, linewidth=2, color='#3498db', marker='o', markersize=4, label='Awareness Score')
        ax.fill_between(episodes, 0, dealer_awareness, alpha=0.15, color='#3498db')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Awareness Score (%)', color='#3498db')
        ax.set_ylim([0, 100])
        ax.tick_params(axis='y', labelcolor='#3498db')
        
        ax2 = ax.twinx()
        ax2.plot(episodes, dealer_variance, linewidth=2, color='#e74c3c', marker='s', markersize=4, alpha=0.7, label='Decision Variance')
        ax2.set_ylabel('Decision Variance', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        
        ax.set_title('Dealer Card Sensitivity', fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)
        
        # Plot 5: Decision Confidence
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(episodes, confidence_mean, linewidth=2, color='#c0392b', marker='o', markersize=4)
        ax.fill_between(episodes, 50, confidence_mean, alpha=0.15, color='#c0392b')
        ax.axhline(y=80, color='green', linestyle='--', alpha=0.3, label='Target: 80%')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Confidence (%)')
        ax.set_title('Decision Confidence', fontweight='bold')
        ax.set_ylim([50, 100])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Plot 6: Usable Ace Handling
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(episodes, ace_agg, linewidth=2, color='#9b59b6', marker='D', markersize=5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(episodes, 0, ace_agg, where=[a > 0 for a in ace_agg], 
                       color='#2ecc71', alpha=0.3, label='Good (aggressive w/ ace)')
        ax.fill_between(episodes, 0, ace_agg, where=[a <= 0 for a in ace_agg], 
                       color='#e74c3c', alpha=0.3, label='Bad (passive w/ ace)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Aggression Difference (%)')
        ax.set_title('Usable Ace Handling', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # =================================================================
        # ROW 3: Critical Skills Stability, Forgetting Summary, Correlation
        # =================================================================
        
        # Plot 7: Critical Skills Stability (stick_on_17_19 and rational_12_16)
        ax = fig.add_subplot(gs[2, 0])
        stick_17_19 = [r['basic_strategy'].get('stick_on_17_19', 0) for r in self.analysis_results]
        rational = [r['basic_strategy'].get('rational_12_16', 0) for r in self.analysis_results]
        ax.plot(episodes, stick_17_19, linewidth=2, color='#f39c12', marker='p', markersize=4, label='Stick 17-19')
        ax.plot(episodes, rational, linewidth=2, color='#1abc9c', marker='h', markersize=4, label='Rational 12-16')
        ax.axhline(y=80, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Correctness (%)')
        ax.set_title('Advanced Skills Stability', fontweight='bold')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Plot 8: Forgetting Events Summary (text box)
        ax = fig.add_subplot(gs[2, 1])
        ax.axis('off')
        
        summary_text = "FORGETTING SUMMARY\n" + "="*30 + "\n\n"
        if hasattr(self, 'forgetting_events') and self.forgetting_events:
            summary_text += f"Total Events: {len(self.forgetting_events)}\n\n"
            
            # Count by type
            type_counts = {}
            severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0}
            for fe in self.forgetting_events:
                for event in fe['events']:
                    t = event['type']
                    type_counts[t] = type_counts.get(t, 0) + 1
                    severity_counts[event['severity']] = severity_counts.get(event['severity'], 0) + 1
            
            summary_text += "By Type:\n"
            for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
                summary_text += f"  • {t.replace('_', ' ').title()}: {c}\n"
            
            summary_text += "\nBy Severity:\n"
            for sev in ['CRITICAL', 'HIGH', 'MEDIUM']:
                if severity_counts[sev] > 0:
                    icon = '🔴' if sev == 'CRITICAL' else ('🟡' if sev == 'HIGH' else '🟢')
                    summary_text += f"  {icon} {sev}: {severity_counts[sev]}\n"
        else:
            summary_text += "✅ No forgetting detected!\n\nModel maintained learned\nbehaviors throughout training."
        
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='center', horizontalalignment='center',
               fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))
        ax.set_title('Forgetting Events', fontweight='bold')
        
        # Plot 9: Strategy vs Win Rate Correlation
        ax = fig.add_subplot(gs[2, 2])
        if win_rates_filtered and len(win_rates_filtered) == len(overall_scores):
            colors = np.arange(len(episodes))
            scatter = ax.scatter(overall_scores, win_rates_filtered, c=colors, cmap='viridis', 
                               s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
            # Trend line
            z = np.polyfit(overall_scores, win_rates_filtered, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(overall_scores), max(overall_scores), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.5, label='Trend')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Episode', fontsize=9)
            ax.set_xlabel('Strategy Score (%)')
            ax.set_ylabel('Win Rate (%)')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Strategy vs Win Rate', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Learning Progression & Forgetting Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Learning progression saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    # Keep old method names as aliases for backward compatibility
    def plot_learning_timeline(self, save_path: Optional[str] = None, show: bool = False):
        """Alias for plot_learning_progression (deprecated)."""
        return self.plot_learning_progression(save_path=save_path, show=show)
    
    def plot_forgetting_analysis(self, save_path: Optional[str] = None, show: bool = False):
        """Alias for plot_learning_progression (deprecated)."""
        return self.plot_learning_progression(save_path=save_path, show=show)
    
    def generate_checkpoint_behavior_heatmaps(
        self,
        output_dir: str,
        max_checkpoints: Optional[int] = None,
        create_gif: bool = True,
        gif_duration: float = 0.8
    ) -> Path:
        """
        Generate comprehensive behavior heatmaps for each checkpoint.
        
        Creates a 3x3 grid per checkpoint:
        
        Row 1: Actions & Deviation
            [Model Actions No Ace] [Model Actions Ace] [Deviation from Basic Strategy]
            
        Row 2: Hit Probabilities
            [Hit Prob No Ace] [Hit Prob Ace] [Basic Strategy Reference]
            
        Row 3: Metrics
            [Confidence Map] [Ace Sensitivity] [Metrics Summary]
        
        Also creates:
        - Individual PNG files per checkpoint in 'behavior_checkpoints/' subfolder
        - Optional animated GIF showing behavior evolution
        
        Args:
            output_dir: Directory to save outputs
            max_checkpoints: Maximum number of checkpoints to process (None = all)
            create_gif: Whether to create an animated GIF
            gif_duration: Duration per frame in seconds (for GIF)
            
        Returns:
            Path to the output directory
        """
        from .strategy import BehaviorAnalyzer, get_basic_strategy_action
        
        output_path = Path(output_dir)
        checkpoint_dir = output_path / 'behavior_checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Select checkpoints to process
        checkpoints_to_analyze = self.checkpoints
        if max_checkpoints and len(self.checkpoints) > max_checkpoints:
            step = len(self.checkpoints) // max_checkpoints
            checkpoints_to_analyze = self.checkpoints[::step]
        
        print(f"\n[DATA] Generating behavior heatmaps for {len(checkpoints_to_analyze)} checkpoints...")
        
        # Pre-compute basic strategy maps for comparison
        basic_strategy_no_ace = np.zeros((18, 10))
        basic_strategy_ace = np.zeros((18, 10))
        for i, player_sum in enumerate(range(4, 22)):
            for j, dealer_card in enumerate(range(1, 11)):
                basic_strategy_no_ace[i, j] = get_basic_strategy_action(player_sum, dealer_card, 0)
                basic_strategy_ace[i, j] = get_basic_strategy_action(player_sum, dealer_card, 1)
        
        saved_frames = []
        
        for idx, ckpt in enumerate(checkpoints_to_analyze):
            print(f"  [{idx+1}/{len(checkpoints_to_analyze)}] Episode {ckpt['episode']}...", end=" ")
            
            policy_net = self._load_checkpoint(ckpt)
            analyzer = BehaviorAnalyzer(policy_net)
            
            # Create figure - 3x3 grid for comprehensive view
            fig = plt.figure(figsize=(18, 14))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)
            
            action_cmap = plt.cm.RdYlGn
            
            # Get metrics for this checkpoint
            result = None
            if self.analysis_results:
                result = next((r for r in self.analysis_results if r['episode'] == ckpt['episode']), None)
            
            win_rate = ckpt.get('win_rate', 'N/A')
            strategy_score = result['basic_strategy']['overall_strategy_score'] if result else 'N/A'
            
            # Get decision maps
            decision_map_no_ace = analyzer.create_decision_map(0)
            decision_map_ace = analyzer.create_decision_map(1)
            
            from matplotlib.colors import LinearSegmentedColormap
            deviation_colors = ['#3498db', '#ffffff', '#e74c3c']
            deviation_cmap = LinearSegmentedColormap.from_list('deviation', deviation_colors, N=256)
            
            # Compute deviation maps
            def compute_deviation(model_actions, basic_strategy):
                deviation = np.zeros_like(model_actions)
                correct = 0
                for i in range(18):
                    for j in range(10):
                        if model_actions[i, j] == basic_strategy[i, j]:
                            deviation[i, j] = 0
                            correct += 1
                        elif model_actions[i, j] == 1:
                            deviation[i, j] = 1  # Too aggressive
                        else:
                            deviation[i, j] = -1  # Too passive
                return deviation, correct / 180 * 100
            
            deviation_no_ace, accuracy_no_ace = compute_deviation(decision_map_no_ace['actions'], basic_strategy_no_ace)
            deviation_ace, accuracy_ace = compute_deviation(decision_map_ace['actions'], basic_strategy_ace)
            
            # =================================================================
            # ROW 1: Model Actions (no ace), Model Actions (ace), Deviation Map
            # =================================================================
            
            # Plot 1: Model Actions - No Ace
            ax = fig.add_subplot(gs[0, 0])
            im = ax.imshow(
                decision_map_no_ace['actions'],
                aspect='auto', cmap=action_cmap, origin='lower',
                extent=[0.5, 10.5, 3.5, 21.5], vmin=0, vmax=1
            )
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ax.set_title('Model Actions (No Ace)', fontweight='bold')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75], fraction=0.046)
            cbar.ax.set_yticklabels(['Stand', 'Hit'])
            
            # Plot 2: Model Actions - Usable Ace
            ax = fig.add_subplot(gs[0, 1])
            im = ax.imshow(
                decision_map_ace['actions'],
                aspect='auto', cmap=action_cmap, origin='lower',
                extent=[0.5, 10.5, 3.5, 21.5], vmin=0, vmax=1
            )
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ax.set_title('Model Actions (Usable Ace)', fontweight='bold')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75], fraction=0.046)
            cbar.ax.set_yticklabels(['Stand', 'Hit'])
            
            # Plot 3: Basic Strategy Reference
            ax = fig.add_subplot(gs[0, 2])
            im = ax.imshow(
                basic_strategy_no_ace,
                aspect='auto', cmap=action_cmap, origin='lower',
                extent=[0.5, 10.5, 3.5, 21.5], vmin=0, vmax=1
            )
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ax.set_title('Basic Strategy Reference', fontweight='bold')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75], fraction=0.046)
            cbar.ax.set_yticklabels(['Stand', 'Hit'])
            
            # =================================================================
            # ROW 2: Hit Probabilities (No Ace), Hit Probabilities (Ace), Deviation
            # =================================================================
            
            # Plot 4: Hit Probability - No Ace
            ax = fig.add_subplot(gs[1, 0])
            im = ax.imshow(
                decision_map_no_ace['hit_probs'],
                aspect='auto', cmap=action_cmap, origin='lower',
                extent=[0.5, 10.5, 3.5, 21.5], vmin=0, vmax=1
            )
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ax.set_title('Hit Probability (No Ace)', fontweight='bold')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            cbar = plt.colorbar(im, ax=ax, fraction=0.046)
            cbar.set_label('P(Hit)')
            
            # Plot 5: Hit Probability - Usable Ace
            ax = fig.add_subplot(gs[1, 1])
            im = ax.imshow(
                decision_map_ace['hit_probs'],
                aspect='auto', cmap=action_cmap, origin='lower',
                extent=[0.5, 10.5, 3.5, 21.5], vmin=0, vmax=1
            )
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ax.set_title('Hit Probability (Usable Ace)', fontweight='bold')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            cbar = plt.colorbar(im, ax=ax, fraction=0.046)
            cbar.set_label('P(Hit)')
            
            # Plot 6: Deviation from Basic Strategy (No Ace)
            ax = fig.add_subplot(gs[1, 2])
            im = ax.imshow(
                deviation_no_ace,
                aspect='auto', cmap=deviation_cmap, origin='lower',
                extent=[0.5, 10.5, 3.5, 21.5], vmin=-1, vmax=1
            )
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ax.set_title(f'Deviation (No Ace): {accuracy_no_ace:.0f}% match', fontweight='bold')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            cbar = plt.colorbar(im, ax=ax, fraction=0.046)
            cbar.set_ticks([-1, 0, 1])
            cbar.ax.set_yticklabels(['Passive', '[OK]', 'Aggressive'])
            
            # =================================================================
            # ROW 3: Confidence, Ace Sensitivity, Metrics Summary
            # =================================================================
            
            # Plot 7: Confidence Map
            ax = fig.add_subplot(gs[2, 0])
            im = ax.imshow(
                decision_map_no_ace['confidence'],
                aspect='auto', cmap='viridis', origin='lower',
                extent=[0.5, 10.5, 3.5, 21.5], vmin=0.5, vmax=1
            )
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ax.set_title('Decision Confidence', fontweight='bold')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            cbar = plt.colorbar(im, ax=ax, fraction=0.046)
            cbar.set_label('Confidence')
            
            # Plot 8: Ace Sensitivity (difference between ace/no-ace hit probs)
            ax = fig.add_subplot(gs[2, 1])
            # Compute difference in hit probability: positive = more aggressive with ace
            ace_sensitivity = decision_map_ace['hit_probs'] - decision_map_no_ace['hit_probs']
            im = ax.imshow(
                ace_sensitivity,
                aspect='auto', cmap='coolwarm', origin='lower',
                extent=[0.5, 10.5, 3.5, 21.5], vmin=-0.5, vmax=0.5
            )
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ax.set_title('Ace Sensitivity (Ace - No Ace)', fontweight='bold')
            ax.set_xticks(range(1, 11))
            ax.set_yticks(range(4, 22, 2))
            cbar = plt.colorbar(im, ax=ax, fraction=0.046)
            cbar.set_label('ΔP(Hit)')
            
            # Plot 9: Metrics Summary
            ax = fig.add_subplot(gs[2, 2])
            ax.axis('off')
            
            metrics_text = f"CHECKPOINT METRICS\n{'='*32}\n\n"
            metrics_text += f"Episode: {ckpt['episode']}\n"
            metrics_text += f"Win Rate: {win_rate}%\n" if win_rate != 'N/A' else "Win Rate: N/A\n"
            metrics_text += f"Strategy Score: {strategy_score:.1f}%\n\n" if strategy_score != 'N/A' else "Strategy Score: N/A\n\n"
            
            metrics_text += f"BASIC STRATEGY MATCH\n"
            metrics_text += f"  No Ace: {accuracy_no_ace:.1f}%\n"
            metrics_text += f"  With Ace: {accuracy_ace:.1f}%\n\n"
            
            if result:
                metrics_text += f"BEHAVIORAL METRICS\n"
                metrics_text += f"  Stick on 20-21: {result['basic_strategy']['stick_on_20_21']:.0f}%\n"
                metrics_text += f"  Hit on 4-11: {result['basic_strategy']['hit_on_low_sums']:.0f}%\n"
                metrics_text += f"  Dealer Awareness: {result['dealer_awareness']['dealer_awareness_score']:.0f}%\n"
                metrics_text += f"  Mean Confidence: {result['confidence']['mean_confidence']:.0f}%"
            
            ax.text(0.5, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9))
            
            # Title with checkpoint info
            title = f'Comprehensive Behavior Analysis - Episode {ckpt["episode"]}'
            if win_rate != 'N/A':
                title += f' (Win Rate: {win_rate}%)'
            plt.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
            
            # Save individual PNG
            frame_path = checkpoint_dir / f'behavior_ep{ckpt["episode"]:05d}.png'
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            saved_frames.append(frame_path)
            plt.close()
            
            print("[OK]")
        
        print(f"  Saved {len(saved_frames)} behavior heatmaps to: {checkpoint_dir}")
        
        # Create animated GIF
        if create_gif and len(saved_frames) > 1:
            try:
                from PIL import Image
                
                gif_path = output_path / 'behavior_evolution.gif'
                
                images = []
                for frame_path in saved_frames:
                    img = Image.open(frame_path)
                    images.append(img.copy())
                    img.close()
                
                # Save as GIF
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=int(gif_duration * 1000),  # ms per frame
                    loop=0  # Infinite loop
                )
                
                print(f"  Created animated GIF: {gif_path}")
                
            except ImportError:
                print("  [WARN] PIL/Pillow not available, skipping GIF creation")
            except Exception as e:
                print(f"  [WARN] GIF creation failed: {e}")
        
        return checkpoint_dir

    def save_analysis_results(self, filepath: str):
        """Save analysis results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        print(f"[OK] Learning analysis saved: {filepath}")
    
    def analyze_streak_hypothesis(self, n_simulations: int = 100, episodes_per_sim: int = 100) -> Dict:
        """
        Analyze whether observed win rate drops could be due to unlucky streaks.
        
        Blackjack has ~42-43% theoretical win rate with optimal play. This method:
        1. Simulates random streaks at true win rate to get confidence intervals
        2. Compares observed win rate drops against random variance
        3. Identifies if drops are statistically significant or within expected variance
        
        Key insight: If strategy score INCREASES but win rate DROPS significantly,
        either (a) it's variance, or (b) the strategy metrics don't capture actual performance.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            episodes_per_sim: Episodes per simulation (should match evaluation window)
            
        Returns:
            Dictionary with streak analysis results
        """
        if not self.analysis_results:
            print("[WARN] Run analyze_learning_progression() first!")
            return {}
        
        print("\n" + "="*80)
        print("STREAK HYPOTHESIS ANALYSIS")
        print("="*80)
        
        # Theoretical win rate for Blackjack with decent strategy
        # (basic strategy achieves ~42-43%, random ~28%)
        theoretical_wr = 0.42
        
        # Simulate streak variance
        print(f"\nSimulating {n_simulations} x {episodes_per_sim}-episode runs at {theoretical_wr*100:.0f}% true WR...")
        
        simulated_wrs = []
        for _ in range(n_simulations):
            wins = np.random.binomial(episodes_per_sim, theoretical_wr)
            simulated_wrs.append(wins / episodes_per_sim * 100)
        
        sim_mean = np.mean(simulated_wrs)
        sim_std = np.std(simulated_wrs)
        sim_min = np.min(simulated_wrs)
        sim_max = np.max(simulated_wrs)
        
        # Confidence intervals
        ci_95_low = np.percentile(simulated_wrs, 2.5)
        ci_95_high = np.percentile(simulated_wrs, 97.5)
        ci_99_low = np.percentile(simulated_wrs, 0.5)
        ci_99_high = np.percentile(simulated_wrs, 99.5)
        
        print(f"\nVariance Analysis (Monte Carlo):")
        print(f"  Expected WR: {sim_mean:.1f}% ± {sim_std:.1f}%")
        print(f"  Range: [{sim_min:.0f}%, {sim_max:.0f}%]")
        print(f"  95% CI: [{ci_95_low:.1f}%, {ci_95_high:.1f}%]")
        print(f"  99% CI: [{ci_99_low:.1f}%, {ci_99_high:.1f}%]")
        
        # Analyze observed win rate drops
        print(f"\n" + "-"*40)
        print("Checkpoint Analysis:")
        print("-"*40)
        
        anomalies = []
        
        for i in range(1, len(self.analysis_results)):
            prev = self.analysis_results[i-1]
            curr = self.analysis_results[i]
            
            if prev.get('win_rate') is None or curr.get('win_rate') is None:
                continue
            
            prev_wr = prev['win_rate']
            curr_wr = curr['win_rate']
            wr_change = curr_wr - prev_wr
            
            prev_strategy = prev['basic_strategy']['overall_strategy_score']
            curr_strategy = curr['basic_strategy']['overall_strategy_score']
            strategy_change = curr_strategy - prev_strategy
            
            # Check for anomaly: big WR drop with strategy improvement
            is_anomaly = False
            anomaly_type = None
            
            # Case 1: Win rate dropped but strategy improved
            if wr_change < -10 and strategy_change > 0:
                is_anomaly = True
                anomaly_type = "PARADOX: Strategy ↑ but WR ↓"
            
            # Case 2: Win rate below 95% CI
            if curr_wr < ci_95_low:
                if not is_anomaly:
                    is_anomaly = True
                    anomaly_type = "BELOW 95% CI"
                else:
                    anomaly_type += " + BELOW 95% CI"
            
            # Case 3: Win rate below 99% CI (extreme)
            if curr_wr < ci_99_low:
                anomaly_type = "EXTREME: BELOW 99% CI"
                is_anomaly = True
            
            # Calculate z-score
            z_score = (curr_wr - sim_mean) / sim_std if sim_std > 0 else 0
            
            # Print checkpoint status
            status = "✅" if not is_anomaly else "[WARN] "
            print(f"\n  Ep {curr['episode']:>5}: WR {curr_wr:.0f}% (Δ{wr_change:+.0f}%), "
                  f"Strategy {curr_strategy:.1f}% (Δ{strategy_change:+.1f}%), z={z_score:.2f}")
            
            if is_anomaly:
                print(f"         → {anomaly_type}")
                anomalies.append({
                    'episode': curr['episode'],
                    'win_rate': curr_wr,
                    'win_rate_change': wr_change,
                    'strategy_score': curr_strategy,
                    'strategy_change': strategy_change,
                    'z_score': z_score,
                    'anomaly_type': anomaly_type,
                    'likely_variance': curr_wr >= ci_99_low  # If above 99% CI, could be variance
                })
        
        # Summary
        print(f"\n" + "="*80)
        print("VERDICT:")
        print("="*80)
        
        if not anomalies:
            print("\n✅ No anomalous drops detected. Win rate variations appear within expected variance.")
        else:
            variance_only = [a for a in anomalies if a['likely_variance']]
            true_anomalies = [a for a in anomalies if not a['likely_variance']]
            
            if variance_only:
                print(f"\n🎲 {len(variance_only)} drop(s) likely due to VARIANCE (within 99% CI):")
                for a in variance_only:
                    print(f"   - Ep {a['episode']}: {a['anomaly_type']}")
            
            if true_anomalies:
                print(f"\n[WARN]  {len(true_anomalies)} drop(s) STATISTICALLY SIGNIFICANT (below 99% CI):")
                for a in true_anomalies:
                    print(f"   - Ep {a['episode']}: WR={a['win_rate']:.0f}%, z={a['z_score']:.2f}")
                print("\n   These drops are unlikely to be pure variance. Possible causes:")
                print("   - Strategy metrics don't capture actual decision quality")
                print("   - Gradient instability / learning rate issues")
                print("   - Catastrophic interference in neural network weights")
        
        results = {
            'simulation_params': {
                'n_simulations': n_simulations,
                'episodes_per_sim': episodes_per_sim,
                'theoretical_wr': theoretical_wr
            },
            'variance_stats': {
                'mean': sim_mean,
                'std': sim_std,
                'min': sim_min,
                'max': sim_max,
                'ci_95': (ci_95_low, ci_95_high),
                'ci_99': (ci_99_low, ci_99_high)
            },
            'anomalies': anomalies,
            'verdict': 'variance_likely' if not true_anomalies else 'significant_drops_detected'
        }
        
        print("="*80)
        
        return results
    
    def print_summary(self):
        """Print a summary of the learning progression."""
        if not self.analysis_results:
            return
        
        first = self.analysis_results[0]
        last = self.analysis_results[-1]
        
        print("\n" + "="*80)
        print("LEARNING PROGRESSION SUMMARY")
        print("="*80)
        print(f"\nEpisode Range: {first['episode']} → {last['episode']}")
        print(f"Checkpoints Analyzed: {len(self.analysis_results)}")
        
        print("\n[DATA] Overall Progress:")
        improvement = last['basic_strategy']['overall_strategy_score'] - first['basic_strategy']['overall_strategy_score']
        print(f"  Strategy Score: {first['basic_strategy']['overall_strategy_score']:.1f}% → "
              f"{last['basic_strategy']['overall_strategy_score']:.1f}% (+{improvement:.1f}%)")
        
        print("\n🎓 Skill Development:")
        for key in ['stick_on_20_21', 'hit_on_low_sums', 'rational_12_16']:
            improvement = last['basic_strategy'][key] - first['basic_strategy'][key]
            print(f"  {key}: {first['basic_strategy'][key]:.1f}% → "
                  f"{last['basic_strategy'][key]:.1f}% (+{improvement:.1f}%)")
        
        print("\n" + "="*80)
