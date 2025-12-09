"""
Bypass Experiment Runner for Quantum vs Classical Contribution Analysis.

This module provides tools to definitively determine if the quantum circuit
contributes meaningfully to decision-making by running controlled experiments.

Experiment Phases:
    A) Normal training (baseline)
    B) Freeze quantum, continue training classical only
    C) Compare win rates and strategy quality

If phase B matches or exceeds phase A, the quantum circuit is dead weight.
"""

import torch
import gymnasium as gym
import numpy as np
import json
import copy
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field, asdict
from collections import deque

from ...networks.hybrid import UniversalBlackjackHybridPolicyNetwork, BypassMode
from ...core.agent import A2CAgent
from ...core.config import Config, TrainingConfig
from ..utils import generate_all_blackjack_states, get_basic_strategy_action
from ..strategy import BehaviorAnalyzer
from ..gradient_flow import GradientFlowAnalyzer
from ..signal_noise import SignalNoiseAnalyzer
from ..intermediate import IntermediateOutputAnalyzer


@dataclass
class BypassExperimentConfig:
    """Configuration for bypass experiments."""
    
    # Phase durations (episodes)
    phase_a_episodes: int = 1000      # Normal training
    phase_b_episodes: int = 1000      # Frozen quantum training
    
    # Evaluation
    eval_episodes: int = 100
    eval_frequency: int = 250         # Evaluate every N episodes
    
    # Bypass modes to test (all three by default)
    bypass_modes: List[str] = field(default_factory=lambda: ['zeros', 'encoder', 'noise'])
    
    # Training settings
    learning_rate: float = 0.001
    gamma: float = 0.99
    
    # Output
    save_checkpoints: bool = True
    verbose: bool = True


@dataclass 
class PhaseResult:
    """Results from a single experiment phase."""
    phase_name: str
    episodes_trained: int
    final_win_rate: float
    best_win_rate: float
    avg_reward: float
    win_rate_history: List[float] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BypassTestResult:
    """Results from testing a specific bypass mode."""
    bypass_mode: str
    win_rate: float
    avg_reward: float
    action_distribution: Dict[str, float]  # P(hit), P(stand)
    matches_normal: float  # % of states with same action as normal
    basic_strategy_agreement: float  # % agreement with basic strategy
    kl_divergence: float  # KL divergence from normal mode
    decision_entropy: float  # Average entropy of decisions
    
    def to_dict(self) -> dict:
        return asdict(self)


class BypassExperimentRunner:
    """
    Runs controlled experiments to determine quantum circuit contribution.
    
    Usage:
        runner = BypassExperimentRunner(output_dir='results/bypass_exp')
        results = runner.run_full_experiment()
        runner.print_verdict()
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        config: Optional[BypassExperimentConfig] = None,
        network_factory: Optional[Callable] = None
    ):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory for saving results
            config: Experiment configuration
            network_factory: Optional factory for creating networks (default: standard hybrid)
        """
        self.config = config or BypassExperimentConfig()
        self.output_dir = Path(output_dir) if output_dir else None
        self.network_factory = network_factory or self._default_network_factory
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.phase_a_result: Optional[PhaseResult] = None
        self.phase_b_result: Optional[PhaseResult] = None
        self.bypass_tests: Dict[str, BypassTestResult] = {}
        self.verdict: Optional[str] = None
    
    def _default_network_factory(self):
        """Create default hybrid network."""
        return UniversalBlackjackHybridPolicyNetwork()
    
    def _create_agent(self, policy_net) -> A2CAgent:
        """Create A2C agent with given policy network."""
        from ...networks.hybrid import UniversalBlackjackHybridValueNetwork
        
        value_net = UniversalBlackjackHybridValueNetwork()
        return A2CAgent(
            policy_net=policy_net,
            value_net=value_net,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            seed=self.config.seed
        )
    
    def _evaluate(self, agent: A2CAgent, env: gym.Env, n_episodes: int = 100) -> Tuple[float, float]:
        """Evaluate agent performance."""
        rewards = []
        
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    action = agent.select_action(state, training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
        
        win_rate = sum(1 for r in rewards if r > 0) / len(rewards) * 100
        avg_reward = np.mean(rewards)
        return win_rate, avg_reward
    
    def _evaluate_comprehensive(self, agent: A2CAgent) -> Dict:
        """Comprehensive evaluation using all valid Blackjack states."""
        states = generate_all_blackjack_states()
        policy_net = agent.policy_net
        value_net = agent.value_net
        
        # Collect metrics
        total_value = 0.0
        basic_strategy_matches = 0
        all_probs = []
        all_entropies = []
        
        for state in states:
            with torch.no_grad():
                # Get action probabilities
                probs = policy_net(state).squeeze()
                action = torch.argmax(probs).item()
                
                # Get value estimate
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                value = value_net(state_tensor).item()
                
                # Calculate entropy
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                
                # Check basic strategy
                player_sum, dealer_card, usable_ace = state
                optimal_action = get_basic_strategy_action(player_sum, dealer_card, usable_ace)
                if action == optimal_action:
                    basic_strategy_matches += 1
                
                total_value += value
                all_probs.append(probs.cpu().numpy())
                all_entropies.append(entropy)
        
        return {
            'basic_strategy_agreement': basic_strategy_matches / len(states) * 100,
            'avg_state_value': total_value / len(states),
            'avg_decision_entropy': np.mean(all_entropies),
            'all_probs': np.array(all_probs)
        }
    
    def _train_phase(
        self,
        agent: A2CAgent,
        env: gym.Env,
        n_episodes: int,
        phase_name: str
    ) -> PhaseResult:
        """Train for one phase and return results."""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Phase: {phase_name}")
            print(f"{'='*60}")
        
        win_rate_history = []
        reward_history = []
        recent_rewards = deque(maxlen=100)
        best_win_rate = 0.0
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            agent.reset_episode()
            episode_reward = 0
            
            while True:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.step_update(reward, next_state, done)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            agent.update()
            recent_rewards.append(episode_reward)
            
            # Periodic evaluation
            if (episode + 1) % self.config.eval_frequency == 0:
                win_rate, avg_reward = self._evaluate(agent, env, self.config.eval_episodes)
                win_rate_history.append(win_rate)
                reward_history.append(avg_reward)
                best_win_rate = max(best_win_rate, win_rate)
                
                if self.config.verbose:
                    print(f"  Episode {episode+1}/{n_episodes} | "
                          f"Win Rate: {win_rate:.1f}% | Avg Reward: {avg_reward:.2f}")
        
        # Final evaluation
        final_win_rate, final_avg_reward = self._evaluate(agent, env, self.config.eval_episodes)
        best_win_rate = max(best_win_rate, final_win_rate)
        
        return PhaseResult(
            phase_name=phase_name,
            episodes_trained=n_episodes,
            final_win_rate=final_win_rate,
            best_win_rate=best_win_rate,
            avg_reward=final_avg_reward,
            win_rate_history=win_rate_history,
            reward_history=reward_history
        )
    
    def run_phase_a(self) -> PhaseResult:
        """
        Phase A: Normal training (baseline).
        Train hybrid network with all components learning.
        """
        env = gym.make('Blackjack-v1', sab=True)
        policy_net = self.network_factory()
        agent = self._create_agent(policy_net)
        
        result = self._train_phase(
            agent, env, 
            self.config.phase_a_episodes,
            "A: Normal Training (Baseline)"
        )
        
        self.phase_a_result = result
        self._phase_a_agent = agent  # Keep for phase B
        
        if self.output_dir and self.config.save_checkpoints:
            path = self.output_dir / "phase_a_model.pth"
            agent.save(str(path))
        
        env.close()
        return result
    
    def run_phase_b(self) -> PhaseResult:
        """
        Phase B: Freeze quantum, continue training classical only.
        If performance matches phase A, quantum is not contributing.
        """
        if not hasattr(self, '_phase_a_agent'):
            raise RuntimeError("Must run phase A before phase B")
        
        env = gym.make('Blackjack-v1', sab=True)
        agent = self._phase_a_agent
        
        # Freeze quantum weights
        agent.policy_net.freeze_component('quantum')
        
        if self.config.verbose:
            frozen = agent.policy_net.get_frozen_components()
            print(f"\n[ATOM]  Frozen components: {frozen}")
        
        result = self._train_phase(
            agent, env,
            self.config.phase_b_episodes,
            "B: Quantum Frozen (Classical Only)"
        )
        
        self.phase_b_result = result
        self._phase_b_agent = agent  # Store Phase B agent for analysis
        
        if self.output_dir and self.config.save_checkpoints:
            path = self.output_dir / "phase_b_model.pth"
            agent.save(str(path))
        
        env.close()
        return result
    
    def run_bypass_tests(self) -> Dict[str, BypassTestResult]:
        """
        Test different bypass modes on the trained model.
        Compares quantum circuit output vs zeros/encoder/noise.
        """
        if not hasattr(self, '_phase_a_agent'):
            raise RuntimeError("Must run phase A before bypass tests")
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Bypass Mode Tests")
            print(f"{'='*60}")
        
        env = gym.make('Blackjack-v1', sab=True)
        agent = self._phase_a_agent
        policy_net = agent.policy_net
        
        # Get normal mode baseline with comprehensive evaluation
        policy_net.set_bypass_mode('none')
        normal_win_rate, normal_reward = self._evaluate(agent, env, self.config.eval_episodes)
        normal_actions = self._get_action_distribution(policy_net)
        normal_comprehensive = self._evaluate_comprehensive(agent)
        normal_probs = normal_comprehensive['all_probs']
        
        results = {}
        
        for mode in self.config.bypass_modes:
            if self.config.verbose:
                print(f"\n  Testing bypass mode: {mode}")
            
            policy_net.set_bypass_mode(mode)
            win_rate, avg_reward = self._evaluate(agent, env, self.config.eval_episodes)
            action_dist = self._get_action_distribution(policy_net)
            matches = self._compare_actions(policy_net, normal_actions)
            
            # Comprehensive evaluation
            comprehensive = self._evaluate_comprehensive(agent)
            
            # Calculate KL divergence from normal mode
            bypass_probs = comprehensive['all_probs']
            kl_div = self._kl_divergence(normal_probs, bypass_probs)
            
            result = BypassTestResult(
                bypass_mode=mode,
                win_rate=win_rate,
                avg_reward=avg_reward,
                action_distribution=action_dist,
                matches_normal=matches,
                basic_strategy_agreement=comprehensive['basic_strategy_agreement'],
                kl_divergence=kl_div,
                decision_entropy=comprehensive['avg_decision_entropy']
            )
            results[mode] = result
            
            if self.config.verbose:
                print(f"    Win Rate: {win_rate:.1f}% (normal: {normal_win_rate:.1f}%)")
                print(f"    Action Match: {matches:.1f}%")
                print(f"    Basic Strategy Agreement: {comprehensive['basic_strategy_agreement']:.1f}%")
                print(f"    KL Divergence: {kl_div:.4f}")
        
        # Reset to normal mode
        policy_net.set_bypass_mode('none')
        
        self.bypass_tests = results
        env.close()
        return results
    
    def _get_action_distribution(self, policy_net) -> Dict[str, float]:
        """Get average action probabilities across all Blackjack states."""
        total_hit = 0.0
        total_stand = 0.0
        n_states = 0
        
        for player_sum in range(4, 22):
            for dealer_card in range(1, 11):
                for usable_ace in [False, True]:
                    state = (player_sum, dealer_card, int(usable_ace))
                    with torch.no_grad():
                        probs = policy_net(state)
                    total_stand += probs[0, 0].item()
                    total_hit += probs[0, 1].item()
                    n_states += 1
        
        return {
            'P(stand)': total_stand / n_states,
            'P(hit)': total_hit / n_states
        }
    
    def _compare_actions(self, policy_net, normal_actions: Dict) -> float:
        """Compare action choices between current bypass mode and normal."""
        # Store current mode
        current_mode = policy_net.get_bypass_mode()
        
        matches = 0
        total = 0
        
        for player_sum in range(4, 22):
            for dealer_card in range(1, 11):
                for usable_ace in [False, True]:
                    state = (player_sum, dealer_card, int(usable_ace))
                    
                    # Get bypass mode action
                    with torch.no_grad():
                        bypass_probs = policy_net(state)
                    bypass_action = bypass_probs.argmax(dim=-1).item()
                    
                    # Get normal mode action
                    policy_net.set_bypass_mode('none')
                    with torch.no_grad():
                        normal_probs = policy_net(state)
                    normal_action = normal_probs.argmax(dim=-1).item()
                    policy_net.set_bypass_mode(current_mode)
                    
                    if bypass_action == normal_action:
                        matches += 1
                    total += 1
        
        return matches / total * 100
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """Calculate average KL divergence KL(P||Q) across all states."""
        # Ensure probabilities are valid
        p = np.clip(p, epsilon, 1.0)
        q = np.clip(q, epsilon, 1.0)
        
        # Normalize
        p = p / p.sum(axis=1, keepdims=True)
        q = q / q.sum(axis=1, keepdims=True)
        
        # KL(P||Q) = sum(P * log(P/Q))
        kl = (p * np.log(p / q)).sum(axis=1)
        return float(np.mean(kl))
    
    def run_full_experiment(self, run_full_analysis: bool = True) -> Dict:
        """
        Run complete bypass experiment with all phases.
        
        Args:
            run_full_analysis: If True, run comprehensive analysis on both phase models
        
        Returns:
            Full experiment results dictionary
        """
        if self.config.verbose:
            print("\n" + "="*70)
            print("[TEST] BYPASS EXPERIMENT: Quantum vs Classical Contribution")
            print("="*70)
        
        # Phase A: Normal training
        self.run_phase_a()
        
        # Bypass tests on phase A model
        self.run_bypass_tests()
        
        # Phase B: Frozen quantum
        self.run_phase_b()
        
        # Run comprehensive analysis if requested
        if run_full_analysis:
            self._run_comprehensive_analysis()
        
        # Generate verdict
        self._generate_verdict()
        
        # Save results
        results = self.get_results()
        if self.output_dir:
            with open(self.output_dir / "bypass_experiment_results.json", 'w') as f:
                json.dump(results, f, indent=2)
        
        if self.config.verbose:
            self.print_verdict()
        
        return results
    
    def _generate_verdict(self):
        """Generate verdict based on comprehensive experiment results."""
        if not self.phase_a_result or not self.phase_b_result:
            self.verdict = "INCOMPLETE: Missing phase results"
            return
        
        a_wr = self.phase_a_result.final_win_rate
        b_wr = self.phase_b_result.final_win_rate
        wr_diff = b_wr - a_wr
        
        # Analyze bypass test results using multiple metrics
        bypass_findings = []
        high_kl_modes = []  # Modes that differ significantly from normal
        low_kl_modes = []   # Modes very similar to normal
        
        for mode, result in self.bypass_tests.items():
            # Low KL divergence (<0.05) means bypass mode produces similar decisions
            if result.kl_divergence < 0.05:
                low_kl_modes.append(f"{mode}(KL={result.kl_divergence:.4f})")
            elif result.kl_divergence > 0.2:
                high_kl_modes.append(f"{mode}(KL={result.kl_divergence:.4f})")
            
            # Check if bypass mode maintains similar quality
            if result.matches_normal > 90 and abs(result.basic_strategy_agreement - 
                   self.bypass_tests.get('zeros', result).basic_strategy_agreement) < 5:
                bypass_findings.append(
                    f"{mode}: {result.matches_normal:.1f}% action match, "
                    f"{result.basic_strategy_agreement:.1f}% basic strategy agreement"
                )
        
        # Generate verdict based on multiple signals
        signals = []
        
        # Signal 1: Phase comparison (unreliable in Blackjack, so weight it less)
        if abs(wr_diff) < 5:
            signals.append("Phase B win rate similar to Phase A (±5%)")
        elif wr_diff >= 5:
            signals.append(f"Phase B OUTPERFORMED Phase A by {wr_diff:.1f}%")
        elif wr_diff < -10:
            signals.append(f"Phase B significantly worse ({wr_diff:.1f}%)")
        
        # Signal 2: Bypass mode KL divergence (more reliable)
        if low_kl_modes:
            signals.append(f"Low KL divergence bypass modes: {', '.join(low_kl_modes)}")
        if high_kl_modes:
            signals.append(f"High KL divergence (quantum changes behavior): {', '.join(high_kl_modes)}")
        
        # Signal 3: Action consistency
        avg_action_match = np.mean([r.matches_normal for r in self.bypass_tests.values()])
        if avg_action_match > 85:
            signals.append(f"High action consistency ({avg_action_match:.1f}% match)")
        
        # Generate overall verdict
        quantum_is_dead_weight = (
            len(low_kl_modes) >= 2 or  # Multiple bypass modes produce similar output
            (wr_diff >= 0 and avg_action_match > 85)  # Phase B better AND consistent
        )
        
        quantum_is_essential = (
            len(high_kl_modes) >= 2 and  # Multiple bypass modes differ significantly
            wr_diff < -10  # Phase B significantly worse
        )
        
        if quantum_is_dead_weight:
            self.verdict = (
                f"[WARN]  QUANTUM LIKELY DEAD WEIGHT\n"
                f"Evidence:\n  • {chr(10).join('  • ' + s for s in signals)}\n"
                f"Interpretation: Bypassing quantum circuit has minimal impact on decision quality. "
                f"Classical encoder/postprocessing may be doing all meaningful computation."
            )
        elif quantum_is_essential:
            self.verdict = (
                f"[OK] QUANTUM CONTRIBUTES MEANINGFULLY\n"
                f"Evidence:\n  • {chr(10).join('  • ' + s for s in signals)}\n"
                f"Interpretation: Removing quantum circuit significantly changes behavior. "
                f"Quantum provides unique transformations."
            )
        else:
            self.verdict = (
                f"❓ QUANTUM CONTRIBUTION UNCLEAR\n"
                f"Evidence:\n  • {chr(10).join('  • ' + s for s in signals)}\n"
                f"Interpretation: Mixed signals. May need longer training or different hyperparameters. "
                f"Consider gradient flow analysis and signal-to-noise ratio."
            )
        
        # Add specific bypass findings
        if bypass_findings:
            self.verdict += f"\n\nBypass Mode Details:\n  • " + "\n  • ".join(bypass_findings)
    
    def _run_comprehensive_analysis(self):
        """Run full analysis suite on both phase A and phase B models."""
        if not self.output_dir:
            if self.config.verbose:
                print("\n[WARN]  Skipping comprehensive analysis: no output_dir specified")
            return
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Running Comprehensive Analysis")
            print(f"{'='*60}")
        
        # Create subdirectories
        phase_a_dir = self.output_dir / "phase_a_analysis"
        phase_b_dir = self.output_dir / "phase_b_analysis"
        phase_a_dir.mkdir(exist_ok=True)
        phase_b_dir.mkdir(exist_ok=True)
        
        # Analyze Phase A (normal training)
        if hasattr(self, '_phase_a_agent'):
            if self.config.verbose:
                print("\n  Analyzing Phase A (Normal Training)...")
            self._analyze_phase(
                self._phase_a_agent,
                phase_a_dir,
                "Phase A: Normal Training",
                analyze_quantum=True
            )
        
        # Analyze Phase B (frozen quantum)
        if hasattr(self, '_phase_b_agent'):
            if self.config.verbose:
                print("\n  Analyzing Phase B (Quantum Frozen)...")
            self._analyze_phase(
                self._phase_b_agent if hasattr(self, '_phase_b_agent') else self._phase_a_agent,
                phase_b_dir,
                "Phase B: Quantum Frozen",
                analyze_quantum=False  # Quantum was frozen, so quantum analysis less relevant
            )
        
        if self.config.verbose:
            print(f"\n  [OK] Analysis complete. Results saved to {self.output_dir}")
    
    def _analyze_phase(
        self,
        agent: A2CAgent,
        output_dir: Path,
        phase_name: str,
        analyze_quantum: bool = True
    ):
        """Run all analyzers on a specific phase."""
        policy_net = agent.policy_net
        value_net = agent.value_net
        
        # 1. Behavior Analysis
        try:
            behavior_analyzer = BehaviorAnalyzer(policy_net)
            
            # Try to generate report if possible
            try:
                behavior_analyzer.generate_full_report(output_dir=str(output_dir), show=False)
                if self.config.verbose:
                    print(f"    [OK] Behavior analysis complete")
            except:
                # Fallback: just note that analysis was attempted
                if self.config.verbose:
                    print(f"    [WARN] Behavior analysis skipped (method compatibility)")
        except Exception as e:
            if self.config.verbose:
                print(f"    [WARN]  Behavior analysis failed: {e}")
        
        # 2. Quantum-specific analysis (only if hybrid network)
        if analyze_quantum and isinstance(policy_net, UniversalBlackjackHybridPolicyNetwork):
            # Gradient Flow Analysis
            try:
                gradient_analyzer = GradientFlowAnalyzer(policy_net, value_net)
                grad_results = gradient_analyzer.analyze_gradient_flow(n_samples=100)
                
                with open(output_dir / "gradient_flow_analysis.json", 'w') as f:
                    json.dump(grad_results, f, indent=2)
                
                if self.config.verbose:
                    print(f"    [OK] Gradient flow: quantum contributes "
                          f"{grad_results.get('quantum_contribution_pct', 0):.1f}% of gradients")
            except Exception as e:
                if self.config.verbose:
                    print(f"    [WARN]  Gradient flow analysis failed: {e}")
            
            # Signal-to-Noise Analysis
            try:
                signal_analyzer = SignalNoiseAnalyzer(policy_net)
                signal_results = signal_analyzer.analyze_all()
                
                with open(output_dir / "signal_noise_analysis.json", 'w') as f:
                    json.dump(signal_results, f, indent=2)
                
                if self.config.verbose:
                    snr = signal_results.get('quantum_snr', {}).get('overall_snr', 0)
                    print(f"    [OK] Signal-to-noise: {snr:.2f} dB")
            except Exception as e:
                if self.config.verbose:
                    print(f"    [WARN]  Signal-noise analysis failed: {e}")
            
            # Intermediate Output Analysis
            try:
                intermediate_analyzer = IntermediateOutputAnalyzer(policy_net)
                intermediate_results = intermediate_analyzer.analyze_all_stages()
                
                with open(output_dir / "intermediate_analysis.json", 'w') as f:
                    json.dump(intermediate_results, f, indent=2)
                
                if self.config.verbose:
                    print(f"    [OK] Intermediate analysis complete")
            except Exception as e:
                if self.config.verbose:
                    print(f"    [WARN]  Intermediate analysis failed: {e}")
    
    def get_results(self) -> Dict:
        """Get all experiment results as dictionary."""
        return {
            'config': asdict(self.config),
            'phase_a': self.phase_a_result.to_dict() if self.phase_a_result else None,
            'phase_b': self.phase_b_result.to_dict() if self.phase_b_result else None,
            'bypass_tests': {k: v.to_dict() for k, v in self.bypass_tests.items()},
            'verdict': self.verdict
        }
    
    def print_verdict(self):
        """Print experiment verdict with summary."""
        print("\n" + "="*70)
        print("[DATA] BYPASS EXPERIMENT VERDICT")
        print("="*70)
        
        if self.phase_a_result and self.phase_b_result:
            print(f"\n  Phase A (Normal):         {self.phase_a_result.final_win_rate:.1f}% win rate")
            print(f"  Phase B (Quantum Frozen): {self.phase_b_result.final_win_rate:.1f}% win rate")
            print(f"  Difference:               {self.phase_b_result.final_win_rate - self.phase_a_result.final_win_rate:+.1f}%")
        
        if self.bypass_tests:
            print(f"\n  Bypass Test Results:")
            print(f"  {'Mode':<10} {'WR%':<8} {'Act%':<8} {'BS%':<8} {'KL Div':<10} {'Entropy':<8}")
            print(f"  {'-'*60}")
            for mode, result in self.bypass_tests.items():
                print(f"  {mode:<10} {result.win_rate:<8.1f} {result.matches_normal:<8.1f} "
                      f"{result.basic_strategy_agreement:<8.1f} {result.kl_divergence:<10.4f} "
                      f"{result.decision_entropy:<8.4f}")
            print(f"\n  Legend: WR%=Win Rate, Act%=Action Match w/ Normal, BS%=Basic Strategy Agreement")
        
        print(f"\n  🎯 VERDICT:\n{self.verdict}")
        print("="*70)


class QuantumFirstTrainer:
    """
    Implements quantum-first training strategy.
    
    Trains quantum weights first with classical components frozen,
    forcing the circuit to develop meaningful rotations before
    classical components can compensate.
    """
    
    def __init__(
        self,
        quantum_first_episodes: int = 1000,
        normal_episodes: int = 4000,
        quantum_dropout_rate: float = 0.0,
        verbose: bool = True
    ):
        """
        Initialize quantum-first trainer.
        
        Args:
            quantum_first_episodes: Episodes to train quantum-only (default: 1000)
            normal_episodes: Episodes for normal training after (default: 4000)
            quantum_dropout_rate: Rate of random quantum weight freezing (0.0-1.0)
            verbose: Print progress
        """
        self.quantum_first_episodes = quantum_first_episodes
        self.normal_episodes = normal_episodes
        self.quantum_dropout_rate = quantum_dropout_rate
        self.verbose = verbose
    
    def configure_network(self, policy_net: UniversalBlackjackHybridPolicyNetwork):
        """
        Configure network for quantum-first training.
        
        Call this before starting training to set up the freeze schedule.
        """
        # Set quantum dropout if enabled
        if self.quantum_dropout_rate > 0:
            policy_net.set_quantum_dropout(self.quantum_dropout_rate)
            if self.verbose:
                print(f"[ATOM]  Quantum dropout enabled: {self.quantum_dropout_rate:.0%}")
        
        # Freeze classical components for quantum-first phase
        policy_net.freeze_component('classical')
        
        if self.verbose:
            frozen = policy_net.get_frozen_components()
            print(f"[LOCK] Frozen for quantum-first phase: {frozen}")
    
    def on_episode(
        self,
        episode: int,
        policy_net: UniversalBlackjackHybridPolicyNetwork
    ):
        """
        Call this after each training episode to manage freeze schedule.
        
        Args:
            episode: Current episode number (0-indexed)
            policy_net: The policy network being trained
        """
        # Transition from quantum-first to normal training
        if episode == self.quantum_first_episodes:
            policy_net.unfreeze_component('all')
            
            if self.verbose:
                print(f"\n🔓 Episode {episode}: Unfreezing all components")
                print(f"   Transitioning to normal training for {self.normal_episodes} episodes")
    
    def get_total_episodes(self) -> int:
        """Get total episodes for quantum-first + normal training."""
        return self.quantum_first_episodes + self.normal_episodes
