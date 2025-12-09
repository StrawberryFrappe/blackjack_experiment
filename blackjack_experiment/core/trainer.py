"""
Training loop and evaluation for Blackjack A2C experiments.

This module provides:
- Trainer: Main training loop with checkpoint saving and analysis
- evaluate(): Policy evaluation function
- create_networks(): Factory function to create network pairs
- create_agent(): Factory function to create A2C agent
"""

import gymnasium as gym
import numpy as np
import copy
import torch
from typing import Optional, Callable, Tuple, List, Dict, Any
from collections import deque
from pathlib import Path

from .agent import A2CAgent
from .config import Config, TrainingConfig
from .session import SessionManager


# =============================================================================
# NETWORK FACTORY FUNCTIONS
# =============================================================================

def create_networks(network_type: str, config: Optional[Config] = None) -> Tuple:
    """
    Create policy and value networks based on network type.
    
    Args:
        network_type: 'classical', 'minimal_classical', or 'hybrid'
        config: Optional config for network settings
        
    Returns:
        Tuple of (policy_net, value_net)
    """
    if network_type == 'classical':
        from ..networks.classical import (
            BlackjackClassicalPolicyNetwork,
            BlackjackClassicalValueNetwork
        )
        hidden_sizes = config.network.hidden_sizes if config else [4, 6, 8, 4]
        activation = config.network.activation if config else 'tanh'
        
        policy_net = BlackjackClassicalPolicyNetwork(
            hidden_sizes=hidden_sizes,
            activation=activation
        )
        value_net = BlackjackClassicalValueNetwork(
            hidden_sizes=[32, 16, 8],
            activation=activation
        )
        
    elif network_type == 'minimal_classical':
        from ..networks.classical import (
            BlackjackMinimalClassicalPolicyNetwork,
            BlackjackMinimalClassicalValueNetwork
        )
        policy_net = BlackjackMinimalClassicalPolicyNetwork()
        value_net = BlackjackMinimalClassicalValueNetwork()
        
    elif network_type == 'hybrid':
        from ..networks.hybrid import (
            UniversalBlackjackHybridPolicyNetwork,
            UniversalBlackjackHybridValueNetwork
        )
        # Use class defaults - network design is owned by hybrid/config.py
        policy_net = UniversalBlackjackHybridPolicyNetwork()
        value_net = UniversalBlackjackHybridValueNetwork()
        
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    return policy_net, value_net


def create_agent(
    policy_net,
    value_net,
    config: Optional[Config] = None,
    seed: Optional[int] = None
) -> A2CAgent:
    """
    Create A2C agent with networks.
    
    Args:
        policy_net: Policy network
        value_net: Value network
        config: Optional config for agent settings
        
    Returns:
        A2CAgent instance
    """
    if config is None:
        config = Config()
    
    return A2CAgent(
        policy_net=policy_net,
        value_net=value_net,
        learning_rate=config.agent.learning_rate,
        gamma=config.agent.gamma,
        entropy_coef=config.agent.entropy_coef,
        value_coef=config.agent.value_coef,
        max_grad_norm=config.agent.max_grad_norm,
        n_steps=config.agent.n_steps,
        use_gae=config.agent.use_gae,
        gae_lambda=config.agent.gae_lambda,
        use_encoding_diversity=config.network.use_encoding_diversity,
        encoding_diversity_coef=config.network.encoding_diversity_coef,
        seed=seed,
        encoder_lr=config.agent.encoder_lr
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(
    agent: A2CAgent,
    env: gym.Env,
    n_episodes: int = 100,
    max_steps: int = 200,
    verbose: bool = True
) -> Tuple[List[float], List[int], float]:
    """
    Evaluate the agent's performance on Blackjack.
    
    Args:
        agent: A2C agent to evaluate
        env: Gym environment
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        verbose: Print progress
        
    Returns:
        Tuple of (eval_rewards, eval_lengths, win_rate)
    """
    eval_rewards = []
    eval_lengths = []
    
    if verbose:
        print("\n" + "=" * 60)
        print("Evaluating Policy")
        print("=" * 60)
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            with torch.no_grad():
                action = agent.select_action(state, training=False)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if steps >= max_steps:
                done = True
            
            if done:
                break
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(steps)
        
        if verbose and (episode + 1) % 10 == 0:
            outcome = "WIN" if episode_reward > 0 else ("LOSS" if episode_reward < 0 else "DRAW")
            print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}, Steps = {steps}, Outcome = {outcome}")
    
    # Calculate statistics
    if verbose:
        avg_reward = np.mean(eval_rewards)
        wins = sum(1 for r in eval_rewards if r > 0)
        losses = sum(1 for r in eval_rewards if r < 0)
        draws = sum(1 for r in eval_rewards if r == 0)
        
        print("=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Win Rate:  {wins/n_episodes*100:.1f}% ({wins}/{n_episodes})")
        print(f"Loss Rate: {losses/n_episodes*100:.1f}% ({losses}/{n_episodes})")
        print(f"Draw Rate: {draws/n_episodes*100:.1f}% ({draws}/{n_episodes})")
        print("=" * 60)
    
    win_rate = sum(1 for r in eval_rewards if r > 0) / len(eval_rewards) * 100
    return eval_rewards, eval_lengths, win_rate


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Main training loop for A2C agent on Blackjack."""
    
    def __init__(
        self,
        agent: A2CAgent,
        env: gym.Env,
        config: TrainingConfig,
        session_manager: Optional[SessionManager] = None,
        enable_analysis: bool = True,
        verbose: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            agent: A2C agent
            env: Gym environment
            config: Training configuration
            session_manager: Session manager for output organization
            enable_analysis: Enable post-training analysis
            verbose: Print detailed progress
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.session_manager = session_manager
        self.enable_analysis = enable_analysis and session_manager is not None
        self.verbose = verbose
        
        # config is a TrainingConfig object
        self.analysis_frequency = config.save_frequency
        
        # Detect Blackjack environment
        env_name = getattr(env, 'spec', None)
        if env_name:
            self.is_blackjack = 'Blackjack' in str(env_name.id)
        else:
            self.is_blackjack = hasattr(env, 'env') and 'Blackjack' in str(type(env.env))
        
        # Setup gradient analyzer for hybrid networks
        self.gradient_analyzer = self._setup_gradient_analyzer()
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
        self.checkpoint_episodes = []
    
    def _setup_gradient_analyzer(self):
        """Setup hybrid gradient analyzer if network is hybrid."""
        policy_net = self.agent.policy_net
        
        is_hybrid = (hasattr(policy_net, 'feature_encoder') and
                    hasattr(policy_net, 'quantum_circuit') and
                    hasattr(policy_net, 'postprocessing'))
        
        if not is_hybrid:
            return None
        
        n_qubits = getattr(policy_net, 'n_qubits', None)
        if n_qubits is None:
            return None
        
        try:
            from ..analysis.gradient_flow import GradientFlowAnalyzer as HybridGradientAnalyzer
            analyzer = HybridGradientAnalyzer(policy_net, n_qubits=n_qubits)
            print(f"[OK] Hybrid gradient analyzer initialized ({n_qubits}-qubit network)")
            return analyzer
        except Exception as e:
            print(f"[WARN] Failed to initialize gradient analyzer: {e}")
            return None
    
    def train(self, start_episode: int = 0) -> Tuple[List[float], List[int], List[Dict]]:
        """
        Train the agent.
        
        Args:
            start_episode: Episode to start from (for resuming)
            
        Returns:
            Tuple of (episode_rewards, episode_lengths, training_metrics)
        """
        # Save session configuration at start of training
        if self.session_manager and start_episode == 0:
            self._save_session_config()
        
        if self.verbose:
            print("=" * 60)
            print("Starting Training" if start_episode == 0 else f"Resuming from Episode {start_episode + 1}")
            print("=" * 60)
            print(f"Episodes: {start_episode + 1} to {self.config.n_episodes}")
            print(f"Max steps per episode: {self.config.max_steps_per_episode}")
            print(f"Gamma: {self.agent.gamma}")
            print("=" * 60)
        
        # Microwaved encoder logic
        microwaved_unfreeze_episode = 0
        if getattr(self.config, 'microwaved_encoder', False) and hasattr(self.agent.policy_net, 'freeze_component'):
            microwaved_unfreeze_episode = int(self.config.n_episodes * getattr(self.config, 'microwaved_fraction', 0.5))
            if start_episode < microwaved_unfreeze_episode:
                if self.verbose:
                    print(f"MICROWAVED ENCODER: Freezing encoder until episode {microwaved_unfreeze_episode}")
                self.agent.policy_net.freeze_component('encoder')
            else:
                if self.verbose:
                    print(f"MICROWAVED ENCODER: Starting with unfrozen encoder (past episode {microwaved_unfreeze_episode})")
                self.agent.policy_net.unfreeze_component('encoder')

        recent_rewards = deque(maxlen=100)
        
        for episode in range(start_episode, self.config.n_episodes):
            # Check for unfreeze
            if getattr(self.config, 'microwaved_encoder', False) and episode == microwaved_unfreeze_episode:
                if hasattr(self.agent.policy_net, 'unfreeze_component'):
                    if self.verbose:
                        print(f"MICROWAVED ENCODER: Unfreezing encoder at episode {episode}")
                    self.agent.policy_net.unfreeze_component('encoder')

            state, _ = self.env.reset()
            self.agent.reset_episode()
            episode_reward = 0
            step_metrics_list = []
            
            for step in range(self.config.max_steps_per_episode):
                action = self.agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                step_metrics = self.agent.step_update(reward, next_state, done)
                if step_metrics:
                    step_metrics_list.append(step_metrics)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update agent
            metrics = self.agent.update()
            
            # Capture gradient metrics
            if self.gradient_analyzer:
                try:
                    grad_metrics = self.gradient_analyzer.capture_gradients()
                    if grad_metrics and metrics:
                        metrics.update(grad_metrics)
                except Exception:
                    pass
            
            # Aggregate step metrics
            if step_metrics_list and metrics:
                avg_metrics = {
                    k: sum(m.get(k, 0) for m in step_metrics_list) / len(step_metrics_list)
                    for k in step_metrics_list[0].keys()
                }
                metrics.update(avg_metrics)
            
            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            recent_rewards.append(episode_reward)
            
            if metrics:
                self.training_metrics.append(metrics)
            
            # Print progress
            if (episode + 1) % self.config.print_every == 0:
                self._print_progress(episode + 1, recent_rewards)
            
            # Save checkpoint
            if self.config.save_model and (episode + 1) % self.config.save_frequency == 0:
                self._save_checkpoint(episode + 1, recent_rewards)
        
        # Final save
        if self.config.save_model:
            self._save_final_model()
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Training Completed!")
            print("=" * 60)
        
        # Save gradient analysis
        if self.gradient_analyzer is not None:
            self._save_gradient_analysis()
        
        # Run post-training analysis
        if self.enable_analysis and self.is_blackjack:
            self._run_post_training_analysis()
        
        return self.episode_rewards, self.episode_lengths, self.training_metrics
    
    def _print_progress(self, episode: int, recent_rewards: deque):
        """Print training progress."""
        avg_reward = np.mean(recent_rewards)
        
        if self.is_blackjack:
            win_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards) * 100
            print(f"Episode {episode}/{self.config.n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | Win Rate: {win_rate:.1f}%")
        else:
            print(f"Episode {episode}/{self.config.n_episodes} | Avg Reward: {avg_reward:.2f}")
    
    def _save_checkpoint(self, episode: int, recent_rewards: deque):
        """Save training checkpoint."""
        if self.session_manager:
            if self.is_blackjack and len(recent_rewards) >= 100:
                win_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards) * 100
                path = self.session_manager.get_checkpoint_path(episode, win_rate)
            else:
                path = self.session_manager.get_checkpoint_path(episode)
            
            self.agent.save(str(path))
            self.checkpoint_episodes.append(episode)
            
            if self.verbose:
                print(f"  Checkpoint saved: {path.name}")
    
    def _save_final_model(self):
        """Save final trained model."""
        if self.session_manager:
            if self.is_blackjack and len(self.episode_rewards) >= 100:
                win_rate = sum(1 for r in self.episode_rewards[-100:] if r > 0)
                avg_reward = np.mean(self.episode_rewards[-100:])
                path = self.session_manager.get_final_model_path(win_rate, avg_reward)
            else:
                path = self.session_manager.get_final_model_path()
            
            self.agent.save(str(path))
            print(f"\nFinal model saved: {path}")
    
    def _save_gradient_analysis(self):
        """Save gradient analysis for hybrid networks."""
        if self.gradient_analyzer is None or self.session_manager is None:
            return
        
        try:
            print("\n" + "=" * 80)
            print("[DATA] GRADIENT ANALYSIS - Hybrid Network")
            print("=" * 80)
            
            self.gradient_analyzer.print_summary()
            
            plot_path = self.session_manager.get_plot_path('gradient_analysis')
            self.gradient_analyzer.plot_gradient_analysis(save_path=str(plot_path), show=False)
            print(f"[OK] Saved: {plot_path.name}")
            
            json_path = self.gradient_analyzer.save_analysis(self.session_manager.session_dir)
            print(f"[OK] Saved: {json_path.name}")
            
            print("=" * 80 + "\n")
        except Exception as e:
            print(f"\n[WARN] Gradient analysis failed: {e}")
    
    def _run_post_training_analysis(self):
        """Run comprehensive post-training analysis."""
        print("\n" + "=" * 80)
        print("[TEST] RUNNING POST-TRAINING ANALYSIS")
        print("=" * 80)
        
        try:
            from ..analysis.learning import LearningAnalyzer
            from ..analysis.strategy import BehaviorAnalyzer, create_blackjack_test_states
            from ..analysis.capacity import CapacityAnalyzer
            
            # Network capacity analysis (shows architectural constraints)
            print("[BUILD]  Analyzing network capacity and architecture...")
            capacity_analyzer = CapacityAnalyzer(self.agent.policy_net)
            capacity_analyzer.print_report()
            
            # Save capacity visualizations (skip if no hybrid network)
            if hasattr(self.agent.policy_net, 'weights'):
                try:
                    capacity_plot = self.session_manager.get_plot_path("capacity_analysis")
                    capacity_analyzer.visualize_architecture(save_path=str(capacity_plot), show=False)
                except Exception as e:
                    print(f"   [WARN] Could not create capacity visualization: {e}")
            
            capacity_json = self.session_manager.get_plot_path("capacity_analysis", ".json")
            capacity_analyzer.save_analysis(str(self.session_manager.session_dir))
            
            # Learning progression analysis
            print("[DATA] Analyzing learning progression...")
            
            def model_factory():
                return self.agent.policy_net
            
            analyzer = LearningAnalyzer(
                checkpoint_dir=str(self.session_manager.session_dir),
                model_factory_fn=model_factory
            )
            
            # Use all checkpoints for full time-lapse (166 frames)
            max_checkpoints = len(self.checkpoint_episodes)
            analyzer.analyze_learning_progression(max_checkpoints=max_checkpoints)
            
            # Save learning analysis JSON
            learning_json = self.session_manager.get_plot_path("learning_progression", ".json")
            analyzer.save_analysis_results(str(learning_json))
            
            # Plot unified learning progression (combines timeline + forgetting)
            learning_plot = self.session_manager.get_plot_path("learning_progression")
            analyzer.plot_learning_progression(save_path=str(learning_plot), show=False)
            
            # Generate per-checkpoint behavior heatmaps + animated GIF
            print("\n[CHART] Generating checkpoint behavior heatmaps...")
            # Calculate exact frame duration for 10s @ 24fps time-lapse
            frame_duration = 10.0 / max(1, len(self.checkpoint_episodes))
            analyzer.generate_checkpoint_behavior_heatmaps(
                output_dir=str(self.session_manager.session_dir),
                max_checkpoints=max_checkpoints,
                create_gif=True,
                gif_duration=frame_duration
            )
            
            analyzer.print_summary()
            
            # Final model analysis with comprehensive report
            print("\n[CHART] Analyzing final model behavior...")
            
            behavior_analyzer = BehaviorAnalyzer(
                policy_net=self.agent.policy_net,
                value_net=getattr(self.agent, 'value_net', None)
            )
            
            # Generate full report (includes confusion matrix, behavior analysis)
            behavior_analyzer.generate_full_report(
                output_dir=str(self.session_manager.session_dir),
                show=False
            )
            
            # Quantum contribution analysis (hybrid models only)
            if hasattr(self.agent.policy_net, 'quantum_circuit'):
                self._run_quantum_contribution_analysis()
            
            print("\n[OK] Analysis Complete!")
            print(f"   Results saved to: {self.session_manager.session_dir}")
            
        except Exception as e:
            import traceback
            print(f"\n[WARN] Analysis failed: {e}")
            traceback.print_exc()
        
        print("=" * 80)
    
    def _run_quantum_contribution_analysis(self):
        """Run quantum contribution analysis for hybrid models."""
        try:
            from ..analysis.signal_noise import SignalNoiseAnalyzer as QuantumContributionAnalyzer
            
            print("\n[ATOM]  Analyzing quantum contribution...")
            
            analyzer = QuantumContributionAnalyzer(self.agent.policy_net)
            analysis = analyzer.run_full_analysis()
            
            # Save results
            json_path = self.session_manager.get_plot_path("quantum_contribution", ".json")
            analyzer.save_analysis(str(json_path))
            
            plot_path = self.session_manager.get_plot_path("quantum_contribution")
            analyzer.plot_analysis(str(plot_path))
            
            # Print summary
            summary = analysis['summary']
            lin = analysis['linearity']
            print(f"   Linearity R^2: {lin['r2_overall']:.4f} ({lin['interpretation'].split(':')[0]})")
            print(f"   Verdict: {summary['verdict'].split(':')[0]}")
            
        except Exception as e:
            print(f"   [WARN] Quantum analysis failed: {e}")
    
    def _save_session_config(self):
        """Save complete session configuration for reproducibility."""
        import torch
        import sys
        import numpy as np
        from datetime import datetime
        
        # Extract network configuration from policy network
        policy_net = self.agent.policy_net
        network_config = {}
        
        if hasattr(policy_net, 'quantum_circuit'):
            # Hybrid network
            network_config = {
                'type': 'hybrid',
                'n_qubits': getattr(policy_net, 'n_qubits', None),
                'n_layers': getattr(policy_net, 'n_layers', None),
                'encoding': getattr(policy_net, 'encoding', None),
                'entanglement_strategy': getattr(policy_net, 'entanglement_strategy', None),
                'measurement_mode': getattr(policy_net, 'measurement_mode', None),
                'data_reuploading': getattr(policy_net, 'data_reuploading', None),
                'device_name': getattr(policy_net, 'device_name', 'default.qubit'),
                'single_axis_encoding': getattr(policy_net, 'single_axis_encoding', True),
                'encoder_compression': getattr(policy_net, 'encoder_compression', 'minimal'),
                'encoding_transform': getattr(policy_net, 'encoding_transform', 'arctan'),
                'reuploading_transform': getattr(policy_net, 'reuploading_transform', 'arctan'),
                'encoding_scale': getattr(policy_net, 'encoding_scale', 2.0),
                'learnable_input_scaling': getattr(policy_net, 'learnable_input_scaling', False),
                'quantum_dropout_rate': getattr(policy_net, '_quantum_dropout_rate', 0.0),
                'frozen_components': list(getattr(policy_net, '_frozen_components', set())),
                'microwaved_encoder': getattr(self.config, 'microwaved_encoder', False),
                'microwaved_fraction': getattr(self.config, 'microwaved_fraction', 0.5),
            }
        else:
            # Classical network
            network_config = {
                'type': 'classical',
                'hidden_sizes': getattr(policy_net, 'hidden_sizes', None),
                'activation': getattr(policy_net, 'activation_name', None),
            }
        
        # Build complete session info
        session_info = {
            'session_name': self.session_manager.session_name,
            'created_at': datetime.now().isoformat(),
            'seed': self.agent.seed,
            'network': network_config,
            'agent': {
                'learning_rate': self.agent.training_config['learning_rate'],
                'encoder_lr_scale': getattr(self.agent, 'encoder_lr_scale', 1.0) if hasattr(self.agent, 'encoder_lr_scale') else self.agent.training_config.get('encoder_lr_scale', 1.0),
                'gamma': self.agent.training_config['gamma'],
                'entropy_coef': self.agent.training_config['entropy_coef'],
                'value_coef': self.agent.training_config['value_coef'],
                'max_grad_norm': self.agent.training_config['max_grad_norm'],
                'n_steps': self.agent.training_config['n_steps'],
                'use_gae': self.agent.training_config['use_gae'],
                'gae_lambda': self.agent.training_config['gae_lambda'],
            },
            'training': {
                'n_episodes': self.config.n_episodes,
                'max_steps_per_episode': self.config.max_steps_per_episode,
                'microwaved_encoder': getattr(self.config, 'microwaved_encoder', False),
                'microwaved_fraction': getattr(self.config, 'microwaved_fraction', 0.5),
                'print_every': self.config.print_every,
                'eval_every': self.config.eval_every,
                'eval_episodes': self.config.eval_episodes,
                'save_frequency': self.config.save_frequency,
                'enable_analysis': self.config.enable_analysis,
                'monitor_gradients': self.config.monitor_gradients,
            },
            'environment': {
                'python_version': sys.version,
                'torch_version': torch.__version__,
                'numpy_version': np.__version__,
            }
        }
        
        # Save session_info.json
        self.session_manager.metadata = session_info
        self.session_manager.save_metadata()
        
        if self.verbose:
            print(f"[DATA] Session configuration saved: {self.session_manager.session_dir / 'session_info.json'}")
