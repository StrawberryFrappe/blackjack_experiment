#!/usr/bin/env python
"""
Quantum-First Training CLI - Train quantum before classical.

Usage:
    python -m blackjack_experiment.cli.qfirst
    python -m blackjack_experiment.cli.qfirst --dropout 0.3
    python -m blackjack_experiment.cli.qfirst --qf-episodes 500 --episodes 3000
"""

import argparse
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Quantum-first training: train quantum weights before classical',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quantum-first strategy:
  1. Freeze classical components (encoder, postprocessing)
  2. Train quantum weights alone for N episodes
  3. Unfreeze all, continue normal training

Optional quantum dropout randomly freezes some quantum weights each step,
preventing classical from learning to compensate for specific quantum behavior.

Examples:
    python -m blackjack_experiment.cli.qfirst
    python -m blackjack_experiment.cli.qfirst --dropout 0.3
    python -m blackjack_experiment.cli.qfirst --qf-episodes 500 --episodes 4000
"""
    )
    
    parser.add_argument('-e', '--episodes', type=int, default=5000,
                       help='Total episodes (default: 5000)')
    parser.add_argument('-q', '--qf-episodes', type=int, default=1000,
                       help='Quantum-first episodes before unfreezing (default: 1000)')
    parser.add_argument('-d', '--dropout', type=float, default=0.0,
                       help='Quantum dropout rate 0-1 (default: 0)')
    parser.add_argument('-o', '--output', type=str,
                       help='Output directory')
    parser.add_argument('-s', '--seed', type=int,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Import here to avoid slow startup
    import gymnasium as gym
    import torch
    import numpy as np
    import random
    from collections import deque
    
    from ..networks.hybrid import UniversalBlackjackHybridPolicyNetwork
    from ..networks.classical import BlackjackClassicalValueNetwork
    from ..core.agent import A2CAgent
    from ..core.session import SessionManager
    from ..analysis.experiments.bypass import QuantumFirstTrainer
    from ..analysis.gradient_flow import GradientFlowAnalyzer as HybridGradientAnalyzer
    
    # Set seed
    seed = args.seed or random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Calculate phase split
    qf_episodes = min(args.qf_episodes, args.episodes - 100)
    normal_episodes = args.episodes - qf_episodes
    
    print(f"\n{'='*60}")
    print("QUANTUM-FIRST TRAINING")
    print(f"{'='*60}")
    print(f"Quantum-first: {qf_episodes} episodes")
    print(f"Normal:        {normal_episodes} episodes")
    print(f"Dropout:       {args.dropout:.0%}")
    print(f"Seed:          {seed}")
    
    # Create network
    policy_net = UniversalBlackjackHybridPolicyNetwork()
    value_net = BlackjackClassicalValueNetwork()
    
    # Configure quantum-first
    qf = QuantumFirstTrainer(
        quantum_first_episodes=qf_episodes,
        normal_episodes=normal_episodes,
        quantum_dropout_rate=args.dropout
    )
    qf.configure_network(policy_net)
    
    agent = A2CAgent(policy_net=policy_net, value_net=value_net, 
                     learning_rate=0.001, gamma=0.99, seed=args.seed)
    
    env = gym.make('Blackjack-v1', sab=True)
    output = args.output or f"results/qfirst_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = SessionManager(base_dir=output, session_name="")
    
    # Setup gradient analyzer
    gradient_analyzer = None
    try:
        gradient_analyzer = HybridGradientAnalyzer(policy_net, n_qubits=policy_net.n_qubits)
        print(f"[OK] Hybrid gradient analyzer initialized ({policy_net.n_qubits}-qubit network)")
    except Exception as e:
        print(f"[WARN] Failed to initialize gradient analyzer: {e}")
    
    # Training loop
    total = qf.get_total_episodes()
    recent = deque(maxlen=100)
    all_rewards = []
    checkpoint_episodes = []
    
    # Training config
    save_frequency = max(100, total // 10)
    print_every = 500
    
    for ep in range(total):
        qf.on_episode(ep, policy_net)
        
        state, _ = env.reset()
        agent.reset_episode()
        reward_sum = 0
        
        while True:
            action = agent.select_action(state, training=True)
            state, reward, term, trunc, _ = env.step(action)
            agent.step_update(reward, state, term or trunc)
            reward_sum += reward
            if term or trunc:
                break
        
        metrics = agent.update()
        
        # Capture gradient metrics
        if gradient_analyzer:
            try:
                gradient_analyzer.capture_gradients()
            except Exception:
                pass
        
        recent.append(reward_sum)
        all_rewards.append(reward_sum)
        
        # Print progress
        if (ep + 1) % print_every == 0:
            wr = sum(1 for r in recent if r > 0) / len(recent) * 100
            phase = "QF" if ep < qf_episodes else "NORMAL"
            print(f"Ep {ep+1}/{total} [{phase}] | WR: {wr:.1f}%")
        
        # Save checkpoint
        if (ep + 1) % save_frequency == 0 and len(recent) >= 100:
            wr = sum(1 for r in recent if r > 0) / len(recent) * 100
            path = session.get_checkpoint_path(ep + 1, wr)
            agent.save(str(path))
            checkpoint_episodes.append(ep + 1)
            print(f"  Checkpoint saved: {path.name}")
    
    # Save final model
    final_wr = sum(1 for r in recent if r > 0) / len(recent) * 100
    path = session.get_final_model_path(final_wr, np.mean(recent))
    agent.save(str(path))
    
    print(f"\n[OK] Saved: {path}")
    print(f"  Final WR: {final_wr:.1f}%")
    
    env.close()
    
    # Save gradient analysis
    if gradient_analyzer is not None:
        try:
            print("\n" + "=" * 80)
            print("[DATA] GRADIENT ANALYSIS - Quantum-First Hybrid Network")
            print("=" * 80)
            
            gradient_analyzer.print_summary()
            
            plot_path = session.get_plot_path('gradient_analysis')
            gradient_analyzer.plot_gradient_analysis(save_path=str(plot_path), show=False)
            print(f"[OK] Saved: {plot_path.name}")
            
            json_path = gradient_analyzer.save_analysis(session.session_dir)
            print(f"[OK] Saved: {json_path.name}")
            
            print("=" * 80 + "\n")
        except Exception as e:
            print(f"\n[WARN] Gradient analysis failed: {e}")
    
    # Run full post-training analysis
    _run_analysis(agent, session, checkpoint_episodes)


def _run_analysis(agent, session, checkpoint_episodes):
    """Run comprehensive post-training analysis."""
    print("\n" + "=" * 80)
    print("[TEST] RUNNING POST-TRAINING ANALYSIS (Quantum-First)")
    print("=" * 80)
    
    try:
        from ..analysis.learning import LearningAnalyzer
        from ..analysis.strategy import BehaviorAnalyzer
        
        # Learning progression analysis
        print("[DATA] Analyzing learning progression...")
        
        def model_factory():
            return agent.policy_net
        
        analyzer = LearningAnalyzer(
            checkpoint_dir=str(session.models_dir),
            model_factory_fn=model_factory
        )
        
        max_checkpoints = min(20, len(checkpoint_episodes))
        if max_checkpoints > 0:
            analyzer.analyze_learning_progression(max_checkpoints=max_checkpoints)
            
            # Save learning analysis JSON
            learning_json = session.get_plot_path("learning_progression", ".json")
            analyzer.save_analysis_results(str(learning_json))
            
            # Plot unified learning progression
            learning_plot = session.get_plot_path("learning_progression")
            analyzer.plot_learning_progression(save_path=str(learning_plot), show=False)
            
            # Generate per-checkpoint behavior heatmaps + animated GIF
            print("\n[CHART] Generating checkpoint behavior heatmaps...")
            analyzer.generate_checkpoint_behavior_heatmaps(
                output_dir=str(session.session_dir),
                max_checkpoints=max_checkpoints,
                create_gif=True,
                gif_duration=0.8
            )
            
            analyzer.print_summary()
        else:
            print("   [SKIP] No checkpoints available for learning analysis")
        
        # Final model analysis
        print("\n[CHART] Analyzing final model behavior...")
        
        behavior_analyzer = BehaviorAnalyzer(
            policy_net=agent.policy_net,
            value_net=getattr(agent, 'value_net', None)
        )
        
        # Generate full report
        behavior_analyzer.generate_full_report(
            output_dir=str(session.session_dir),
            show=False
        )
        
        # Quantum contribution analysis
        if hasattr(agent.policy_net, 'quantum_circuit'):
            _run_quantum_contribution_analysis(agent, session)
        
        print("\n[OK] Analysis Complete!")
        print(f"   Results saved to: {session.session_dir}")
        
    except Exception as e:
        import traceback
        print(f"\n[WARN] Analysis failed: {e}")
        traceback.print_exc()
    
    print("=" * 80)


def _run_quantum_contribution_analysis(agent, session):
    """Run quantum contribution analysis for hybrid models."""
    try:
        from ..analysis.signal_noise import SignalNoiseAnalyzer as QuantumContributionAnalyzer
        
        print("\n[ATOM]  Analyzing quantum contribution...")
        
        analyzer = QuantumContributionAnalyzer(agent.policy_net)
        analysis = analyzer.run_full_analysis()
        
        # Save results
        json_path = session.get_plot_path("quantum_contribution", ".json")
        analyzer.save_analysis(str(json_path))
        
        plot_path = session.get_plot_path("quantum_contribution")
        analyzer.plot_analysis(str(plot_path))
        
        # Print summary
        summary = analysis['summary']
        lin = analysis['linearity']
        print(f"   Linearity R^2: {lin['r2_overall']:.4f} ({lin['interpretation'].split(':')[0]})")
        print(f"   Verdict: {summary['verdict'].split(':')[0]}")
        
    except Exception as e:
        print(f"   [WARN] Quantum analysis failed: {e}")


if __name__ == '__main__':
    main()
