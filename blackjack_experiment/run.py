#!/usr/bin/env python
"""
Blackjack RL Experiment Runner - Simple entry point.

Commands:
    python run.py                    # Compare classical vs hybrid
    python run.py classical          # Train classical only
    python run.py hybrid             # Train hybrid only
    python run.py eval <checkpoint>  # Evaluate a model
    python run.py bypass             # [DEPRECATED] Module removed, use qfirst instead
    python run.py qfirst             # Quantum-first training
    python run.py everything         # Run all experiments
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

# Ensure parent directory is in path for direct script execution
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))


def train(network_type: str, episodes: int, output: str = None, seed: int = None, dropout: float = 0.0, frozen_encoder: bool = False, checkpoint_count: int = 180, encoder_lr_scale: float = 1.0, microwaved_encoder: bool = False, microwaved_fraction: float = 0.5):
    """Train a model."""
    import gymnasium as gym
    import torch
    import numpy as np
    import random
    
    from blackjack_experiment.networks.classical import (
        BlackjackClassicalPolicyNetwork,
        BlackjackClassicalValueNetwork
    )
    from blackjack_experiment.networks.hybrid import UniversalBlackjackHybridPolicyNetwork
    from blackjack_experiment.core.config import Config, NetworkConfig, AgentConfig, TrainingConfig
    from blackjack_experiment.core.agent import A2CAgent
    from blackjack_experiment.core.trainer import Trainer
    from blackjack_experiment.core.session import SessionManager
    
    seed = seed or random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"TRAINING: {network_type.upper()}")
    print(f"{'='*60}")
    print(f"Episodes: {episodes} | Seed: {seed}")
    
    # Create networks
    if network_type == 'hybrid':
        policy_net = UniversalBlackjackHybridPolicyNetwork()
        print(f"Qubits: {policy_net.n_qubits} | Layers: {policy_net.n_layers}")
        print(f"Encoder: {policy_net.encoder_compression} | Single-axis: {policy_net.single_axis_encoding}")
        if dropout > 0:
            policy_net.set_quantum_dropout(dropout)
            print(f"Quantum dropout: {dropout:.0%}")
        if frozen_encoder:
            policy_net.set_bypass_mode('scrambler')
            print(f"Frozen encoder: ENABLED (scrambler mode - random frozen weights)")
        if microwaved_encoder:
            print(f"Microwaved encoder: ENABLED (frozen until {microwaved_fraction:.0%} of training)")
        if encoder_lr_scale != 1.0:
            base_lr = 1e-3 # Default from AgentConfig below
            print(f"Encoder LR Scale: {encoder_lr_scale}x (LR: {base_lr * encoder_lr_scale:.6f})")
    else:
        policy_net = BlackjackClassicalPolicyNetwork()
    value_net = BlackjackClassicalValueNetwork()
    
    # Config
    config = Config(
        seed=seed,
        network=NetworkConfig(network_type=network_type),
        agent=AgentConfig(lr_policy=1e-3, lr_value=1e-3, gamma=0.99, entropy_coef=0.01, encoder_lr_scale=encoder_lr_scale),
        training=TrainingConfig(n_episodes=episodes, 
                               checkpoint_count=checkpoint_count,
                               eval_every=max(50, episodes//20),
                               microwaved_encoder=microwaved_encoder,
                               microwaved_fraction=microwaved_fraction)
    )
    
    # Setup
    env = gym.make('Blackjack-v1')
    agent = A2CAgent(
        policy_net, value_net,
        learning_rate=config.agent.learning_rate,
        gamma=config.agent.gamma,
        entropy_coef=config.agent.entropy_coef,
        value_coef=config.agent.value_coef,
        max_grad_norm=config.agent.max_grad_norm,
        n_steps=config.agent.n_steps,
        use_gae=config.agent.use_gae,
        gae_lambda=config.agent.gae_lambda,
        seed=seed,
        encoder_lr_scale=config.agent.encoder_lr_scale
    )
    
    output = output or f"results/{network_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = SessionManager(base_dir=output, session_name="")
    
    # Train
    trainer = Trainer(agent, env, config.training, session)
    rewards, _, _ = trainer.train()
    
    # Summary
    recent = rewards[-100:] if len(rewards) >= 100 else rewards
    wr = sum(1 for r in recent if r > 0) / len(recent) * 100 if recent else 0
    
    return {'win_rate': wr, 'episodes': len(rewards), 'seed': seed, 'output': output}


def compare(episodes: int, output: str = None, seed: int = None, dropout: float = 0.0, frozen_encoder: bool = False, checkpoint_count: int = 180, encoder_lr_scale: float = 1.0, microwaved_encoder: bool = False, microwaved_fraction: float = 0.5):
    """Compare classical vs hybrid."""
    from blackjack_experiment.core.session import ComparisonSessionManager
    from blackjack_experiment.analysis.comparison import ComparisonAnalyzer
    from blackjack_experiment.networks.loader import load_policy_network
    import random
    import numpy as np
    
    seed = seed or random.randint(0, 2**31 - 1)
    output = output or f"results/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n{'='*60}")
    print("COMPARISON: Classical vs Hybrid")
    print(f"{'='*60}")
    print(f"Episodes: {episodes} | Seed: {seed}")
    if dropout > 0:
        print(f"Quantum dropout: {dropout:.0%}")
    if frozen_encoder:
        print(f"Frozen encoder: ENABLED (hybrid only)")
    if microwaved_encoder:
        print(f"Microwaved encoder: ENABLED (frozen until {microwaved_fraction:.0%} of training)")
    if encoder_lr_scale != 1.0:
        print(f"Encoder LR Scale: {encoder_lr_scale}x")
    
    session = ComparisonSessionManager(output)
    results = {}
    
    # Train both
    print("\n[1/2] Classical...")
    classical_out = str(session.create_session('classical').session_dir)
    results['classical'] = train('classical', episodes, classical_out, seed, checkpoint_count=checkpoint_count)
    
    print("\n[2/2] Hybrid...")
    hybrid_out = str(session.create_session('hybrid').session_dir)
    results['hybrid'] = train('hybrid', episodes, hybrid_out, seed + 1, dropout, frozen_encoder, checkpoint_count, encoder_lr_scale, microwaved_encoder, microwaved_fraction)
    
    # Run comparison analysis
    print(f"\n{'='*60}")
    print("RUNNING COMPARISON ANALYSIS")
    print(f"{'='*60}")
    
    try:
        # Load final models
        classical_model_path = Path(classical_out) / results['classical']['output'].split('/')[-1] if '/' in results['classical']['output'] else Path(results['classical']['output'])
        hybrid_model_path = Path(hybrid_out) / results['hybrid']['output'].split('/')[-1] if '/' in results['hybrid']['output'] else Path(results['hybrid']['output'])
        
        # Find final model files
        classical_final = list(Path(classical_out).glob("final_*.pth"))
        hybrid_final = list(Path(hybrid_out).glob("final_*.pth"))
        
        if classical_final and hybrid_final:
            classical_net = load_policy_network(str(classical_final[0]), strict=False)
            hybrid_net = load_policy_network(str(hybrid_final[0]), strict=False)
            
            # Run comparison analysis
            models = {
                'classical': classical_net,
                'hybrid': hybrid_net
            }
            
            analyzer = ComparisonAnalyzer(models)
            
            # Generate comparison plots
            print("\n[DATA] Generating comparison visualizations...")
            
            # Side-by-side decision heatmaps
            plot_path = session.results_dir / "decision_comparison.png"
            analyzer.plot_comparison_heatmaps(save_path=str(plot_path), show=False)
            print(f"   [OK] Saved: {plot_path.name}")
            
            # Agreement analysis
            agreement_no_ace = analyzer.compute_agreement_matrix(usable_ace=0)
            agreement_usable = analyzer.compute_agreement_matrix(usable_ace=1)
            
            agreement_path = session.results_dir / "agreement_analysis.json"
            with open(agreement_path, 'w') as f:
                json.dump({
                    'no_ace': {
                        'agreement_matrix': agreement_no_ace['agreement_matrix'].tolist(),
                        'model_names': agreement_no_ace['model_names']
                    },
                    'usable_ace': {
                        'agreement_matrix': agreement_usable['agreement_matrix'].tolist(),
                        'model_names': agreement_usable['model_names']
                    }
                }, f, indent=2)
            print(f"   [OK] Saved: {agreement_path.name}")
            
            # Behavioral profiles
            profiles = analyzer.compute_behavioral_profiles()
            profiles_path = session.results_dir / "behavioral_profiles.json"
            
            def json_encoder(o):
                """Handle numpy types and other non-serializable objects."""
                import numpy as np
                if isinstance(o, np.integer):
                    return int(o)
                elif isinstance(o, np.floating):
                    return float(o)
                elif isinstance(o, np.ndarray):
                    return o.tolist()
                elif isinstance(o, (np.bool_)):
                    return bool(o)
                else:
                    return str(o)
            
            with open(profiles_path, 'w') as f:
                json.dump(profiles, f, indent=2, default=json_encoder)
            print(f"   [OK] Saved: {profiles_path.name}")
            
            print("\n[OK] Comparison analysis complete!")
        else:
            print("\n[WARN] Could not find final model files for comparison analysis")
            
    except Exception as e:
        import traceback
        print(f"\n[WARN] Comparison analysis failed: {e}")
        traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Classical: {results['classical']['win_rate']:.1f}% WR")
    print(f"Hybrid:    {results['hybrid']['win_rate']:.1f}% WR")
    print(f"\nSaved to: {output}")
    
    # Save summary
    with open(Path(output) / 'comparison_summary.json', 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 
                  'episodes': episodes, 'seed': seed, 'results': results}, f, indent=2)
    
    return results


def evaluate(checkpoint: str, episodes: int = 500):
    """Evaluate a checkpoint."""
    import gymnasium as gym
    from blackjack_experiment.networks.loader import load_policy_network
    from blackjack_experiment.analysis.strategy import BehaviorAnalyzer
    
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint}")
    
    # Load
    policy_net = load_policy_network(checkpoint, strict=False)
    net_type = 'hybrid' if hasattr(policy_net, 'quantum_circuit') else 'classical'
    print(f"Type: {net_type}")
    
    # Evaluate
    env = gym.make('Blackjack-v1')
    rewards = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            import torch
            with torch.no_grad():
                probs = policy_net(state)
                action = probs.argmax(dim=-1).item()
            state, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            done = term or trunc
        rewards.append(ep_reward)
    
    env.close()
    
    # Results
    wr = sum(1 for r in rewards if r > 0) / len(rewards) * 100
    print(f"\nWin Rate: {wr:.1f}% ({episodes} episodes)")
    
    # Quick behavior analysis
    analyzer = BehaviorAnalyzer(policy_net)
    cm = analyzer.generate_confusion_matrix()
    print(f"Basic Strategy Accuracy: {cm['accuracy']:.1f}%")
    
    return {'win_rate': wr, 'accuracy': cm['accuracy']}


def bypass(episodes: int = 1000, output: str = None, seed: int = None):
    """Run bypass experiment. [DEPRECATED - Module removed]"""
    print("[ERROR] The bypass experiment module has been removed.")
    print("Please use 'qfirst' for quantum-first training experiments.")
    return None


def qfirst(episodes: int = 5000, qf_episodes: int = 1000, dropout: float = 0.0, 
           output: str = None, seed: int = None):
    """Run quantum-first training with full analysis pipeline."""
    import gymnasium as gym
    import torch
    import numpy as np
    import random
    from collections import deque
    
    from blackjack_experiment.networks.hybrid import UniversalBlackjackHybridPolicyNetwork
    from blackjack_experiment.networks.classical import BlackjackClassicalValueNetwork
    from blackjack_experiment.core.agent import A2CAgent
    from blackjack_experiment.core.session import SessionManager
    from blackjack_experiment.analysis.experiments.qfirst import QuantumFirstTrainer
    from blackjack_experiment.analysis.gradient_flow import GradientFlowAnalyzer as HybridGradientAnalyzer
    
    seed = seed or random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    qf_episodes = min(qf_episodes, episodes - 100)
    normal_episodes = episodes - qf_episodes
    
    print(f"\n{'='*60}")
    print("QUANTUM-FIRST TRAINING")
    print(f"{'='*60}")
    print(f"QF: {qf_episodes} | Normal: {normal_episodes} | Dropout: {dropout:.0%}")
    print(f"Seed: {seed}")
    
    policy_net = UniversalBlackjackHybridPolicyNetwork()
    value_net = BlackjackClassicalValueNetwork()
    
    qf = QuantumFirstTrainer(
        quantum_first_episodes=qf_episodes,
        normal_episodes=normal_episodes,
        quantum_dropout_rate=dropout
    )
    qf.configure_network(policy_net)
    
    agent = A2CAgent(policy_net=policy_net, value_net=value_net, 
                     learning_rate=0.001, gamma=0.99, seed=seed)
    
    env = gym.make('Blackjack-v1', sab=True)
    output = output or f"results/qfirst_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = SessionManager(base_dir=output, session_name="")
    
    # Setup gradient analyzer for hybrid networks
    gradient_analyzer = None
    try:
        gradient_analyzer = HybridGradientAnalyzer(policy_net, n_qubits=policy_net.n_qubits)
        print(f"[OK] Hybrid gradient analyzer initialized ({policy_net.n_qubits}-qubit network)")
    except Exception as e:
        print(f"[WARN] Failed to initialize gradient analyzer: {e}")
    
    total = qf.get_total_episodes()
    recent = deque(maxlen=100)
    all_rewards = []
    checkpoint_episodes = []
    
    # Training config for checkpointing
    save_frequency = max(100, total // 10)  # ~10 checkpoints
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
                grad_metrics = gradient_analyzer.capture_gradients()
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
    _run_qfirst_analysis(agent, session, checkpoint_episodes, all_rewards)
    
    return {'win_rate': final_wr, 'output': output}


def _run_qfirst_analysis(agent, session, checkpoint_episodes, all_rewards):
    """Run comprehensive post-training analysis for quantum-first training."""
    print("\n" + "=" * 80)
    print("[TEST] RUNNING POST-TRAINING ANALYSIS (Quantum-First)")
    print("=" * 80)
    
    try:
        from blackjack_experiment.analysis.learning import LearningAnalyzer
        from blackjack_experiment.analysis.strategy import BehaviorAnalyzer
        
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
        
        # Final model analysis with comprehensive report
        print("\n[CHART] Analyzing final model behavior...")
        
        behavior_analyzer = BehaviorAnalyzer(
            policy_net=agent.policy_net,
            value_net=getattr(agent, 'value_net', None)
        )
        
        # Generate full report (includes confusion matrix, behavior analysis)
        behavior_analyzer.generate_full_report(
            output_dir=str(session.session_dir),
            show=False
        )
        
        # Quantum contribution analysis (hybrid models only)
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
        from blackjack_experiment.analysis.signal_noise import SignalNoiseAnalyzer as QuantumContributionAnalyzer
        
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


def everything(episodes: int = 5000, output: str = None, seed: int = None):
    """Run ALL experiments: comparison + bypass + qfirst."""
    import random
    
    seed = seed or random.randint(0, 2**31 - 1)
    base_output = output or f"results/everything_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(base_output).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("🚀 RUNNING ALL EXPERIMENTS")
    print(f"{'='*70}")
    print(f"Base output: {base_output}")
    print(f"Episodes: {episodes} | Seed: {seed}")
    
    results = {}
    
    # 1. Comparison
    print(f"\n{'='*70}")
    print("[1/3] COMPARISON EXPERIMENT")
    print(f"{'='*70}")
    results['comparison'] = compare(episodes, f"{base_output}/comparison", seed)
    
    # 2. Bypass
    print(f"\n{'='*70}")
    print("[2/3] BYPASS EXPERIMENT")
    print(f"{'='*70}")
    bypass_ep = min(1000, episodes // 5)  # Scale bypass episodes
    results['bypass'] = bypass(bypass_ep, f"{base_output}/bypass", seed + 100)
    
    # 3. Quantum-first
    print(f"\n{'='*70}")
    print("[3/3] QUANTUM-FIRST EXPERIMENT")
    print(f"{'='*70}")
    results['qfirst'] = qfirst(episodes, episodes // 5, 0.2, f"{base_output}/qfirst", seed + 200)
    
    # Final summary
    print(f"\n{'='*70}")
    print("[DATA] ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"\nComparison:")
    print(f"  Classical: {results['comparison']['classical']['win_rate']:.1f}% WR")
    print(f"  Hybrid:    {results['comparison']['hybrid']['win_rate']:.1f}% WR")
    print(f"\nBypass verdict: {results['bypass'].get('verdict', 'N/A')}")
    print(f"\nQuantum-first: {results['qfirst']['win_rate']:.1f}% WR")
    print(f"\nAll results saved to: {base_output}")
    
    # Save combined summary
    with open(Path(base_output) / 'everything_summary.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'episodes': episodes,
            'seed': seed,
            'results': {
                'comparison': {
                    'classical_wr': results['comparison']['classical']['win_rate'],
                    'hybrid_wr': results['comparison']['hybrid']['win_rate']
                },
                'bypass_verdict': results['bypass'].get('verdict'),
                'qfirst_wr': results['qfirst']['win_rate']
            }
        }, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Blackjack RL Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
    (default)     Compare classical vs hybrid
    classical     Train classical network
    hybrid        Train hybrid quantum network  
    eval FILE     Evaluate a checkpoint
    bypass        Bypass experiment (test quantum contribution)
    qfirst        Quantum-first training
    everything    Run ALL experiments

Examples:
    python run.py                          # Compare (5000 ep)
    python run.py classical -e 2000        # Train classical
    python run.py hybrid -e 3000 -d 0.2    # Train hybrid + 20% dropout
    python run.py hybrid --frozen-encoder  # Train with frozen random encoder
    python run.py compare -d 0.3           # Compare with dropout
    python run.py compare --frozen-encoder # Compare with frozen encoder (hybrid)
    python run.py eval results/final.pth   # Evaluate model
    python run.py bypass -e 1000           # Bypass experiment
    python run.py qfirst -e 5000 -d 0.3    # Quantum-first + dropout
    python run.py everything -e 3000       # Run everything
"""
    )
    
    parser.add_argument('command', nargs='?', default='compare',
                       choices=['compare', 'classical', 'hybrid', 'eval', 'bypass', 'qfirst', 'everything'],
                       help='Command to run')
    parser.add_argument('checkpoint', nargs='?',
                       help='Checkpoint path (for eval)')
    parser.add_argument('-e', '--episodes', type=int, default=5000,
                       help='Training episodes (default: 5000)')
    parser.add_argument('-c', '--checkpoint-count', type=int, default=None,
                       help='Number of evenly-spaced checkpoints. Default depends on --generate-gif.')
    parser.add_argument('--generate-gif', action='store_true',
                       help='Enable high-frequency checkpoints for smooth GIF generation.')
    parser.add_argument('-q', '--qf-episodes', type=int, default=1000,
                       help='Quantum-first episodes (for qfirst)')
    parser.add_argument('-d', '--dropout', type=float, default=0.0,
                       help='Quantum dropout rate 0-1 (for hybrid/compare/qfirst)')
    parser.add_argument('--frozen-encoder', action='store_true',
                       help='Freeze encoder with random weights (scrambler mode, hybrid only)')
    parser.add_argument('--microwave', type=float,
                       help='Fraction of training to keep encoder frozen (0-1). Enables microwaved encoder mode.')
    parser.add_argument('--encoder-lr-scale', type=float, default=1.0,
                       help='Scaling factor for feature encoder learning rate (default: 1.0)')
    parser.add_argument('-o', '--output', type=str,
                       help='Output directory')
    parser.add_argument('-s', '--seed', type=int,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Determine checkpoint count if not specified
    if args.checkpoint_count is None:
        if args.generate_gif:
            args.checkpoint_count = 240
        else:
            args.checkpoint_count = 10
    
    # Handle microwaved encoder logic
    microwaved_encoder = False
    microwaved_fraction = 0.5
    
    if args.microwave is not None:
        microwaved_encoder = True
        microwaved_fraction = args.microwave
        if not (0.0 <= microwaved_fraction <= 1.0):
            parser.error("Microwave fraction must be between 0 and 1")

    if args.command == 'eval':
        if not args.checkpoint:
            parser.error("eval requires a checkpoint path")
        evaluate(args.checkpoint, args.episodes)
    elif args.command == 'compare':
        compare(args.episodes, args.output, args.seed, args.dropout, args.frozen_encoder, args.checkpoint_count, args.encoder_lr_scale, microwaved_encoder, microwaved_fraction)
    elif args.command == 'classical':
        train('classical', args.episodes, args.output, args.seed, checkpoint_count=args.checkpoint_count)
    elif args.command == 'hybrid':
        train('hybrid', args.episodes, args.output, args.seed, args.dropout, args.frozen_encoder, args.checkpoint_count, args.encoder_lr_scale, microwaved_encoder, microwaved_fraction)
    elif args.command == 'bypass':
        bypass(args.episodes, args.output, args.seed)
    elif args.command == 'qfirst':
        qfirst(args.episodes, args.qf_episodes, args.dropout, args.output, args.seed)
    elif args.command == 'everything':
        everything(args.episodes, args.output, args.seed)
    else:
        compare(args.episodes, args.output, args.seed, args.dropout, args.frozen_encoder, args.checkpoint_count, args.encoder_lr_scale, microwaved_encoder, microwaved_fraction)


if __name__ == '__main__':
    main()
