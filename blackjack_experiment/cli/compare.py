"""
CLI tool for running experiments and comparisons.

Usage:
    python -m blackjack_experiment.cli.compare --help
    python -m blackjack_experiment.cli.compare train --type hybrid --episodes 5000
    python -m blackjack_experiment.cli.compare eval --checkpoint results/final.pth
    python -m blackjack_experiment.cli.compare analyze --dir results/comparison_xyz
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from datetime import datetime


def create_model(network_type: str):
    """Create policy and value networks based on type.
    
    Uses class defaults for network architecture.
    """
    from ..networks.classical import (
        BlackjackClassicalPolicyNetwork,
        BlackjackClassicalValueNetwork,
        BlackjackMinimalClassicalPolicyNetwork,
        BlackjackMinimalClassicalValueNetwork
    )
    from ..networks.hybrid import UniversalBlackjackHybridPolicyNetwork
    
    if network_type == 'classical':
        return (
            BlackjackClassicalPolicyNetwork(),
            BlackjackClassicalValueNetwork()
        )
    elif network_type == 'minimal':
        return (
            BlackjackMinimalClassicalPolicyNetwork(),
            BlackjackMinimalClassicalValueNetwork()
        )
    elif network_type == 'hybrid':
        # Hybrid uses classical value network, architecture from class defaults
        return (
            UniversalBlackjackHybridPolicyNetwork(),
            BlackjackClassicalValueNetwork()
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def load_checkpoint(checkpoint_path: str, strict: bool = True):
    """Load model weights from checkpoint using the network loader.
    
    Args:
        checkpoint_path: Path to checkpoint file
        strict: If True, fail when network config is missing
        
    Returns:
        Loaded policy network in eval mode
    """
    from ..networks.loader import load_policy_network
    return load_policy_network(checkpoint_path, strict=strict)


def cmd_train(args):
    """Train a model.
    
    Network architecture is determined by class defaults in the network files.
    """
    import gymnasium as gym
    from ..core.agent import A2CAgent
    from ..core.config import Config, NetworkConfig, AgentConfig, TrainingConfig
    from ..core.trainer import Trainer
    from ..core.session import SessionManager
    
    print("\n" + "="*70)
    print("BLACKJACK TRAINING")
    print("="*70)
    print(f"Network type: {args.type}")
    print(f"Episodes: {args.episodes}")
    
    # Create config - network architecture comes from class defaults
    network_config = NetworkConfig(network_type=args.type)
    
    agent_config = AgentConfig(
        lr_policy=args.lr,
        lr_value=args.lr,
        gamma=0.99,
        entropy_coef=0.01
    )
    
    training_config = TrainingConfig(
        n_episodes=args.episodes,
        checkpoint_count=args.checkpoint_count,
        eval_every=args.eval_freq
    )
    
    config = Config(
        network=network_config,
        agent=agent_config,
        training=training_config
    )
    
    # Setup - use class defaults for architecture
    env = gym.make('Blackjack-v1')
    policy_net, value_net = create_model(args.type)
    
    if args.type == 'hybrid':
        print(f"Quantum circuit: {policy_net.n_qubits} qubits, {policy_net.n_layers} layers")
    
    agent = A2CAgent(
        policy_net, value_net,
        learning_rate=agent_config.learning_rate,
        gamma=agent_config.gamma,
        entropy_coef=agent_config.entropy_coef,
        seed=config.seed
    )
    
    output_dir = args.output or f"results/{args.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = SessionManager(output_dir)
    
    # Train (pass TrainingConfig like run.py does)
    trainer = Trainer(agent, env, training_config, session)
    trainer.train()
    
    print(f"\n[OK] Training complete! Results saved to: {output_dir}")


def cmd_eval(args):
    """Evaluate a checkpoint.
    
    Uses network loader to recreate network from checkpoint config.
    """
    import gymnasium as gym
    from ..core.trainer import evaluate
    from ..networks.loader import NetworkLoadError
    
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    
    # Load model using network loader (extracts config from checkpoint)
    try:
        policy_net = load_checkpoint(args.checkpoint, strict=False)
    except NetworkLoadError as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Determine type from loaded network
    network_type = 'hybrid' if hasattr(policy_net, 'quantum_circuit') else 'classical'
    print(f"Network type: {network_type}")
    if network_type == 'hybrid':
        print(f"  n_qubits: {policy_net.n_qubits}, n_layers: {policy_net.n_layers}")
    
    # Evaluate
    env = gym.make('Blackjack-v1')
    results = evaluate(env, policy_net, args.episodes)
    
    print(f"\n[DATA] RESULTS:")
    print(f"   Win rate: {results['win_rate']:.1f}%")
    print(f"   Loss rate: {results['loss_rate']:.1f}%")
    print(f"   Draw rate: {results['draw_rate']:.1f}%")
    print(f"   Avg reward: {results['avg_reward']:.3f}")


def cmd_analyze(args):
    """Analyze a checkpoint or comparison directory."""
    print("\n" + "="*70)
    print("MODEL ANALYSIS")
    print("="*70)
    
    if args.learning:
        _analyze_learning(args)
    elif args.decisions:
        _analyze_decisions(args)
    elif args.quantum:
        _analyze_quantum(args)
    else:
        # Default: run all analyses
        if args.checkpoint:
            _analyze_decisions(args)
            if 'hybrid' in args.checkpoint.lower():
                _analyze_quantum(args)
        elif args.dir:
            _analyze_learning(args)


def _analyze_learning(args):
    """Run learning progression analysis."""
    from ..analysis.learning import LearningAnalyzer
    from ..networks.loader import load_policy_network
    
    if not args.dir:
        print("[ERROR] --dir required for learning analysis")
        return
    
    checkpoint_dir = Path(args.dir)
    if not checkpoint_dir.exists():
        print(f"[ERROR] Directory not found: {checkpoint_dir}")
        return
    
    print(f"Analyzing learning progression in: {checkpoint_dir}")
    
    # Model factory that uses network loader
    def model_factory():
        # Find any checkpoint to determine network type
        checkpoints = list(checkpoint_dir.glob('*.pth'))
        if checkpoints:
            return load_policy_network(str(checkpoints[0]), strict=False)
        # Fallback to class defaults
        if 'hybrid' in str(checkpoint_dir).lower():
            from ..networks.hybrid import UniversalBlackjackHybridPolicyNetwork
            return UniversalBlackjackHybridPolicyNetwork()
        else:
            from ..networks.classical import BlackjackClassicalPolicyNetwork
            return BlackjackClassicalPolicyNetwork()
    
    analyzer = LearningAnalyzer(str(checkpoint_dir), model_factory)
    analyzer.analyze_learning_progression(max_checkpoints=args.max_checkpoints)
    analyzer.print_summary()
    
    if args.plot:
        output_path = checkpoint_dir / 'learning_timeline.png'
        analyzer.plot_learning_timeline(save_path=str(output_path))
    
    if args.save:
        output_path = checkpoint_dir / 'learning_analysis.json'
        analyzer.save_analysis_results(str(output_path))


def _analyze_decisions(args):
    """Run decision visualization analysis."""
    from ..analysis.strategy import DecisionAnalyzer
    from ..networks.loader import NetworkLoadError
    
    if not args.checkpoint:
        print("[ERROR] --checkpoint required for decision analysis")
        return
    
    print(f"Analyzing decisions for: {args.checkpoint}")
    
    # Load model using network loader
    try:
        policy_net = load_checkpoint(args.checkpoint, strict=False)
    except NetworkLoadError as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    network_type = 'hybrid' if hasattr(policy_net, 'quantum_circuit') else 'classical'
    print(f"Network type: {network_type}")
    
    analyzer = DecisionAnalyzer(policy_net)
    
    # Generate confusion matrix analysis
    cm_data = analyzer.generate_confusion_matrix()
    print(f"\n📋 BEHAVIOR SUMMARY:")
    print(f"   Accuracy vs Basic Strategy: {cm_data['accuracy']:.1f}%")
    print(f"   True Stand: {cm_data['counts']['true_stand']} states")
    print(f"   True Hit: {cm_data['counts']['true_hit']} states")
    print(f"   Too Passive: {cm_data['counts']['false_stand']} states")
    print(f"   Too Aggressive: {cm_data['counts']['false_hit']} states")
    
    if args.plot:
        output_dir = Path(args.checkpoint).parent
        analyzer.plot_confusion_matrix(save_path=str(output_dir / 'confusion_matrix.png'))
    
    if args.save:
        output_dir = Path(args.checkpoint).parent
        analyzer.generate_full_report(str(output_dir))


def _analyze_quantum(args):
    """Run quantum contribution analysis."""
    from ..analysis.gradient_flow import GradientFlowAnalyzer as QuantumContributionAnalyzer
    from ..networks.loader import load_policy_network, NetworkLoadError
    
    if not args.checkpoint:
        print("[ERROR] --checkpoint required for quantum analysis")
        return
    
    print(f"Analyzing quantum contribution for: {args.checkpoint}")
    
    # Load model using network loader
    try:
        policy_net = load_policy_network(args.checkpoint, strict=False)
    except NetworkLoadError as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    if not hasattr(policy_net, 'quantum_circuit'):
        print("[WARN] Quantum analysis only meaningful for hybrid models")
        return
    
    print(f"Loaded hybrid model: {policy_net.n_qubits} qubits, {policy_net.n_layers} layers")
    
    # Run quantum contribution analysis
    from ..analysis.gradient_flow import analyze_model_quantum_contribution
    
    def model_factory():
        return load_policy_network(args.checkpoint, strict=False)
    
    results = analyze_model_quantum_contribution(
        args.checkpoint,
        model_factory,
        n_samples=args.n_samples or 100
    )
    
    if 'error' in results:
        print(f"[WARN] {results['error']}")
        if 'suggestion' in results:
            print(f"       {results['suggestion']}")
    else:
        print(f"\n[DATA] QUANTUM OUTPUT ANALYSIS:")
        print(f"   Avg quantum variance: {results['avg_quantum_variance']:.4e}")
        print(f"   Avg classical variance: {results['avg_classical_variance']:.4e}")


def cmd_compare(args):
    """Compare multiple models."""
    from ..analysis.strategy import compare_decisions
    from ..networks.loader import load_policy_network, NetworkLoadError
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    comparison_dir = Path(args.dir)
    if not comparison_dir.exists():
        print(f"[ERROR] Directory not found: {comparison_dir}")
        return
    
    # Find models in subdirectories
    models = {}
    
    for subdir in comparison_dir.iterdir():
        if subdir.is_dir():
            final_models = list(subdir.glob('final_*.pth'))
            if final_models:
                model_path = sorted(final_models, key=lambda x: x.stat().st_mtime)[-1]
                
                # Load using network loader
                try:
                    policy_net = load_policy_network(str(model_path), strict=False)
                    models[subdir.name] = policy_net
                    print(f"[OK] Loaded: {subdir.name}")
                except NetworkLoadError as e:
                    print(f"[WARN] Failed to load {subdir.name}: {e}")
    
    if len(models) < 2:
        print("[WARN] Need at least 2 models for comparison")
        return
    
    # Compare decisions
    results = compare_decisions(
        models,
        save_path=str(comparison_dir / 'decision_comparison.png'),
        show=args.show
    )
    
    if 'agreements' in results:
        print(f"\n[DATA] DECISION AGREEMENT:")
        for pair, agreement in results['agreements'].items():
            print(f"   {pair}: {agreement:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Blackjack RL Experiment CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a hybrid model:
    python -m blackjack_experiment.cli.compare train --type hybrid --episodes 5000
    
  Evaluate a checkpoint:
    python -m blackjack_experiment.cli.compare eval --checkpoint results/final.pth
    
  Analyze learning progression:
    python -m blackjack_experiment.cli.compare analyze --dir results/hybrid/ --learning --plot
    
  Compare models:
    python -m blackjack_experiment.cli.compare compare --dir results/comparison_xyz

Note: Network architecture (n_qubits, n_layers) is controlled by class defaults
in the network files, not CLI arguments. Modify hybrid/config.py to change architecture.
When loading checkpoints, architecture is read from the saved config.
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--type', choices=['classical', 'minimal', 'hybrid'], 
                             default='classical', help='Network type')
    train_parser.add_argument('--episodes', type=int, default=10000, help='Training episodes')
    train_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    train_parser.add_argument('--checkpoint-count', type=int, default=180, 
                             help='Number of evenly-spaced checkpoints (default: 180 for 10s @ 18fps timelapse)')
    train_parser.add_argument('--eval-freq', type=int, default=500, help='Evaluation frequency')
    train_parser.add_argument('--output', type=str, help='Output directory')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a checkpoint')
    eval_parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    eval_parser.add_argument('--episodes', type=int, default=1000, help='Evaluation episodes')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze model(s)')
    analyze_parser.add_argument('--checkpoint', help='Single checkpoint to analyze')
    analyze_parser.add_argument('--dir', help='Directory with checkpoints')
    analyze_parser.add_argument('--learning', action='store_true', help='Run learning analysis')
    analyze_parser.add_argument('--decisions', action='store_true', help='Run decision analysis')
    analyze_parser.add_argument('--quantum', action='store_true', help='Run quantum analysis')
    analyze_parser.add_argument('--plot', action='store_true', help='Generate plots')
    analyze_parser.add_argument('--save', action='store_true', help='Save results to JSON')
    analyze_parser.add_argument('--max-checkpoints', type=int, default=20, help='Max checkpoints to analyze')
    analyze_parser.add_argument('--n-samples', type=int, default=100, help='Samples for quantum analysis')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--dir', required=True, help='Comparison directory')
    compare_parser.add_argument('--show', action='store_true', help='Show plots')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'eval':
        cmd_eval(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'compare':
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
