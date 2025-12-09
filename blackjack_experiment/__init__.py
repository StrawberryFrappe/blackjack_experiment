"""
Blackjack RL Experiment Package

A streamlined reinforcement learning framework for comparing classical
and quantum-hybrid neural networks on the Blackjack-v1 environment.

Structure:
    networks/       - Neural network architectures (classical & hybrid)
    core/           - Training infrastructure (agent, trainer, config)
    analysis/       - Analysis tools (learning, decisions, quantum)
    cli/            - Command-line interface
    run.py          - Unified entry point

Quick Start:
    from blackjack_experiment.run import quick_train, run_comparison
    
    # Train a single model
    results = quick_train('hybrid', episodes=5000, output_dir='results/hybrid')
    
    # Run comparison
    results = run_comparison(episodes=10000, output_base='results/comparison')

CLI Usage:
    python -m blackjack_experiment.run --compare --episodes 5000
    python -m blackjack_experiment.cli.compare train --type hybrid
"""

__version__ = '2.0.0'
__author__ = 'Blackjack RL Team'
