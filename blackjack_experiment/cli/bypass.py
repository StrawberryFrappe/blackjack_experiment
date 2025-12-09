#!/usr/bin/env python
"""
Bypass Experiment CLI - Test if quantum circuit contributes.

Usage:
    python -m blackjack_experiment.cli.bypass
    python -m blackjack_experiment.cli.bypass --episodes 2000
    python -m blackjack_experiment.cli.bypass --modes zeros encoder
"""

import argparse
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Run bypass experiment to test quantum contribution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This experiment tests whether the quantum circuit contributes meaningfully:

  Phase A: Normal training (quantum + classical learning together)
  Phase B: Freeze quantum, continue training classical only
  Bypass Tests: Replace quantum output with zeros/encoder/noise

If Phase B matches Phase A, the quantum circuit is dead weight.

Examples:
    python -m blackjack_experiment.cli.bypass
    python -m blackjack_experiment.cli.bypass --episodes 2000
    python -m blackjack_experiment.cli.bypass --modes zeros encoder --output results/my_test
"""
    )
    
    parser.add_argument('-e', '--episodes', type=int, default=1000,
                       help='Episodes per phase (default: 1000)')
    parser.add_argument('-m', '--modes', nargs='+', default=['zeros', 'encoder', 'noise'],
                       choices=['zeros', 'encoder', 'noise', 'scrambler'],
                       help='Bypass modes to test')
    parser.add_argument('-o', '--output', type=str,
                       help='Output directory')
    parser.add_argument('-s', '--seed', type=int,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Import here to avoid slow startup
    import torch
    import numpy as np
    import random
    
    from ..analysis.experiments.bypass import BypassExperimentRunner, BypassExperimentConfig
    
    # Set seed
    seed = args.seed or random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Output dir
    output = args.output or f"results/bypass_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Configure and run
    config = BypassExperimentConfig(
        phase_a_episodes=args.episodes,
        phase_b_episodes=args.episodes,
        bypass_modes=args.modes
    )
    
    runner = BypassExperimentRunner(output_dir=output, config=config)
    runner.run_full_experiment()


if __name__ == '__main__':
    main()
