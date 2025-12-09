"""
CLI module for Blackjack experiments.

Available CLIs:
    python -m blackjack_experiment.run          # Main entry (compare, train, eval)
    python -m blackjack_experiment.cli.qfirst   # Quantum-first training
    python -m blackjack_experiment.cli.compare  # Legacy compare CLI
"""

from .compare import main as compare_main
from .qfirst import main as qfirst_main

__all__ = ['compare_main', 'qfirst_main']
