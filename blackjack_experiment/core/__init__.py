"""
Core training infrastructure for Blackjack RL experiments.

This module provides:
- Configuration dataclasses (config.py)
- A2C Agent implementation (agent.py)
- Training loop and evaluation (trainer.py)
- Session management for outputs (session.py)
"""

from .config import Config, NetworkConfig, AgentConfig, TrainingConfig
from .agent import A2CAgent
from .trainer import Trainer, evaluate
from .session import SessionManager, ComparisonSessionManager

__all__ = [
    'Config',
    'NetworkConfig',
    'AgentConfig',
    'TrainingConfig',
    'A2CAgent',
    'Trainer',
    'evaluate',
    'SessionManager',
    'ComparisonSessionManager',
]
