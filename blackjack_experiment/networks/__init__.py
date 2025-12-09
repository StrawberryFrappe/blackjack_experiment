"""Neural network architectures for Blackjack RL.

This module provides:
- Base classes and encoding utilities (base.py)
- Classical neural networks (classical.py)
- Hybrid quantum-classical networks (hybrid/ folder)
- Checkpoint loading utilities (loader.py)
"""

from .base import BasePolicyNetwork, BaseValueNetwork, encode_blackjack_state
from .classical import (
    BlackjackClassicalPolicyNetwork,
    BlackjackClassicalValueNetwork,
    BlackjackMinimalClassicalPolicyNetwork,
    BlackjackMinimalClassicalValueNetwork,
)
from .hybrid import UniversalBlackjackHybridPolicyNetwork, UniversalBlackjackHybridValueNetwork
from .loader import (
    load_policy_network,
    load_networks_from_checkpoint,
    extract_network_config,
    get_network_info,
    NetworkLoadError,
)

__all__ = [
    'BasePolicyNetwork',
    'BaseValueNetwork',
    'encode_blackjack_state',
    'BlackjackClassicalPolicyNetwork',
    'BlackjackClassicalValueNetwork',
    'BlackjackMinimalClassicalPolicyNetwork',
    'BlackjackMinimalClassicalValueNetwork',
    'UniversalBlackjackHybridPolicyNetwork',
    'UniversalBlackjackHybridValueNetwork',
    'load_policy_network',
    'load_networks_from_checkpoint',
    'extract_network_config',
    'get_network_info',
    'NetworkLoadError',
]
