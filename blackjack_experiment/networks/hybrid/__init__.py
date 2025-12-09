"""Hybrid network module - modular quantum-classical architecture.

Components:
- config.py: Default configuration and constants
- encoder.py: Classical feature encoder
- quantum.py: Quantum circuit builder
- compression.py: Encoder compression strategies
- controls.py: Freeze/bypass/dropout controls
- policy.py: Main policy network
- value.py: Classical value network
"""

from .policy import UniversalBlackjackHybridPolicyNetwork
from .value import UniversalBlackjackHybridValueNetwork
from .config import HybridDefaults

__all__ = [
    'UniversalBlackjackHybridPolicyNetwork',
    'UniversalBlackjackHybridValueNetwork',
    'HybridDefaults',
    'BypassMode'
]
