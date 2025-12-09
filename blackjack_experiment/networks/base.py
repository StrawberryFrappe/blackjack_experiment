"""Base classes and encoding utilities for Blackjack neural networks.

This module provides:
- encode_blackjack_state(): Shared encoding function for all networks
- BasePolicyNetwork: Abstract base class for policy networks
- BaseValueNetwork: Abstract base class for value networks
- HybridPolicyNetworkBase: Abstract base class for hybrid quantum-classical networks
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple, List


# =============================================================================
# STATE ENCODING UTILITIES
# =============================================================================

def encode_blackjack_state(
    state: Union[Tuple, List, np.ndarray, torch.Tensor], 
    encoding: str = 'one-hot',
    player_sum_dim: int = 32,
    dealer_card_dim: int = 11,
    usable_ace_dim: int = 2
) -> torch.Tensor:
    """
    Encode a Blackjack state into a tensor based on the specified encoding strategy.
    
    Args:
        state: Input state. Can be a tuple (player_sum, dealer_card, usable_ace),
               a list of tuples, a numpy array, or a torch Tensor.
        encoding: 'one-hot' or 'compact'.
        player_sum_dim: Dimension for player sum (used in one-hot).
        dealer_card_dim: Dimension for dealer card (used in one-hot).
        usable_ace_dim: Dimension for usable ace (used in one-hot).
        
    Returns:
        torch.Tensor: Encoded state tensor.
    """
    # Determine input dimensions
    if encoding == 'one-hot':
        input_dim = player_sum_dim + dealer_card_dim + usable_ace_dim
    elif encoding == 'compact':
        input_dim = 3
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    # Handle different input types
    if isinstance(state, tuple):
        states = [state]
        batch_size = 1
    elif isinstance(state, (list, np.ndarray)):
        if len(state) > 0 and isinstance(state[0], tuple):
            states = state
            batch_size = len(state)
        else:
            # Assume it's already a flat array/tensor-like structure
            return torch.tensor(state, dtype=torch.float32) if not isinstance(state, torch.Tensor) else state.float()
    elif isinstance(state, torch.Tensor):
        if state.shape[-1] == input_dim:
            return state.float()
        return state.float()
    else:
        raise ValueError(f"Unexpected state type: {type(state)}")

    features = torch.zeros(batch_size, input_dim, dtype=torch.float32)
    
    for i, (player_sum, dealer_card, usable_ace) in enumerate(states):
        player_sum = float(player_sum)
        dealer_card = float(dealer_card)
        usable_ace = float(usable_ace)

        if encoding == 'one-hot':
            player_idx = int(player_sum)
            if 0 <= player_idx < player_sum_dim:
                features[i, player_idx] = 1.0

            dealer_idx = int(dealer_card) + player_sum_dim
            if player_sum_dim <= dealer_idx < player_sum_dim + dealer_card_dim:
                features[i, dealer_idx] = 1.0

            ace_idx = int(usable_ace) + player_sum_dim + dealer_card_dim
            if player_sum_dim + dealer_card_dim <= ace_idx < input_dim:
                features[i, ace_idx] = 1.0
        
        elif encoding == 'compact':
            p_norm = player_sum / max(1.0, (player_sum_dim - 1))
            d_norm = dealer_card / max(1.0, (dealer_card_dim - 1))
            a_norm = 1.0 if usable_ace else 0.0

            features[i, 0] = float(p_norm)
            features[i, 1] = float(d_norm)
            features[i, 2] = float(a_norm)

    return features


# =============================================================================
# BASE NETWORK CLASSES
# =============================================================================

class BasePolicyNetwork(nn.Module, ABC):
    """Abstract base class for policy networks."""
    
    @abstractmethod
    def forward(self, state):
        """
        Compute action probabilities for given state(s).
        
        Args:
            state: State or batch of states
            
        Returns:
            Action probabilities (batch_size, n_actions)
        """
        pass
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaseValueNetwork(nn.Module, ABC):
    """Abstract base class for value networks (for A2C, PPO, etc.)."""
    
    @abstractmethod
    def forward(self, state):
        """
        Compute state value for given state(s).
        
        Args:
            state: State or batch of states
            
        Returns:
            State values (batch_size, 1)
        """
        pass
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# HYBRID NETWORK BASE CLASS
# =============================================================================

class HybridPolicyNetworkBase(BasePolicyNetwork):
    """
    Abstract base class for all hybrid quantum-classical policy networks.
    
    Enforces consistent architecture across quantum variants.
    
    Subclasses MUST set:
    - self.n_qubits (int)
    - self.feature_encoder (nn.Module)
    - self.postprocessing (nn.Module)
    - self.quantum_circuit (callable)
    
    Provides:
    - Architecture validation
    - Consistent forward signature
    - Dimension properties
    """
    
    @abstractmethod
    def __init__(self, n_layers: int = 4, device: str = 'default.qubit'):
        """
        Initialize hybrid network.
        
        Subclasses must call validate_architecture() at end of __init__.
        """
        super().__init__()
        self.n_qubits: int = None
        self.n_layers: int = n_layers
        self.device_name: str = device
    
    def validate_architecture(self) -> None:
        """
        Validate that network architecture is consistent with n_qubits.
        
        Checks:
        - n_qubits attribute exists and is valid
        - feature_encoder output dimension = n_qubits * 2
        - postprocessing input dimension matches measurement mode
        - All required components exist
        
        Raises:
            ValueError: If architecture is invalid
        """
        if not hasattr(self, 'n_qubits') or self.n_qubits is None:
            raise ValueError(
                f"{type(self).__name__} missing n_qubits attribute. "
                f"Subclass must set self.n_qubits in __init__"
            )
        
        if self.n_qubits not in [3, 4, 5, 6, 7, 8, 10, 15]:
            raise ValueError(
                f"Unsupported qubit count: {self.n_qubits}. "
                f"Supported: 3, 4, 5, 6, 7, 8, 10, 15"
            )
        
        required_components = ['feature_encoder', 'postprocessing', 'quantum_circuit']
        for component in required_components:
            if not hasattr(self, component):
                raise ValueError(f"{type(self).__name__} missing {component} component")
        
        # Verify feature encoder output dimension
        expected_encoder_out = self.n_qubits * 2
        try:
            actual_encoder_out = self.feature_encoder[-2].out_features
        except (AttributeError, IndexError, TypeError):
            raise ValueError(
                f"{type(self).__name__}: Cannot determine feature_encoder output dimension. "
                f"Expected final Linear layer before activation."
            )
        
        if actual_encoder_out != expected_encoder_out:
            raise ValueError(
                f"Feature encoder output dimension mismatch in {type(self).__name__} "
                f"({self.n_qubits}-qubit network):\n"
                f"  Expected: {expected_encoder_out} (n_qubits * 2)\n"
                f"  Got: {actual_encoder_out}"
            )
    
    @property
    def encoding_dimension(self) -> int:
        """Return the feature encoding dimension (n_qubits * 2)."""
        return self.n_qubits * 2
    
    @property
    def quantum_output_dimension(self) -> int:
        """Return the quantum output dimension (depends on measurement mode)."""
        if hasattr(self, 'measurement_mode') and self.measurement_mode == 'pauli_z':
            return self.n_qubits
        return 2 ** self.n_qubits
    
    def forward(self, state, return_encoding_stats: bool = False):
        """
        Forward pass through hybrid network.
        
        Args:
            state: Input state (Blackjack tuple or tensor)
            return_encoding_stats: If True, return encoding diversity metrics
        
        Returns:
            If return_encoding_stats=False:
                torch.Tensor: Action probabilities (batch_size, 2)
            
            If return_encoding_stats=True:
                tuple: (action_probs, encoding_stats_dict)
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward() "
            f"with this exact signature"
        )
    
    def get_parameter_breakdown(self) -> dict:
        """
        Get breakdown of parameters by component.
        
        Returns:
            dict with parameter counts for each component
        """
        feature_encoder_params = sum(
            p.numel() for p in self.feature_encoder.parameters() if p.requires_grad
        )
        quantum_params = self.weights.numel() if hasattr(self, 'weights') else 0
        postprocessing_params = sum(
            p.numel() for p in self.postprocessing.parameters() if p.requires_grad
        )
        
        return {
            'feature_encoder': feature_encoder_params,
            'quantum_circuit': quantum_params,
            'postprocessing': postprocessing_params,
            'total': feature_encoder_params + quantum_params + postprocessing_params
        }
