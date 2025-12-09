"""Classical neural networks for Blackjack environment.

This module provides:
- BlackjackClassicalPolicyNetwork: Full classical policy network with one-hot encoding
- BlackjackClassicalValueNetwork: Full classical value network
- BlackjackMinimalClassicalPolicyNetwork: Minimal network mirroring hybrid structure
- BlackjackMinimalClassicalValueNetwork: Minimal value network
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Union, Tuple

from .base import BasePolicyNetwork, BaseValueNetwork, encode_blackjack_state


# =============================================================================
# FULL CLASSICAL NETWORKS
# =============================================================================

class BlackjackClassicalPolicyNetwork(BasePolicyNetwork):
    """
    Classical feedforward policy network for Blackjack-v1.
    
    Blackjack state: (player_sum, dealer_card, usable_ace)
    - player_sum: 4-21 (current hand total)
    - dealer_card: 1-10 (visible dealer card, Ace=1)
    - usable_ace: 0 or 1 (whether player has usable ace)
    
    Actions: 2 (stick=0, hit=1)
    
    The network uses one-hot encoding for input features.
    """
    
    def __init__(self, hidden_sizes=[4, 6, 8, 4], activation='tanh'):
        """
        Initialize classical policy network for Blackjack.
        
        Args:
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', or 'elu')
        """
        super().__init__()
        
        self.n_actions = 2
        self.hidden_sizes = hidden_sizes
        self.activation_name = activation
        
        self.player_sum_dim = 32
        self.dealer_card_dim = 11
        self.usable_ace_dim = 2
        self.input_dim = self.player_sum_dim + self.dealer_card_dim + self.usable_ace_dim
        
        # Build network layers
        layers = []
        prev_size = self.input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation(activation))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, self.n_actions))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation module by name."""
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def encode_state(self, state) -> torch.Tensor:
        """Encode Blackjack state using one-hot encoding."""
        return encode_blackjack_state(
            state,
            encoding='one-hot',
            player_sum_dim=self.player_sum_dim,
            dealer_card_dim=self.dealer_card_dim,
            usable_ace_dim=self.usable_ace_dim
        )
    
    def forward(self, state) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Blackjack state tuple(s) - (player_sum, dealer_card, usable_ace)
        
        Returns:
            Action probabilities (batch_size, 2)
        """
        x = self.encode_state(state)
        return self.network(x)


class BlackjackClassicalValueNetwork(BaseValueNetwork):
    """
    Classical value network for Blackjack-v1 (critic for actor-critic methods).
    
    Estimates the state value V(s) for a given Blackjack state.
    """
    
    def __init__(self, hidden_sizes=[32, 16, 8], activation='tanh'):
        """
        Initialize classical value network for Blackjack.
        
        Args:
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', or 'elu')
        """
        super().__init__()
        
        self.hidden_sizes = hidden_sizes
        self.activation_name = activation
        
        self.player_sum_dim = 32
        self.dealer_card_dim = 11
        self.usable_ace_dim = 2
        self.input_dim = self.player_sum_dim + self.dealer_card_dim + self.usable_ace_dim
        
        # Build network layers
        layers = []
        prev_size = self.input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation(activation))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation module by name."""
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def encode_state(self, state) -> torch.Tensor:
        """Encode Blackjack state using one-hot encoding."""
        return encode_blackjack_state(
            state,
            encoding='one-hot',
            player_sum_dim=self.player_sum_dim,
            dealer_card_dim=self.dealer_card_dim,
            usable_ace_dim=self.usable_ace_dim
        )
    
    def forward(self, state) -> torch.Tensor:
        """
        Forward pass through the value network.
        
        Args:
            state: Blackjack state tuple(s)
        
        Returns:
            State value estimate (batch_size,)
        """
        x = self.encode_state(state)
        value = self.network(x)
        return value.squeeze(-1)


# =============================================================================
# MINIMAL CLASSICAL NETWORKS (mirrors hybrid structure)
# =============================================================================

class BlackjackMinimalClassicalPolicyNetwork(nn.Module):
    """
    Classical-only policy network mirroring the hybrid network structure.

    Architecture:
    - Compact encoding: 3 normalized inputs
    - Feature encoder: input -> 4 (Tanh)
    - Hidden layer: 4 -> 6 (Tanh) [replaces quantum circuit]
    - Postprocessing: 6 -> 8 (Tanh) -> 2
    - Softmax for action probabilities
    
    This allows direct comparison with hybrid networks by using the same
    classical components while replacing the quantum circuit with a classical layer.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 1, device: str = 'cpu'):
        """
        Initialize minimal classical network.
        
        Args:
            n_qubits: Number of qubits (for compatibility with hybrid interface)
            n_layers: Number of layers (for compatibility)
            device: Device name (for compatibility)
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_name = device
        self.n_actions = 2

        # Compact encoding dimensions
        self.player_sum_dim = 32
        self.dealer_card_dim = 11
        self.usable_ace_dim = 2
        self.input_dim = 3  # compact encoding

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, 4),
            nn.Tanh(),
        )

        # Hidden layer (replaces quantum block)
        self.hidden_layer = nn.Sequential(
            nn.Linear(4, 6),
            nn.Tanh(),
        )

        # Postprocessing
        self.postprocessing = nn.Sequential(
            nn.Linear(6, 8),
            nn.Tanh(),
            nn.Linear(8, self.n_actions),
        )

        self.softmax = nn.Softmax(dim=-1)

    def encode_state(self, state: Union[tuple, list, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Encode Blackjack state using compact encoding (3 normalized values)."""
        return encode_blackjack_state(
            state,
            encoding='compact',
            player_sum_dim=self.player_sum_dim,
            dealer_card_dim=self.dealer_card_dim,
            usable_ace_dim=self.usable_ace_dim
        )

    def forward(self, state: Union[tuple, list, np.ndarray, torch.Tensor], 
                return_encoding_stats: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass.

        Args:
            state: Blackjack state(s)
            return_encoding_stats: If True, return encoding diversity metrics
            
        Returns:
            Action probabilities, optionally with encoding stats
        """
        x = self.encode_state(state)
        batch_size = x.shape[0]

        # Feature encoding
        quantum_encodings = self.feature_encoder(x)

        # Compute encoding stats for compatibility with hybrid networks
        encoding_stats: Dict = {}
        if batch_size > 1:
            encoding_variance = torch.var(quantum_encodings, dim=0).mean()
            if batch_size > 32:
                indices = torch.randperm(batch_size)[:32]
                sample_encodings = quantum_encodings[indices]
            else:
                sample_encodings = quantum_encodings
            pairwise_dist = torch.pdist(sample_encodings).mean()
            encoding_stats = {
                'encoding_variance': encoding_variance,
                'pairwise_distance': pairwise_dist,
            }
        else:
            encoding_stats = {
                'encoding_variance': torch.tensor(0.0), 
                'pairwise_distance': torch.tensor(0.0)
            }

        # Classical hidden layer (replaces quantum circuit)
        hidden = self.hidden_layer(quantum_encodings)

        # Postprocessing
        action_logits = self.postprocessing(hidden)
        action_probs = self.softmax(action_logits)

        if return_encoding_stats:
            return action_probs, encoding_stats

        return action_probs

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Get breakdown of parameters by component."""
        feature_encoder_params = sum(p.numel() for p in self.feature_encoder.parameters() if p.requires_grad)
        hidden_params = sum(p.numel() for p in self.hidden_layer.parameters() if p.requires_grad)
        postprocessing_params = sum(p.numel() for p in self.postprocessing.parameters() if p.requires_grad)
        return {
            'feature_encoder': feature_encoder_params,
            'hidden_layer': hidden_params,
            'postprocessing': postprocessing_params,
            'total': feature_encoder_params + hidden_params + postprocessing_params,
        }


class BlackjackMinimalClassicalValueNetwork(nn.Module):
    """
    Value network sized to match the minimal classical policy network.
    
    Uses one-hot encoding for input features.
    """

    def __init__(self, hidden_sizes=None, activation='tanh'):
        """
        Initialize minimal value network.
        
        Args:
            hidden_sizes: List of hidden layer sizes (default: [8, 8])
            activation: Activation function name
        """
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [8, 8]
        self.hidden_sizes = hidden_sizes
        self.activation_name = activation

        self.player_sum_dim = 32
        self.dealer_card_dim = 11
        self.usable_ace_dim = 2
        self.input_dim = self.player_sum_dim + self.dealer_card_dim + self.usable_ace_dim

        layers = []
        prev_size = self.input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation(activation))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation module by name."""
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def encode_state(self, state: Union[tuple, list, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Encode Blackjack state using one-hot encoding."""
        return encode_blackjack_state(
            state,
            encoding='one-hot',
            player_sum_dim=self.player_sum_dim,
            dealer_card_dim=self.dealer_card_dim,
            usable_ace_dim=self.usable_ace_dim
        )

    def forward(self, state: Union[tuple, list, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the value network.
        
        Args:
            state: Blackjack state tuple(s)
            
        Returns:
            State value estimate (batch_size,)
        """
        x = self.encode_state(state)
        value = self.network(x)
        return value.squeeze(-1)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
