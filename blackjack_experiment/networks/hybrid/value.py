"""Classical value network for hybrid policy training."""

import torch.nn as nn
from typing import List

from ..base import encode_blackjack_state


class UniversalBlackjackHybridValueNetwork(nn.Module):
    """Classical value network for hybrid policy training.
    
    Default architecture: 3 → 4 → 6 → 4 → 1
    """
    
    def __init__(
        self,
        hidden_sizes: List[int] = None,
        activation: str = 'tanh',
        encoding: str = 'compact'
    ):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [4, 6, 4]
        
        self.encoding = encoding
        self.hidden_sizes = hidden_sizes
        self.input_dim = 3 if encoding == 'compact' else 45
        
        # Build network
        layers = []
        prev = self.input_dim
        
        activations = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'elu': nn.ELU}
        act_fn = activations.get(activation.lower(), nn.Tanh)
        
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), act_fn()])
            prev = h
        
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)
    
    def encode_state(self, state):
        return encode_blackjack_state(
            state, encoding=self.encoding,
            player_sum_dim=32, dealer_card_dim=11, usable_ace_dim=2
        )
    
    def forward(self, state):
        x = self.encode_state(state)
        return self.network(x).squeeze(-1)
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
