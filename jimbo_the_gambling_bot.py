"""
Jimbo the Gambling Bot - Single-File Hybrid Quantum Network
For presentation purposes.

Hardcoded Configuration (Defaults):
- Qubits: 5
- Layers: 4
- Encoding: Compact (3 inputs)
- Compression: Full (Encoder output = 2 * Qubits)
- Entanglement: Linear
- Measurement: Pauli Z
- Data Re-uploading: True
- Transform: Arctan (Scale 2.0)
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import random
from enum import Enum
from typing import List, Optional, Dict, Tuple, Union

# =============================================================================
# UTILITIES & CONTROLS
# =============================================================================

class BypassMode(Enum):
    """Bypass modes for quantum circuit during forward pass."""
    NONE = 'none'           # Normal quantum processing
    ZEROS = 'zeros'         # Replace quantum output with zeros
    ENCODER = 'encoder'     # Pass encoder output directly (truncated/padded)
    NOISE = 'noise'         # Replace with random noise in [-1, 1]
    SCRAMBLER = 'scrambler' # Freeze encoder with random weights (data scrambler)

def encode_blackjack_state(state: Union[Tuple, List, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Encode a Blackjack state into a tensor using 'compact' encoding (3 inputs)."""
    player_sum_dim = 32
    dealer_card_dim = 11
    
    if isinstance(state, tuple):
        states = [state]
        batch_size = 1
    elif isinstance(state, (list, np.ndarray)):
        if len(state) > 0 and isinstance(state[0], tuple):
            states = state
            batch_size = len(state)
        else:
            return torch.tensor(state, dtype=torch.float32) if not isinstance(state, torch.Tensor) else state.float()
    elif isinstance(state, torch.Tensor):
        return state.float()
    else:
        raise ValueError(f"Unexpected state type: {type(state)}")

    features = torch.zeros(batch_size, 3, dtype=torch.float32)
    
    for i, (player_sum, dealer_card, usable_ace) in enumerate(states):
        p_norm = float(player_sum) / max(1.0, (player_sum_dim - 1))
        d_norm = float(dealer_card) / max(1.0, (dealer_card_dim - 1))
        a_norm = 1.0 if usable_ace else 0.0

        features[i, 0] = float(p_norm)
        features[i, 1] = float(d_norm)
        features[i, 2] = float(a_norm)

    return features

class FreezeControls:
    """Mixin for freezing/unfreezing network components."""
    def freeze_component(self, component: str):
        if component == 'classical':
            self.freeze_component('encoder')
            self.freeze_component('postprocessing')
            return
        self._frozen_components.add(component)
        if component == 'encoder':
            for p in self.feature_encoder.parameters(): p.requires_grad = False
        elif component == 'quantum':
            self.weights.requires_grad = False
        elif component == 'postprocessing':
            for p in self.postprocessing.parameters(): p.requires_grad = False
    
    def unfreeze_component(self, component: str):
        if component == 'all':
            for c in ['encoder', 'quantum', 'postprocessing']: self.unfreeze_component(c)
            return
        if component == 'classical':
            self.unfreeze_component('encoder')
            self.unfreeze_component('postprocessing')
            return
        self._frozen_components.discard(component)
        if component == 'encoder':
            for p in self.feature_encoder.parameters(): p.requires_grad = True
        elif component == 'quantum':
            self.weights.requires_grad = True
        elif component == 'postprocessing':
            for p in self.postprocessing.parameters(): p.requires_grad = True
    
    def get_frozen_components(self) -> set:
        return self._frozen_components.copy()

class BypassControls:
    """Mixin for bypass mode control."""
    def set_bypass_mode(self, mode: Union[str, BypassMode]):
        if isinstance(mode, str): mode = BypassMode(mode.lower())
        self._bypass_mode = mode
        if mode == BypassMode.SCRAMBLER: self.initialize_scrambler_mode()
    
    def initialize_scrambler_mode(self):
        for module in self.feature_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -1.0, 1.0)
                if module.bias is not None: nn.init.uniform_(module.bias, -0.5, 0.5)
        self.freeze_component('encoder')
    
    def get_bypass_mode(self) -> BypassMode:
        return self._bypass_mode
    
    def get_quantum_output_with_bypass(self, encoded: torch.Tensor, batch_size: int) -> torch.Tensor:
        quantum_output_dim = self.n_qubits
        if self._bypass_mode == BypassMode.NONE: return self._run_quantum(encoded)
        elif self._bypass_mode == BypassMode.ZEROS:
            return torch.zeros(batch_size, quantum_output_dim, device=encoded.device, dtype=encoded.dtype)
        elif self._bypass_mode == BypassMode.ENCODER:
            if encoded.shape[-1] >= quantum_output_dim: return encoded[:, :quantum_output_dim]
            else:
                padding = torch.zeros(batch_size, quantum_output_dim - encoded.shape[-1], device=encoded.device, dtype=encoded.dtype)
                return torch.cat([encoded, padding], dim=-1)
        elif self._bypass_mode == BypassMode.NOISE:
            return torch.rand(batch_size, quantum_output_dim, device=encoded.device, dtype=encoded.dtype) * 2 - 1
        else: return self._run_quantum(encoded)

class QuantumDropout:
    """Mixin for quantum weight dropout."""
    def set_quantum_dropout(self, rate: float):
        self._quantum_dropout_rate = max(0.0, min(1.0, rate))
    
    def get_quantum_dropout(self) -> float:
        return self._quantum_dropout_rate
    
    def apply_quantum_dropout(self, weights: torch.Tensor) -> torch.Tensor:
        if self._quantum_dropout_rate <= 0.0 or not self.training: return weights
        mask_shape = weights.shape
        n_weights = weights.numel()
        n_frozen = int(n_weights * self._quantum_dropout_rate)
        frozen_indices = random.sample(range(n_weights), n_frozen)
        mask = torch.ones(n_weights, device=weights.device)
        mask[frozen_indices] = 0.0
        mask = mask.reshape(mask_shape)
        return mask * weights + (1 - mask) * weights.detach()

# =============================================================================
# MAIN NETWORKS
# =============================================================================

class UniversalBlackjackHybridPolicyNetwork(
    nn.Module,
    FreezeControls,
    BypassControls,
    QuantumDropout
):
    """Hybrid quantum-classical policy network for Blackjack-v1."""
    
    def __init__(self, n_qubits: int = 5, n_layers: int = 4):
        super().__init__()
        
        self._bypass_mode = BypassMode.NONE
        self._frozen_components = set()
        self._quantum_dropout_rate = 0.0
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weights = nn.Parameter(
            (torch.rand(n_layers, n_qubits, 2) - 0.5) * np.pi)
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(3, 2 * n_qubits))
        
        dev = qml.device('default.qubit', wires=n_qubits)
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            
            for i in range(n_qubits):
                qml.RY(torch.arctan(inputs[i, 0]) * 2.0, wires=i)
                qml.RZ(torch.arctan(inputs[i, 1]) * 2.0, wires=i)
            
            qml.Barrier(wires=range(n_qubits), only_visual=True)

            for layer in range(n_layers):
                
                for i in range(n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                if layer < n_layers - 1:
                    qml.Barrier(wires=range(n_qubits), only_visual=True)
                    for i in range(n_qubits):
                        qml.RY(torch.arctan(inputs[i, 0]) * 2.0, wires=i)
                        qml.RZ(torch.arctan(inputs[i, 1]) * 2.0, wires=i)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        self.quantum_circuit = circuit
        
        self.postprocessing = nn.Sequential(
            nn.Linear(n_qubits, 2)
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, state, return_encoding_stats: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        # 1. Encode State
        x = encode_blackjack_state(state)
        batch_size = x.shape[0]
        
        # 2. Feature Encoder
        encoded = self.feature_encoder(x)
        
        # Stats (optional)
        stats = self._encoding_stats(encoded) if return_encoding_stats else {}
        
        # 3. Quantum Circuit (with bypass support)
        quantum_out = self.get_quantum_output_with_bypass(encoded, batch_size)
        
        # 4. Postprocessing
        logits = self.postprocessing(quantum_out)
        probs = self.softmax(logits)
        
        return (probs, stats) if return_encoding_stats else probs
    
    def _run_quantum(self, encoded: torch.Tensor) -> torch.Tensor:
        batch_size = encoded.shape[0]
        # Reshape [batch, 2*n_qubits] -> [batch, n_qubits, 2] for dual-axis encoding
        inputs = encoded.reshape(batch_size, self.n_qubits, 2)
        
        effective_weights = self.apply_quantum_dropout(self.weights)
        
        outputs = []
        for i in range(batch_size):
            out = self.quantum_circuit(inputs[i], effective_weights)
            if isinstance(out, (list, tuple)): out = torch.stack(out)
            outputs.append(out.float())
        
        return torch.stack(outputs)
    
    def _encoding_stats(self, encoded: torch.Tensor) -> Dict[str, torch.Tensor]:
        if encoded.shape[0] <= 1:
            return {'encoding_variance': torch.tensor(0.0), 'pairwise_distance': torch.tensor(0.0)}
        variance = torch.var(encoded, dim=0).mean()
        sample = encoded[:32] if encoded.shape[0] > 32 else encoded
        pairwise = torch.pdist(sample).mean()
        return {'encoding_variance': variance, 'pairwise_distance': pairwise}
    
    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Parameter count by component."""
        enc = sum(p.numel() for p in self.feature_encoder.parameters() if p.requires_grad)
        qc = self.weights.numel()
        post = sum(p.numel() for p in self.postprocessing.parameters() if p.requires_grad)
        
        return {
            'feature_encoder': enc,
            'quantum_circuit': qc,
            'postprocessing': post,
            'total': enc + qc + post
        }

    def get_config_summary(self) -> str:
        p = self.get_parameter_breakdown()
        return f"Jimbo: {self.n_qubits} qubits, {self.n_layers} layers. Total params: {p['total']}"

class UniversalBlackjackHybridValueNetwork(nn.Module):
    """Classical value network."""
    def __init__(self):
        super().__init__()
        # Default architecture: 3 -> 4 -> 6 -> 4 -> 1
        hidden_sizes = [4, 6, 4]
        layers = []
        prev = 3 # Compact input dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.Tanh()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        x = encode_blackjack_state(state)
        return self.network(x).squeeze(-1)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Initializing Jimbo the Gambling Bot...")
    
    # Create the bot with default settings (5 qubits, 4 layers)
    jimbo = UniversalBlackjackHybridPolicyNetwork()
    
    print("\nJimbo's Brain Configuration:")
    print(jimbo.get_config_summary())
    
    # Test with a sample hand
    # Player: 16, Dealer: 10, Usable Ace: No
    sample_hand = (16, 10, False)
    
    print(f"\nThinking about hand: {sample_hand}...")
    probs = jimbo(sample_hand)
    
    print(f"Action Probabilities: Stand={probs[0,0]:.4f}, Hit={probs[0,1]:.4f}")
    
    action = "Hit" if probs[0,1] > probs[0,0] else "Stand"
    print(f"Jimbo says: {action}!")
