"""Main hybrid policy network - assembles all components."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Tuple, Union

from ..base import HybridPolicyNetworkBase, encode_blackjack_state
from ...config import HybridConfig as HybridArchitectureConfig
from .encoder import build_encoder, compute_encoder_dim
from .compression import compress_to_quantum_input
from .quantum import build_quantum_circuit
from .controls import FreezeControls, QuantumDropout


class UniversalBlackjackHybridPolicyNetwork(
    HybridPolicyNetworkBase,
    FreezeControls,
    QuantumDropout
):
    """Hybrid quantum-classical policy network for Blackjack-v1.
    
    Modular architecture with separated concerns:
    - Encoder: Classical feature extraction
    - Compression: Map encoder → quantum input dimensions
    - Quantum: Variational quantum circuit
    - Postprocessing: Quantum → action probabilities
    - Controls: Freeze/bypass/dropout mechanisms
    
    Usage:
        net = UniversalBlackjackHybridPolicyNetwork()
        probs = net((player_sum, dealer_card, usable_ace))  # → [P(stand), P(hit)]
        print(net.get_config_summary())
    """
    
    def __init__(
        self,
        config: Optional[HybridArchitectureConfig] = None,
        **kwargs
    ):
        """Initialize the hybrid network.
        
        Args:
            config: Optional HybridArchitectureConfig object.
            **kwargs: Overrides for configuration parameters.
        """
        super().__init__()
        
        # Merge config and kwargs
        cfg = config or HybridArchitectureConfig()
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        
        # Store configuration
        self.n_qubits = cfg.n_qubits
        self.n_layers = cfg.n_layers
        self.encoding = cfg.encoding
        self.entanglement_strategy = cfg.entanglement
        self.measurement_mode = cfg.measurement
        self.device_name = cfg.device
        self.data_reuploading = cfg.data_reuploading
        self.learnable_input_scaling = cfg.learnable_input_scaling
        self.encoder_compression = cfg.encoder_compression
        self.n_actions = 2
        
        # Quantum encoding strategy
        self.single_axis_encoding = cfg.single_axis_encoding
        self.encoding_transform = cfg.encoding_transform
        self.reuploading_transform = cfg.reuploading_transform
        self.encoding_scale = cfg.encoding_scale
        
        # Control state
        self._frozen_components = set()
        self._quantum_dropout_rate = 0.0
        
        # Compute dimensions
        self.input_dim = 3 if self.encoding == 'compact' else 45
        self.encoder_output_dim = compute_encoder_dim(
            self.n_qubits, self.input_dim, self.encoder_compression, self.single_axis_encoding
        )
        self.quantum_input_dim = self.n_qubits * 2
        self.quantum_output_dim = self.n_qubits if self.measurement_mode == 'pauli_z' else 2**self.n_qubits
        
        # Build components
        encoder_layers = cfg.encoder_layers
        if encoder_layers is None and self.encoding == 'one-hot':
            encoder_layers = [32, 16]
        
        self.feature_encoder = build_encoder(self.input_dim, self.encoder_output_dim, encoder_layers)
        self.encoder_layers = encoder_layers
        self.input_scale, self.input_bias = self._build_scaling()
        
        self.weights = nn.Parameter(
            (torch.rand(self.n_layers, self.n_qubits, 2) - 0.5) * np.pi
        )
        
        self.quantum_circuit = build_quantum_circuit(
            self.n_qubits, self.n_layers, self.device_name, self.entanglement_strategy,
            self.measurement_mode, self.data_reuploading, self.single_axis_encoding,
            self.encoding_transform, self.reuploading_transform, self.encoding_scale
        )
        self.postprocessing = self._build_postprocessing(cfg.postprocessing_layers)
        self.softmax = nn.Softmax(dim=-1)
        
    
    def _build_scaling(self) -> Tuple[Optional[nn.Parameter], Optional[nn.Parameter]]:
        """Build learnable input scaling: α*x + β."""
        if self.learnable_input_scaling:
            return (
                nn.Parameter(torch.ones(self.quantum_input_dim)),
                nn.Parameter(torch.zeros(self.quantum_input_dim))
            )
        self.register_buffer('_scale_placeholder', None)
        self.register_buffer('_bias_placeholder', None)
        return None, None
    
    def _build_postprocessing(self, layers: Optional[List[int]]) -> nn.Sequential:
        """Build classical postprocessing: quantum_output → n_actions."""
        if layers is None:
            layers = []  # Minimal: direct mapping, no hidden layers
        
        modules = []
        prev = self.quantum_output_dim
        
        for h in layers:
            modules.extend([nn.Linear(prev, h), nn.Tanh()])
            prev = h
        
        modules.append(nn.Linear(prev, self.n_actions))
        return nn.Sequential(*modules)
    
    def encode_state(self, state) -> torch.Tensor:
        """Encode Blackjack state to tensor."""
        return encode_blackjack_state(
            state, encoding=self.encoding,
            player_sum_dim=32, dealer_card_dim=11, usable_ace_dim=2
        )
    
    def forward(
        self,
        state,
        return_encoding_stats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Forward pass: state → action probabilities."""
        x = self.encode_state(state)
        batch_size = x.shape[0]
        
        # Encoder
        encoded = self.feature_encoder(x)
        
        # Compression/expansion to quantum_input_dim
        encoded = compress_to_quantum_input(
            encoded, self.n_qubits, self.encoder_compression, self.single_axis_encoding
        )
        
        # Scaling
        if self.learnable_input_scaling:
            encoded = self.input_scale * encoded + self.input_bias
        
        # Stats (optional)
        stats = self._encoding_stats(encoded) if return_encoding_stats else {}
        
        # Quantum
        quantum_out = self._run_quantum(encoded)
        
        # Postprocessing
        logits = self.postprocessing(quantum_out)
        probs = self.softmax(logits)
        
        return (probs, stats) if return_encoding_stats else probs
    
    def forward_with_intermediates(self, state) -> Dict[str, torch.Tensor]:
        """Forward pass returning all intermediate outputs."""
        x = self.encode_state(state)
        batch_size = x.shape[0]
        
        encoder_out = self.feature_encoder(x)
        compressed_out = compress_to_quantum_input(
            encoder_out, self.n_qubits, self.encoder_compression, self.single_axis_encoding
        )
        scaled_out = (self.input_scale * compressed_out + self.input_bias 
                      if self.learnable_input_scaling else compressed_out)
        quantum_out = self._run_quantum(scaled_out)
        logits = self.postprocessing(quantum_out)
        probs = self.softmax(logits)
        
        return {
            'input': x,
            'feature_encoder_output': encoder_out,
            'compressed_encoder_output': compressed_out,
            'scaled_encoder_output': scaled_out,
            'quantum_output': quantum_out,
            'action_probs': probs
        }
    
    def _run_quantum(self, encoded: torch.Tensor) -> torch.Tensor:
        """Run quantum circuit on encoded input."""
        batch_size = encoded.shape[0]
        inputs = encoded.reshape(batch_size, self.n_qubits, 2)
        
        # Apply quantum dropout if enabled
        effective_weights = self.apply_quantum_dropout(self.weights)
        
        outputs = []
        for i in range(batch_size):
            out = self.quantum_circuit(inputs[i], effective_weights)
            if isinstance(out, (list, tuple)):
                out = torch.stack(out)
            outputs.append(out.float())
        
        return torch.stack(outputs)
    
    def _encoding_stats(self, encoded: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute encoding diversity statistics."""
        if encoded.shape[0] <= 1:
            return {'encoding_variance': torch.tensor(0.0), 
                    'pairwise_distance': torch.tensor(0.0)}
        
        variance = torch.var(encoded, dim=0).mean()
        sample = encoded[:32] if encoded.shape[0] > 32 else encoded
        pairwise = torch.pdist(sample).mean()
        
        return {'encoding_variance': variance, 'pairwise_distance': pairwise}
    
    def get_num_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Parameter count by component."""
        enc = sum(p.numel() for p in self.feature_encoder.parameters() if p.requires_grad)
        qc = self.weights.numel()
        post = sum(p.numel() for p in self.postprocessing.parameters() if p.requires_grad)
        scale = (self.input_scale.numel() + self.input_bias.numel() 
                 if self.learnable_input_scaling else 0)
        
        return {
            'feature_encoder': enc,
            'input_scaling': scale,
            'quantum_circuit': qc,
            'postprocessing': post,
            'total': enc + scale + qc + post
        }
    
    def get_config_summary(self) -> str:
        """Human-readable configuration summary."""
        p = self.get_parameter_breakdown()
        post_arch = ' -> '.join([str(self.quantum_output_dim)] + 
                               [str(m.out_features) for m in self.postprocessing 
                                if isinstance(m, nn.Linear)])
        
        return f"""
+------------------------------------------------------------------+
|  HYBRID NETWORK CONFIGURATION                                     |
+------------------------------------------------------------------+
|  Quantum Circuit                                                  |
|    Qubits: {self.n_qubits}    Layers: {self.n_layers}    Entanglement: {self.entanglement_strategy:<8}        |
|    Measurement: {self.measurement_mode:<10}  Data Re-uploading: {str(self.data_reuploading):<5}       |
+------------------------------------------------------------------+
|  Classical Components                                             |
|    Encoding: {self.encoding:<8}  Input Scaling: {str(self.learnable_input_scaling):<5}               |
|    Encoder Compression: {self.encoder_compression:<8}  Single-axis: {str(self.single_axis_encoding):<5}        |
|    Input: {self.input_dim} -> Encoder: {self.encoder_output_dim} -> Compressed: {self.quantum_input_dim} -> Quantum: {self.quantum_output_dim}  |
|    Postprocessing: {post_arch:<40}  |
+------------------------------------------------------------------+
|  Parameters                                                       |
|    Encoder: {p['feature_encoder']:<4}  Scaling: {p['input_scaling']:<4}  Quantum: {p['quantum_circuit']:<4}  Post: {p['postprocessing']:<4}   |
|    TOTAL: {p['total']:<4}                                                     |
+------------------------------------------------------------------+"""
