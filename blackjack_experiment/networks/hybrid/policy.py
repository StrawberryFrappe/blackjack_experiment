"""Main hybrid policy network - assembles all components."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Tuple, Union

from ..base import HybridPolicyNetworkBase, encode_blackjack_state
from .config import HybridDefaults, BypassMode
from .encoder import build_encoder, compute_encoder_dim
from .compression import compress_to_quantum_input
from .quantum import build_quantum_circuit
from .controls import FreezeControls, BypassControls, QuantumDropout


class UniversalBlackjackHybridPolicyNetwork(
    HybridPolicyNetworkBase,
    FreezeControls,
    BypassControls,
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
        n_qubits: int = HybridDefaults.N_QUBITS,
        n_layers: int = HybridDefaults.N_LAYERS,
        encoding: str = HybridDefaults.ENCODING,
        entanglement_strategy: str = HybridDefaults.ENTANGLEMENT,
        measurement_mode: str = HybridDefaults.MEASUREMENT,
        postprocessing_layers: Optional[List[int]] = None,
        encoder_layers: Optional[List[int]] = None,
        device_name: str = HybridDefaults.DEVICE,
        data_reuploading: bool = HybridDefaults.DATA_REUPLOADING,
        learnable_input_scaling: bool = HybridDefaults.LEARNABLE_INPUT_SCALING,
        encoder_compression: str = HybridDefaults.ENCODER_COMPRESSION
    ):
        """Initialize the hybrid network.
        
        Args:
            n_qubits: Number of qubits (default: 4)
            n_layers: Variational layers (default: 3)
            encoding: 'compact' (3 features) or 'one-hot' (45 features)
            entanglement_strategy: 'linear', 'cyclic', 'full', or 'none'
            measurement_mode: 'pauli_z' (n outputs) or 'amplitude' (2^n outputs)
            postprocessing_layers: Hidden sizes for postprocessing (None = minimal)
            encoder_layers: Hidden sizes for encoder (None = single layer)
            device_name: PennyLane device
            data_reuploading: Re-encode inputs at each layer (breaks linearity)
            learnable_input_scaling: Add α*x + β before quantum encoding
            encoder_compression: How to map encoder to quantum inputs
        """
        super().__init__()
        
        # Store configuration
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.entanglement_strategy = entanglement_strategy
        self.measurement_mode = measurement_mode
        self.device_name = device_name
        self.data_reuploading = data_reuploading
        self.learnable_input_scaling = learnable_input_scaling
        self.encoder_compression = encoder_compression
        self.n_actions = 2
        
        # Quantum encoding strategy
        self.single_axis_encoding = getattr(HybridDefaults, 'SINGLE_AXIS_ENCODING', True)
        self.encoding_transform = getattr(HybridDefaults, 'ENCODING_TRANSFORM', 'arctan')
        self.reuploading_transform = getattr(HybridDefaults, 'REUPLOADING_TRANSFORM', 'arctan')
        self.encoding_scale = getattr(HybridDefaults, 'ENCODING_SCALE', 2.0)
        
        # Control state
        self._bypass_mode = BypassMode.NONE
        self._frozen_components = set()
        self._quantum_dropout_rate = 0.0
        
        # Compute dimensions
        self.input_dim = 3 if encoding == 'compact' else 45
        self.encoder_output_dim = compute_encoder_dim(
            n_qubits, self.input_dim, encoder_compression, self.single_axis_encoding
        )
        self.quantum_input_dim = n_qubits * 2
        self.quantum_output_dim = n_qubits if measurement_mode == 'pauli_z' else 2**n_qubits
        
        # Auto-select encoder layers based on encoding if not specified
        if encoder_layers is None:
            encoder_layers = (
                HybridDefaults.ENCODER_LAYERS_ONE_HOT if encoding == 'one-hot'
                else HybridDefaults.ENCODER_LAYERS_COMPACT
            )
        
        # Build components
        self.feature_encoder = build_encoder(self.input_dim, self.encoder_output_dim, encoder_layers)
        self.encoder_layers = encoder_layers  # Store for config summary
        self.input_scale, self.input_bias = self._build_scaling()
        
        # Initialize quantum weights: uniform in [-π/2, π/2]
        self.weights = nn.Parameter(
            (torch.rand(n_layers, n_qubits, 2) - 0.5) * np.pi
        )
        
        self.quantum_circuit = build_quantum_circuit(
            n_qubits, n_layers, device_name, entanglement_strategy,
            measurement_mode, data_reuploading, self.single_axis_encoding,
            self.encoding_transform, self.reuploading_transform, self.encoding_scale
        )
        self.postprocessing = self._build_postprocessing(postprocessing_layers)
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
        
        # Quantum (with bypass support)
        quantum_out = self.get_quantum_output_with_bypass(
            encoded, batch_size, self.quantum_output_dim
        )
        
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
        quantum_out = self.get_quantum_output_with_bypass(
            scaled_out, batch_size, self.quantum_output_dim
        )
        logits = self.postprocessing(quantum_out)
        probs = self.softmax(logits)
        
        return {
            'input': x,
            'feature_encoder_output': encoder_out,
            'compressed_encoder_output': compressed_out,
            'scaled_encoder_output': scaled_out,
            'quantum_output': quantum_out,
            'bypass_mode': self._bypass_mode.value,
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
        post_arch = ' → '.join([str(self.quantum_output_dim)] + 
                               [str(m.out_features) for m in self.postprocessing 
                                if isinstance(m, nn.Linear)])
        
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║  HYBRID NETWORK CONFIGURATION                                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Quantum Circuit                                                  ║
║    Qubits: {self.n_qubits}    Layers: {self.n_layers}    Entanglement: {self.entanglement_strategy:<8}        ║
║    Measurement: {self.measurement_mode:<10}  Data Re-uploading: {str(self.data_reuploading):<5}       ║
╠══════════════════════════════════════════════════════════════════╣
║  Classical Components                                             ║
║    Encoding: {self.encoding:<8}  Input Scaling: {str(self.learnable_input_scaling):<5}               ║
║    Encoder Compression: {self.encoder_compression:<8}  Single-axis: {str(self.single_axis_encoding):<5}        ║
║    Input: {self.input_dim} → Encoder: {self.encoder_output_dim} → Compressed: {self.quantum_input_dim} → Quantum: {self.quantum_output_dim}  ║
║    Postprocessing: {post_arch:<40}  ║
╠══════════════════════════════════════════════════════════════════╣
║  Parameters                                                       ║
║    Encoder: {p['feature_encoder']:<4}  Scaling: {p['input_scaling']:<4}  Quantum: {p['quantum_circuit']:<4}  Post: {p['postprocessing']:<4}   ║
║    TOTAL: {p['total']:<4}                                                     ║
╚══════════════════════════════════════════════════════════════════╝"""
