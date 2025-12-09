"""Classical feature encoder for hybrid networks."""

import torch
import torch.nn as nn
from typing import List, Optional


def build_encoder(
    input_dim: int,
    output_dim: int,
    encoder_layers: Optional[List[int]] = None
) -> nn.Sequential:
    """Build classical encoder: input_dim → output_dim.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Encoder output dimension
        encoder_layers: Optional hidden layer sizes
        
    Returns:
        Sequential encoder network
        
    NOTE: No activation on final layer - output is unbounded.
    The quantum encoding layer will apply nonlinear transformations
    (arctan, sin, cos) to map these to rotation angles.
    """
    layers = []
    prev = input_dim
    
    if encoder_layers:
        for h in encoder_layers:
            layers.extend([nn.Linear(prev, h), nn.Tanh()])
            prev = h
    
    # Final layer: NO activation - raw linear output for quantum encoding
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def compute_encoder_dim(
    n_qubits: int,
    input_dim: int,
    encoder_compression: str,
    single_axis_encoding: bool
) -> int:
    """Compute encoder output dimension based on configuration.
    
    Args:
        n_qubits: Number of qubits
        input_dim: Raw input dimension
        encoder_compression: Compression mode
        single_axis_encoding: Whether to use single-axis encoding
        
    Returns:
        Encoder output dimension
        
    Logic:
        - If single_axis: always n_qubits (one value per qubit for RY)
        - Otherwise: depends on compression mode
    """
    if single_axis_encoding:
        # Single-axis: encoder outputs only n_qubits values (one per qubit)
        # These become RY rotations; RZ is purely variational
        return n_qubits
    
    # Original dual-axis encoding
    if encoder_compression == 'full':
        return 2 * n_qubits  # Full compression: 2 * n_qubits (e.g., 3→8 for 4 qubits)
    elif encoder_compression == 'minimal':
        return n_qubits      # 3→4, will duplicate
    elif encoder_compression == 'direct':
        return input_dim     # 3→3, will pad
    elif encoder_compression == 'shared':
        return n_qubits      # 3→4, same for RY/RZ
    else:
        return 2 * n_qubits  # Fallback to full
