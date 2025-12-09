"""Encoder compression strategies for hybrid networks."""

import torch


def compress_to_quantum_input(
    encoded: torch.Tensor,
    n_qubits: int,
    encoder_compression: str,
    single_axis_encoding: bool
) -> torch.Tensor:
    """Map encoder output to quantum_input_dim based on compression mode.
    
    Args:
        encoded: [batch, encoder_output_dim] from feature encoder
        n_qubits: Number of qubits
        encoder_compression: Compression mode
        single_axis_encoding: Whether using single-axis encoding
        
    Returns:
        [batch, n_qubits*2] ready for quantum circuit
        
    Single-axis encoding (SINGLE_AXIS_ENCODING=True):
        Encoder outputs [n_qubits] values → duplicate to [n_qubits*2]
        First half: data-encoded RY angles
        Second half: zeros (RZ will be handled by variational weights only)
        Example: [x1,x2,x3,x4] → [x1,x2,x3,x4,0,0,0,0]
        
    Compression modes (dual-axis encoding):
        'full': Already correct size (encoder outputs 2*n_qubits, e.g., 3→8 for 4 qubits), pass through
        'minimal': 3→4, duplicate each value (x1,x2,x3,x4 → x1,x1,x2,x2,x3,x3,x4,x4)
        'direct': 3→3, pad with zeros (x1,x2,x3 → x1,x2,x3,0,0,0,0,0)
        'shared': 3→4, repeat for RY/RZ (x1,x2,x3,x4 → x1,x2,x3,x4,x1,x2,x3,x4)
    """
    batch_size = encoded.shape[0]
    quantum_input_dim = n_qubits * 2
    
    # Single-axis encoding: data only in RY, RZ stays at zero (weights handle it)
    if single_axis_encoding:
        # encoded is [batch, n_qubits]
        # Create [batch, n_qubits*2] with data in first half, zeros in second
        zeros = torch.zeros(
            batch_size,
            n_qubits,
            device=encoded.device,
            dtype=encoded.dtype
        )
        return torch.cat([encoded, zeros], dim=1)
    
    # Original dual-axis compression modes
    if encoder_compression == 'full':
        # Already correct size
        return encoded
    
    elif encoder_compression == 'minimal':
        # Duplicate each value: [b, n_qubits] → [b, n_qubits*2]
        # x1,x2,x3,x4 → x1,x1,x2,x2,x3,x3,x4,x4
        return encoded.repeat_interleave(2, dim=1)
    
    elif encoder_compression == 'direct':
        # Pad with zeros: [b, input_dim] → [b, n_qubits*2]
        if encoded.shape[1] >= quantum_input_dim:
            return encoded[:, :quantum_input_dim]
        else:
            padding = torch.zeros(
                batch_size,
                quantum_input_dim - encoded.shape[1],
                device=encoded.device,
                dtype=encoded.dtype
            )
            return torch.cat([encoded, padding], dim=1)
    
    elif encoder_compression == 'shared':
        # Repeat for RY and RZ: [b, n_qubits] → [b, n_qubits*2]
        # x1,x2,x3,x4 → x1,x2,x3,x4,x1,x2,x3,x4
        return torch.cat([encoded, encoded], dim=1)
    
    else:
        # Fallback: pass through or pad
        if encoded.shape[1] == quantum_input_dim:
            return encoded
        elif encoded.shape[1] > quantum_input_dim:
            return encoded[:, :quantum_input_dim]
        else:
            padding = torch.zeros(
                batch_size,
                quantum_input_dim - encoded.shape[1],
                device=encoded.device,
                dtype=encoded.dtype
            )
            return torch.cat([encoded, padding], dim=1)
