"""Configuration and defaults for hybrid networks."""

from enum import Enum


class BypassMode(Enum):
    """Bypass modes for quantum circuit during forward pass."""
    NONE = 'none'           # Normal quantum processing
    ZEROS = 'zeros'         # Replace quantum output with zeros
    ENCODER = 'encoder'     # Pass encoder output directly (truncated/padded)
    NOISE = 'noise'         # Replace with random noise in [-1, 1]
    SCRAMBLER = 'scrambler' # Freeze encoder with random weights (data scrambler)


class HybridDefaults:
    """Default configuration values. Modify here to change defaults globally.
    
    IMPORTANT: To experiment with different architectures, modify values HERE,
    not in CLI or config files. This keeps network design decisions centralized.
    """
    
    # Quantum circuit
    N_QUBITS = 5
    N_LAYERS = 4
    ENTANGLEMENT = 'linear'       # 'linear', 'cyclic', 'full', 'none'
    MEASUREMENT = 'pauli_z'       # 'pauli_z', 'amplitude'
    DATA_REUPLOADING = True
    
    # Classical components  
    ENCODING = 'compact'          # 'compact' (3 features) or 'one-hot' (45)
    LEARNABLE_INPUT_SCALING = False  # Disabled to prevent linear regime collapse
    ENCODER_COMPRESSION = 'full'  # 'full', 'minimal', 'direct', 'shared' - CHANGE HERE for experiments
    
    # Encoder architecture (None = single layer, auto-scales with encoding)
    # one-hot (45 inputs) gets [32, 16] hidden layers for feature extraction
    # compact (3 inputs) gets None (direct mapping) to avoid over-parameterization
    ENCODER_LAYERS_ONE_HOT = None  # Hidden layers for one-hot encoding
    ENCODER_LAYERS_COMPACT = None      # Hidden layers for compact encoding (direct mapping)
    
    # Quantum encoding strategy
    SINGLE_AXIS_ENCODING = False  # Dual-axis encoding: encode data in both RY and RZ
    
    # Encoding transforms
    # Controls how RL inputs (unbounded) are mapped to rotation angles (bounded)
    ENCODING_TRANSFORM = 'arctan'  # 'arctan', 'sin', 'cos', 'tanh', 'identity'
    REUPLOADING_TRANSFORM = 'arctan'  # Transform for data re-uploading layers
    ENCODING_SCALE = 2  # arctan(x) * ENCODING_SCALE.
    
    DEVICE = 'default.qubit'
