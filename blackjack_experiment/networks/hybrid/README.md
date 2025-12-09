# Hybrid Network Module - Modular Architecture

## Structure

The hybrid quantum-classical network is organized into focused, maintainable components:

```
blackjack_experiment/networks/hybrid/
├── __init__.py           # Main exports
├── config.py             # Configuration and defaults (MODIFY HERE!)
├── encoder.py            # Classical feature encoder
├── compression.py        # Encoder compression strategies
├── quantum.py            # Quantum circuit builder
├── controls.py           # Freeze/bypass/dropout controls
├── policy.py             # Main policy network (assembles components)
└── value.py              # Classical value network
```

## Quick Start

```python
from blackjack_experiment.networks.hybrid import (
    UniversalBlackjackHybridPolicyNetwork,
    HybridDefaults
)

# Use defaults
net = UniversalBlackjackHybridPolicyNetwork()

# Or customize
net = UniversalBlackjackHybridPolicyNetwork(
    n_qubits=6,
    n_layers=4,
    encoder_compression='direct'
)
```

## Configuration

**To experiment with architectures**, edit `config.py`:

```python
# blackjack_experiment/networks/hybrid/config.py

class HybridDefaults:
    N_QUBITS = 4                     # Change to 6, 8, etc.
    N_LAYERS = 3                     # More layers = more expressibility
    ENCODER_COMPRESSION = 'minimal'  # 'full', 'minimal', 'direct', 'shared'
    SINGLE_AXIS_ENCODING = True      # Encode only in RY, weights handle RZ
    DATA_REUPLOADING = True          # Re-encode at each layer
```

Then train normally:
```powershell
python -m blackjack_experiment.run hybrid -e 5000
```

All hybrid networks will use the new defaults.

## Component Details

### `config.py` - Central Configuration
- `HybridDefaults`: Global defaults for all networks
- `BypassMode`: Enum for bypass experiments
- **Modify this file** to change architecture for experiments

### `encoder.py` - Classical Feature Encoder
- `build_encoder()`: Creates encoder network
- `compute_encoder_dim()`: Calculates encoder output size based on compression mode
- Pure function design - no state

### `compression.py` - Encoder Compression
- `compress_to_quantum_input()`: Maps encoder output to quantum input dimensions
- Handles: single-axis, full, minimal, direct, shared modes
- Clean separation of compression logic

### `quantum.py` - Quantum Circuit
- `build_quantum_circuit()`: Creates PennyLane QNode
- `apply_entanglement()`: Handles different entanglement strategies
- **Key feature**: Hadamard FIRST, then data encoding
  - Rotates from Z-basis to X-basis before Y rotation
  - Encodes data into X-Z plane for better variational access

### `controls.py` - Control Mechanisms
- `FreezeControls`: Mixin for freezing components
- `BypassControls`: Mixin for bypass experiments
- `QuantumDropout`: Mixin for quantum weight dropout
- Reusable control mechanisms as mixins

### `policy.py` - Main Network
- `UniversalBlackjackHybridPolicyNetwork`: Assembles all components
- Inherits control mixins
- Clean forward pass through modular components

### `value.py` - Classical Value Network
- Simple classical network for critic
- No quantum components needed for A2C value estimation

## Key Architecture Features

### Hadamard-First Encoding
```python
# Quantum circuit order:
qml.Hadamard(wires=i)        # Rotate to X-basis FIRST
qml.RY(angle_y, wires=i)     # Encode data into X-Z plane
qml.RZ(theta_z, wires=i)     # Variational parameters
```

After Hadamard, qubit is in |+⟩ state (superposition on X-axis). RY rotation then encodes data into the X-Z plane, giving variational weights better access to the encoded information.

### Single-Axis Encoding (Default)
```python
# Encoder: 3 → 4 (one value per qubit)
# Compression: [x1,x2,x3,x4,0,0,0,0]  ← zeros for RZ
# Quantum: RY uses data, RZ is purely variational
```

**Benefits**:
- Reduced encoder parameters (16 vs 32)
- Forces quantum weights to learn RZ rotations
- Cleaner separation: data → RY, learned → RZ

### Minimal Postprocessing (Default)
```python
# No hidden layers - direct quantum_output → action_logits
postprocessing = nn.Linear(4, 2)  # 10 params only
```

Prevents postprocessing from learning around a useless quantum circuit. If quantum doesn't contribute, performance will suffer immediately.

## Benefits of Modular Structure

1. **Easier to understand**: Each file has one job
2. **Easier to modify**: Change quantum circuit? Edit `quantum.py` only
3. **Easier to test**: Test compression strategies independently
4. **Easier to extend**: Add new compression mode? Edit `compression.py`
5. **Better collaboration**: Different people can work on different components
6. **Reusable**: Control mixins can be used in other networks

## Example: Adding a New Compression Mode

Edit `compression.py`:
```python
def compress_to_quantum_input(...):
    # existing modes...
    
    elif encoder_compression == 'my_new_mode':
        # Your custom compression logic here
        return custom_compressed_tensor
```

Then set it in `config.py`:
```python
class HybridDefaults:
    ENCODER_COMPRESSION = 'my_new_mode'
```

Done! No need to touch anything else.

