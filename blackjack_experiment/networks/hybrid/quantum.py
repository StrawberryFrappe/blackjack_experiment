"""Quantum circuit builder for hybrid networks."""

import torch
import pennylane as qml
import numpy as np


def apply_transform(x: torch.Tensor, transform: str, scale: float) -> torch.Tensor:
    """Apply nonlinear transform to map unbounded inputs to bounded rotation angles.
    
    Args:
        x: Input tensor (unbounded from RL environment)
        transform: Transform type ('arctan', 'sin', 'cos', 'tanh', 'identity')
        scale: Output multiplier (e.g., scale=2 maps arctan to [-π, π])
        
    Returns:
        Bounded angle for quantum rotation gate
        
    Design rationale:
    - RL inputs are unbounded (player_sum/31, dealer_card/10, etc.)
    - Rotation gates are periodic (RY(θ) = RY(θ + 2π))
    - Without bounded mapping, similar game states → orthogonal quantum states (chaos)
    - arctan: Smooth monotonic mapping, full range with scale=2
    - tanh: Similar to arctan but saturates faster
    - sin/cos: Periodic - use only if inputs are already bounded
    - identity: No transform (only for debugging)
    """
    if transform == 'arctan':
        return torch.arctan(x) * scale
    elif transform == 'tanh':
        return torch.tanh(x) * scale
    elif transform == 'sin':
        return torch.sin(x * np.pi) * scale
    elif transform == 'cos':
        return torch.cos(x * np.pi) * scale
    elif transform == 'identity':
        return x * scale
    else:
        raise ValueError(f"Unknown transform: {transform}")


def build_quantum_circuit(
    n_qubits: int,
    n_layers: int,
    device_name: str,
    entanglement_strategy: str,
    measurement_mode: str,
    data_reuploading: bool,
    single_axis_encoding: bool,
    encoding_transform: str = 'arctan',
    reuploading_transform: str = 'arctan',
    encoding_scale: float = 2.0
):
    """Build PennyLane quantum circuit with stabilized nonlinear encoding.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        device_name: PennyLane device
        entanglement_strategy: 'linear', 'cyclic', 'full', or 'none'
        measurement_mode: 'pauli_z' or 'amplitude'
        data_reuploading: Whether to re-encode data at each layer
        single_axis_encoding: Whether to encode only in RY (True) or RY+RZ (False)
        encoding_transform: Transform for initial encoding ('arctan', 'sin', 'cos', 'tanh', 'identity')
        reuploading_transform: Transform for re-uploading layers (use same as encoding_transform for stability)
        encoding_scale: Multiplier for transform output (e.g., arctan(x) * encoding_scale)
        
    Returns:
        PennyLane QNode function
        
    Key stabilization improvements:
    1. Hadamard FIRST: Rotates from Z-basis to X-basis before data encoding
    2. Consistent encoding: Same transform for initial and re-uploading phases (prevents chaotic state-space)
    3. Bounded transforms: arctan/tanh map unbounded RL inputs → bounded rotation angles
    4. Full-range scaling: encoding_scale=2 maps arctan output to full [-π, π] period
    5. Single-axis option: Encode data only in RY, let weights handle RZ (reduces parameter interference)
    """
    dev = qml.device(device_name, wires=n_qubits, shots=None)
    
    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(inputs, weights):
        for i in range(n_qubits):
            
            angle_y = apply_transform(inputs[i, 0], encoding_transform, encoding_scale)
            qml.RY(angle_y, wires=i)
            
            if not single_axis_encoding:
                # Dual-axisq: also encode to RZ
                angle_z = apply_transform(inputs[i, 1], encoding_transform, encoding_scale)
                qml.RZ(angle_z, wires=i)
        
        qml.Barrier(wires=range(n_qubits), only_visual=True)

        # Variational layers
        for layer in range(n_layers):
            # Variational rotations 
            for i in range(n_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
            

            # Entanglement
            apply_entanglement(n_qubits, entanglement_strategy, layer)
            
            

            if data_reuploading and layer < n_layers - 1:
                for i in range(n_qubits):
                    # Apply same transform as initial encoding (default: arctan)
                    angle_y = apply_transform(inputs[i, 0], reuploading_transform, encoding_scale)
                    qml.RY(angle_y, wires=i)
                    
                    if not single_axis_encoding:
                        # Dual-axis: also re-upload to RZ with same transform
                        angle_z = apply_transform(inputs[i, 1], reuploading_transform, encoding_scale)
                        qml.RZ(angle_z, wires=i)
                    # Single-axis: only RY re-uploading
                
        
        # Measurement
        if measurement_mode == 'pauli_z':
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        else:
            return qml.probs(wires=range(n_qubits))
    
    return circuit


def apply_entanglement(n_qubits: int, strategy: str, layer: int):
    """Apply entanglement gates based on strategy.
    
    Args:
        n_qubits: Number of qubits
        strategy: 'linear', 'cyclic', 'full', or 'none'
        layer: Current layer index (for layer-dependent patterns)
        
    Entanglement Strategy Trade-offs:
    
    1. 'none' - No entanglement (independent qubits)
       - Pros: Simplest, no barren plateaus, easy gradients
       - Cons: No quantum advantage, equivalent to 4 classical neurons
       - Use when: Debugging, baseline comparison
       
    2. 'linear' - Nearest-neighbor CNOT chain [0→1, 1→2, 2→3]
       - Pros: Minimal gate overhead, stable gradients, good connectivity
       - Cons: No direct long-range entanglement (qubit 0 and 3 need 3 hops)
       - Use when: Default choice for 4-qubit circuits, good expressibility/trainability balance
       - Gradient behavior: Consistent, well-localized
       
    3. 'cyclic' - Circular CNOT ring [0→1, 1→2, 2→3, 3→0]
       - Pros: Symmetric topology, all qubits connected uniformly
       - Cons: One extra gate vs linear, slightly more complex gradient landscape
       - Use when: Symmetry is important, no "edge" qubits desired
       - Gradient behavior: Similar to linear, slightly higher variance
       
    4. 'full' - All-to-all + skip connections [cyclic + even layers: i→i+2]
       - Pros: Maximum expressibility, direct long-range entanglement
       - Cons: More gates = higher noise, barren plateau risk, over-parameterization
       - Use when: Complex tasks require maximum circuit capacity
       - Gradient behavior: Can suffer from gradient vanishing in deep circuits
       
    Recommendation for Blackjack (4 qubits, 3 layers):
    - Start with 'linear': Best balance for small circuits
    - Try 'cyclic' if linear shows edge bias in learned features
    - Avoid 'full' unless linear/cyclic fail to converge
    """
    if strategy == 'none':
        pass
    elif strategy == 'linear':
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    elif strategy == 'cyclic':
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    elif strategy == 'full':
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        if n_qubits >= 4 and layer % 2 == 0:
            for i in range(n_qubits - 2):
                qml.CNOT(wires=[i, i + 2])
    else:
        # Default to linear
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
