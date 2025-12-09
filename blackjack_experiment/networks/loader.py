"""
Network loading utilities.

This module provides functions to load networks from checkpoints,
extracting network configuration from the saved file to ensure
the recreated network matches the original architecture.

IMPORTANT: When loading from a checkpoint, the network configuration
is read from the checkpoint file. If the config is missing or incompatible,
the load will fail rather than silently using incorrect defaults.
"""

import torch
import re
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List


class NetworkLoadError(Exception):
    """Raised when network loading fails due to configuration issues."""
    pass


def _infer_postprocessing_from_state_dict(state_dict: Dict) -> List[int]:
    """
    Infer postprocessing layer sizes from a state dict.
    
    For old checkpoints that don't have postprocessing_layers saved,
    we can infer it from the weight shapes in the state dict.
    
    Args:
        state_dict: The model state dict
        
    Returns:
        List of hidden layer sizes (empty = direct output layer)
    """
    # Find all postprocessing layer weights
    postproc_weights = {}
    for key, value in state_dict.items():
        if key.startswith('postprocessing.') and key.endswith('.weight'):
            # Extract layer index: 'postprocessing.0.weight' -> 0
            match = re.match(r'postprocessing\.(\d+)\.weight', key)
            if match:
                idx = int(match.group(1))
                postproc_weights[idx] = value.shape
    
    if not postproc_weights:
        return []
    
    # Sort by index and extract hidden layer sizes
    # Each weight has shape (out_features, in_features)
    # Hidden layers are all but the last linear layer
    sorted_indices = sorted(postproc_weights.keys())
    hidden_sizes = []
    
    for idx in sorted_indices[:-1]:  # All but the last
        out_features = postproc_weights[idx][0]
        hidden_sizes.append(out_features)
    
    return hidden_sizes


def extract_network_config(checkpoint_path: str) -> Dict[str, Any]:
    """
    Extract network configuration from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary with network configuration
        
    Raises:
        NetworkLoadError: If checkpoint doesn't contain network config
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'network_config' not in checkpoint:
        raise NetworkLoadError(
            f"Checkpoint '{checkpoint_path}' does not contain network configuration.\n"
            "This checkpoint was saved with an older version that didn't store network specs.\n"
            "Cannot reliably recreate the network architecture."
        )
    
    return checkpoint['network_config']


def load_policy_network(checkpoint_path: str, strict: bool = True):
    """
    Load a policy network from checkpoint with correct architecture.
    
    This function extracts network configuration from the checkpoint
    and creates a network with matching architecture before loading weights.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        strict: If True, fail when network config is missing.
                If False, attempt to infer from state_dict shapes or session_info.
        
    Returns:
        Loaded policy network in eval mode
        
    Raises:
        NetworkLoadError: If network config is missing and strict=True,
                         or if architecture cannot be determined
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Try to get network config from checkpoint
    network_config = checkpoint.get('network_config')
    
    if network_config is None:
        if strict:
            raise NetworkLoadError(
                f"Checkpoint '{checkpoint_path}' does not contain network configuration.\n"
                "Use strict=False to attempt inference from state_dict (recommended)."
            )
        else:
            # Try to infer from state_dict (most reliable)
            state_dict = checkpoint.get('policy_state_dict', checkpoint.get('policy_net', checkpoint))
            if isinstance(state_dict, dict):
                network_config = _infer_config_from_state_dict(state_dict)
            
            # Fallback: try session_info.json
            if network_config is None:
                network_config = _infer_config_from_session_info(checkpoint_path)
            
            # Last resort: infer from path
            if network_config is None:
                print(f"[WARN] Could not infer config from state_dict or session_info. Using path-based inference...")
                network_config = _infer_config_from_path(checkpoint_path)
    
    # Create network based on config
    network_type = network_config.get('type', 'classical')
    
    if network_type == 'hybrid':
        from .hybrid import UniversalBlackjackHybridPolicyNetwork
        
        # Extract all hybrid-specific parameters
        n_qubits = network_config.get('n_qubits')
        n_layers = network_config.get('n_layers')
        
        if n_qubits is None or n_layers is None:
            raise NetworkLoadError(
                f"Hybrid network config incomplete: n_qubits={n_qubits}, n_layers={n_layers}\n"
                "Cannot recreate network without these parameters."
            )
        
        # Infer postprocessing architecture from state_dict if not saved
        postproc_layers = network_config.get('postprocessing_layers')
        if postproc_layers is None:
            # Infer from state dict keys
            state_dict = checkpoint.get('policy_state_dict', checkpoint.get('policy_net', checkpoint))
            postproc_layers = _infer_postprocessing_from_state_dict(state_dict)
        
        policy_net = UniversalBlackjackHybridPolicyNetwork(
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoding=network_config.get('encoding', 'compact'),
            entanglement_strategy=network_config.get('entanglement_strategy', 'cyclic'),
            measurement_mode=network_config.get('measurement_mode', 'pauli_z'),
            data_reuploading=network_config.get('data_reuploading', True),
            learnable_input_scaling=network_config.get('learnable_input_scaling', False),
            postprocessing_layers=postproc_layers,
            encoder_compression=network_config.get('encoder_compression', 'full')
        )
        
    elif network_type == 'classical':
        from .classical import BlackjackClassicalPolicyNetwork
        
        hidden_sizes = network_config.get('hidden_sizes')
        activation = network_config.get('activation', 'tanh')
        
        if hidden_sizes is not None:
            policy_net = BlackjackClassicalPolicyNetwork(
                hidden_sizes=hidden_sizes,
                activation=activation
            )
        else:
            # Use defaults for classical
            policy_net = BlackjackClassicalPolicyNetwork()
            
    elif network_type == 'minimal':
        from .classical import BlackjackMinimalClassicalPolicyNetwork
        policy_net = BlackjackMinimalClassicalPolicyNetwork()
        
    else:
        raise NetworkLoadError(f"Unknown network type: {network_type}")
    
    # Load weights
    if 'policy_state_dict' in checkpoint:
        policy_net.load_state_dict(checkpoint['policy_state_dict'])
    elif 'policy_net' in checkpoint:
        policy_net.load_state_dict(checkpoint['policy_net'])
    else:
        # Assume checkpoint is raw state dict
        policy_net.load_state_dict(checkpoint)
    
    policy_net.eval()
    return policy_net


def load_networks_from_checkpoint(checkpoint_path: str, strict: bool = True) -> Tuple:
    """
    Load both policy and value networks from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        strict: If True, fail when config is missing
        
    Returns:
        Tuple of (policy_net, value_net)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    network_config = checkpoint.get('network_config')
    
    if network_config is None and strict:
        raise NetworkLoadError(
            f"Checkpoint '{checkpoint_path}' does not contain network configuration."
        )
    
    # Load policy network
    policy_net = load_policy_network(checkpoint_path, strict=strict)
    
    # Load value network (always classical for now)
    from .classical import BlackjackClassicalValueNetwork
    value_net = BlackjackClassicalValueNetwork()
    
    if 'value_state_dict' in checkpoint:
        value_net.load_state_dict(checkpoint['value_state_dict'])
    
    value_net.eval()
    return policy_net, value_net


def _infer_config_from_state_dict(state_dict: Dict) -> Optional[Dict[str, Any]]:
    """
    Infer network configuration from state_dict shapes.
    
    This is more reliable than path-based inference as it uses
    the actual weight shapes to determine network architecture.
    
    Returns:
        Config dict or None if inference fails
    """
    # Check for quantum weights to determine if hybrid
    if 'weights' in state_dict:
        weights_shape = state_dict['weights'].shape
        # weights shape is (n_layers, n_qubits, 2)
        n_layers = weights_shape[0]
        n_qubits = weights_shape[1]
        
        # Infer postprocessing layers
        postproc_layers = _infer_postprocessing_from_state_dict(state_dict)
        
        # Check for feature encoder (if missing, it's amplitude/direct encoding)
        has_encoder = any('feature_encoder' in k or 'encoder' in k for k in state_dict.keys())
        
        # Check for input scaling (learnable_input_scaling)
        has_input_scaling = 'input_scale' in state_dict
        
        # Infer measurement mode from postprocessing input size
        if postproc_layers and 'postprocessing.0.weight' in state_dict:
            postproc_input_size = state_dict['postprocessing.0.weight'].shape[1]
            # If postproc input is 2^n_qubits, it's amplitude encoding
            if postproc_input_size == 2 ** n_qubits:
                measurement_mode = 'amplitude'
            else:
                measurement_mode = 'pauli_z'
        else:
            measurement_mode = 'pauli_z'
        
        # Infer encoder_compression based on whether encoder exists
        if not has_encoder:
            encoder_compression = 'none'  # No encoder at all (amplitude encoding)
        else:
            encoder_compression = 'full'  # Has encoder (standard)
        
        print(f"[INFO] Inferred hybrid config from state_dict:")
        print(f"       n_qubits={n_qubits}, n_layers={n_layers}")
        print(f"       measurement_mode={measurement_mode}, encoder_compression={encoder_compression}")
        
        return {
            'type': 'hybrid',
            'n_qubits': n_qubits,
            'n_layers': n_layers,
            'postprocessing_layers': postproc_layers,
            'learnable_input_scaling': has_input_scaling,
            'measurement_mode': measurement_mode,
            'encoder_compression': encoder_compression,
        }
    
    # Check for classical network patterns (network.0.weight, etc.)
    if 'network.0.weight' in state_dict:
        # Infer hidden sizes from weight shapes
        hidden_sizes = []
        idx = 0
        while f'network.{idx}.weight' in state_dict:
            weight = state_dict[f'network.{idx}.weight']
            # Each hidden layer has shape (out_features, in_features)
            # Skip the last linear layer (output layer)
            next_idx = idx + 2  # +2 because of activation layers
            if f'network.{next_idx}.weight' in state_dict:
                hidden_sizes.append(weight.shape[0])
            idx += 2  # Linear + Activation
        
        print(f"[INFO] Inferred classical config from state_dict: hidden_sizes={hidden_sizes}")
        return {
            'type': 'classical',
            'hidden_sizes': hidden_sizes if hidden_sizes else None,
        }
    
    return None


def _infer_config_from_session_info(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Try to load network config from session_info.json in the same directory.
    
    Returns:
        Config dict or None if not found
    """
    import json
    checkpoint_dir = Path(checkpoint_path).parent
    session_info_path = checkpoint_dir / 'session_info.json'
    
    if not session_info_path.exists():
        return None
    
    try:
        with open(session_info_path) as f:
            session_info = json.load(f)
        
        # Try to extract from training_metadata.network_info
        network_info = session_info.get('training_metadata', {}).get('network_info', {})
        
        if network_info.get('type') == 'hybrid':
            print(f"[INFO] Loaded config from session_info.json: n_qubits={network_info.get('n_qubits')}, n_layers={network_info.get('n_layers')}")
            return {
                'type': 'hybrid',
                'n_qubits': network_info.get('n_qubits'),
                'n_layers': network_info.get('n_layers'),
            }
        elif network_info.get('type') == 'classical':
            return {'type': 'classical'}
        
    except Exception as e:
        print(f"[WARN] Failed to read session_info.json: {e}")
    
    return None


def _infer_config_from_path(checkpoint_path: str) -> Dict[str, Any]:
    """
    Attempt to infer network type from checkpoint path.
    
    WARNING: This is a fallback and may not be reliable.
    """
    path_lower = str(checkpoint_path).lower()
    
    if 'hybrid' in path_lower:
        print("[WARN] Inferred 'hybrid' type from path. Using class defaults - may not match original!")
        from .hybrid import UniversalBlackjackHybridPolicyNetwork
        # Get class defaults
        temp = UniversalBlackjackHybridPolicyNetwork.__init__.__defaults__
        return {
            'type': 'hybrid',
            'n_qubits': 4,  # Class default
            'n_layers': 3,  # Class default
        }
    elif 'minimal' in path_lower:
        return {'type': 'minimal'}
    else:
        return {'type': 'classical'}


def get_network_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get network information from checkpoint without loading weights.
    
    Useful for inspecting checkpoints before loading.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with network info, or error info if config missing
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        info = {
            'path': str(checkpoint_path),
            'has_network_config': 'network_config' in checkpoint,
            'has_policy_weights': 'policy_state_dict' in checkpoint or 'policy_net' in checkpoint,
            'has_value_weights': 'value_state_dict' in checkpoint,
        }
        
        if 'network_config' in checkpoint:
            info['network_config'] = checkpoint['network_config']
        
        if 'episode_count' in checkpoint:
            info['episode_count'] = checkpoint['episode_count']
            
        return info
        
    except Exception as e:
        return {'path': str(checkpoint_path), 'error': str(e)}
