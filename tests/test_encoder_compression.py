"""Test encoder compression modes."""

import torch
from blackjack_experiment.networks.hybrid import UniversalBlackjackHybridPolicyNetwork

def test_compression_mode(mode: str):
    """Test a specific compression mode."""
    print(f"\n{'='*70}")
    print(f"Testing encoder_compression='{mode}'")
    print('='*70)
    
    net = UniversalBlackjackHybridPolicyNetwork(
        n_qubits=4,
        n_layers=2,
        encoder_compression=mode
    )
    
    print(net.get_config_summary())
    
    # Test forward pass
    state = (18, 10, False)  # player_sum, dealer_card, usable_ace
    with torch.no_grad():
        # Get intermediates to see encoder output
        outputs = net.forward_with_intermediates(state)
        
        print(f"\nIntermediate Outputs:")
        print(f"  Input shape: {outputs['input'].shape}")
        print(f"  Feature encoder output shape: {outputs['feature_encoder_output'].shape}")
        print(f"  Feature encoder output: {outputs['feature_encoder_output'].squeeze()}")
        print(f"  Compressed encoder output shape: {outputs['compressed_encoder_output'].shape}")
        print(f"  Compressed encoder output: {outputs['compressed_encoder_output'].squeeze()}")
        print(f"  Quantum output shape: {outputs['quantum_output'].shape}")
        print(f"  Action probs: {outputs['action_probs'].squeeze()}")
    
    return net

if __name__ == '__main__':
    modes = ['full', 'minimal', 'direct', 'shared']
    
    print("Testing all encoder compression modes...")
    
    for mode in modes:
        net = test_compression_mode(mode)
        
        # Show parameter count difference
        params = net.get_parameter_breakdown()
        print(f"\nParameter count for {mode}: {params['total']} total")
        print(f"  Encoder: {params['feature_encoder']} params")
    
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print("'full':    3→8 encoder (32 params) - default, most freedom")
    print("'minimal': 3→4 encoder (16 params) - duplicate values for RY/RZ pairs")
    print("'direct':  3→3 encoder (12 params) - pad with zeros, minimal processing")
    print("'shared':  3→4 encoder (16 params) - same values for all RY/RZ")
    print("\n✓ All tests passed! Choose compression mode with --encoder-compression flag")
