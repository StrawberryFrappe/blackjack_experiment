"""Test checkpoint loading after cleanup."""

from blackjack_experiment.networks.loader import load_policy_network
import os

# Test checkpoint loading
test_checkpoint = 'results/hybrid_20251205_222306/training/final.pth'

if os.path.exists(test_checkpoint):
    print(f"Testing checkpoint: {test_checkpoint}")
    net = load_policy_network(test_checkpoint, strict=False)
    print(f"✓ Checkpoint loaded successfully")
    print(f"✓ Network type: {type(net).__name__}")
    print(f"✓ Total params: {sum(p.numel() for p in net.parameters())}")
else:
    print("✗ Checkpoint not found for testing")
