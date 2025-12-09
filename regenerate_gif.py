#!/usr/bin/env python
"""
Regenerate behavior GIF for an existing experiment.
Usage: python regenerate_gif.py <results_dir> [max_checkpoints]
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np

# Ensure parent directory is in path
_parent = Path(__file__).resolve().parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from blackjack_experiment.analysis.learning import LearningAnalyzer
from blackjack_experiment.networks.classical import BlackjackClassicalPolicyNetwork
from blackjack_experiment.networks.hybrid import UniversalBlackjackHybridPolicyNetwork

def get_model_factory(checkpoint_path):
    """
    Inspect a checkpoint and return a factory function that creates
    a matching network architecture.
    """
    print(f"Inspecting checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
        
    network_config = checkpoint.get('network_config', {})
    
    # Fallback for older checkpoints
    if not network_config:
        print("Warning: No network_config found in checkpoint. Inferring from path/content...")
        if 'hybrid' in str(checkpoint_path).lower() or 'hybrid' in str(checkpoint_path.parent).lower():
            network_config = {'type': 'hybrid'}
        else:
            network_config = {'type': 'classical'}
            
    network_type = network_config.get('type', 'classical')
    
    if network_type == 'hybrid':
        # Default values if not in config
        n_qubits = network_config.get('n_qubits', 4)
        n_layers = network_config.get('n_layers', 3)
        encoder_compression = network_config.get('encoder_compression', 'full')
        single_axis = network_config.get('single_axis_encoding', False)
        encoding_transform = network_config.get('encoding_transform', 'arctan')
        encoding_scale = network_config.get('encoding_scale', 1.0)
        postproc_layers = network_config.get('postprocessing_layers', [])
        
        print(f"Detected Hybrid Network: {n_qubits} qubits, {n_layers} layers")
        
        def factory():
            return UniversalBlackjackHybridPolicyNetwork(
                n_qubits=n_qubits,
                n_layers=n_layers,
                encoder_compression=encoder_compression,
                single_axis_encoding=single_axis,
                encoding_transform=encoding_transform,
                encoding_scale=encoding_scale,
                postprocessing_layers=postproc_layers
            )
        return factory
        
    else:
        print("Detected Classical Network")
        def factory():
            return BlackjackClassicalPolicyNetwork()
        return factory

def main():
    parser = argparse.ArgumentParser(description='Regenerate behavior GIF')
    parser.add_argument('results_dir', type=str, help='Path to results directory')
    parser.add_argument('--max_checkpoints', type=int, default=180, help='Target number of frames (default: 180 for 10s @ 18fps)')
    parser.add_argument('--duration', type=float, default=10.0, help='Target duration in seconds')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
        
    # Find checkpoints
    checkpoints = sorted(list(results_dir.glob("checkpoint_*.pth")))
    if not checkpoints:
        # Try checkpoints subdir
        checkpoints_dir = results_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = sorted(list(checkpoints_dir.glob("checkpoint_*.pth")))
            
    if not checkpoints:
        # Try finding any .pth files that look like checkpoints
        checkpoints = sorted(list(results_dir.glob("*.pth")))
        # Filter out final model if checkpoints exist
        checkpoints = [c for c in checkpoints if "final" not in c.name]
        
    if not checkpoints:
        print("No checkpoints found!")
        sys.exit(1)
        
    print(f"Found {len(checkpoints)} checkpoints.")
    
    # Get factory from first checkpoint
    factory = get_model_factory(checkpoints[0])
    
    # Initialize analyzer
    analyzer = LearningAnalyzer(
        checkpoint_dir=str(results_dir),
        model_factory_fn=factory
    )
    
    # Calculate frame duration
    # If we have more checkpoints than max, we subsample
    num_frames = min(len(checkpoints), args.max_checkpoints)
    frame_duration = args.duration / max(1, num_frames)
    fps = 1 / frame_duration
    
    print(f"Generating GIF: {num_frames} frames, {args.duration}s duration ({fps:.1f} fps)")
    
    analyzer.generate_checkpoint_behavior_heatmaps(
        output_dir=str(results_dir),
        max_checkpoints=args.max_checkpoints,
        create_gif=True,
        gif_duration=frame_duration
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
