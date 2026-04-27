import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

def cmd_train(args):
    from blackjack_experiment.run import train
    train(
        network_type=args.type,
        episodes=args.episodes,
        output=args.output,
        seed=args.seed,
        dropout=args.dropout,
        frozen_encoder=args.frozen_encoder,
        microwaved_encoder=args.microwave is not None,
        microwaved_fraction=args.microwave if args.microwave is not None else 0.5,
        encoder_lr_scale=args.encoder_lr_scale,
        encoder_layers=args.encoder_layers,
        postprocessing_layers=args.post_layers
    )

def cmd_compare(args):
    from blackjack_experiment.run import compare
    compare(
        episodes=args.episodes,
        output=args.output,
        seed=args.seed,
        dropout=args.dropout,
        frozen_encoder=args.frozen_encoder,
        microwaved_encoder=args.microwave is not None,
        microwaved_fraction=args.microwave if args.microwave is not None else 0.5,
        encoder_lr_scale=args.encoder_lr_scale,
        encoder_layers=args.encoder_layers,
        postprocessing_layers=args.post_layers
    )

def cmd_simulate(args):
    from blackjack_experiment.networks.loader import load_policy_network
    from blackjack_experiment.simulator import BlackjackSimulator
    import pygame
    
    print(f"Loading model: {args.checkpoint}")
    policy_net = load_policy_network(args.checkpoint, strict=False)
    net_type = 'hybrid' if hasattr(policy_net, 'quantum_circuit') else 'classical'
    
    sim = BlackjackSimulator(policy_net, network_type=net_type)
    sim.run()

def cmd_eval(args):
    from blackjack_experiment.run import evaluate
    evaluate(args.checkpoint, args.episodes)

def main():
    parser = argparse.ArgumentParser(description="Blackjack Hybrid Quantum-Classical Exploration")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a single model")
    train_parser.add_argument("--type", choices=["classical", "hybrid"], default="hybrid", help="Network type")
    train_parser.add_argument("-e", "--episodes", type=int, default=5000, help="Number of episodes")
    train_parser.add_argument("-d", "--dropout", type=float, default=0.0, help="Quantum dropout rate")
    train_parser.add_argument("--frozen-encoder", action="store_true", help="Freeze encoder with random weights")
    train_parser.add_argument("--microwave", type=float, help="Fraction of training to keep encoder frozen")
    train_parser.add_argument("--encoder-lr-scale", type=float, default=1.0, help="Scale encoder learning rate")
    train_parser.add_argument("--encoder-layers", type=int, nargs="+", help="Hidden layers for encoder (e.g. 32 16)")
    train_parser.add_argument("--post-layers", type=int, nargs="+", help="Hidden layers for postprocessing (e.g. 8)")
    train_parser.add_argument("-o", "--output", type=str, help="Output directory")
    train_parser.add_argument("-s", "--seed", type=int, help="Random seed")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare classical vs hybrid models")
    compare_parser.add_argument("-e", "--episodes", type=int, default=5000, help="Number of episodes")
    compare_parser.add_argument("-d", "--dropout", type=float, default=0.0, help="Quantum dropout rate for hybrid")
    compare_parser.add_argument("--frozen-encoder", action="store_true", help="Freeze encoder for hybrid")
    compare_parser.add_argument("--microwave", type=float, help="Microwave fraction for hybrid")
    compare_parser.add_argument("--encoder-lr-scale", type=float, default=1.0, help="Scale encoder LR for hybrid")
    compare_parser.add_argument("--encoder-layers", type=int, nargs="+", help="Hidden layers for hybrid encoder")
    compare_parser.add_argument("--post-layers", type=int, nargs="+", help="Hidden layers for hybrid postprocessing")
    compare_parser.add_argument("-o", "--output", type=str, help="Output directory")
    compare_parser.add_argument("-s", "--seed", type=int, help="Random seed")

    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run interactive visualizer")
    sim_parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pth)")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pth)")
    eval_parser.add_argument("-e", "--episodes", type=int, default=500, help="Evaluation episodes")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
