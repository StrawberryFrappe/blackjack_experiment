import os
import sys
import torch
import argparse
from pathlib import Path

# Ensure we can import the project package
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from blackjack_experiment.networks.loader import load_policy_network
from blackjack_experiment.networks import UniversalBlackjackHybridPolicyNetwork

def run_demo(checkpoint_path=None):
    print("=" * 60)
    print(" JIMBO THE GAMBLING BOT - Quantum Hybrid Demo")
    print("=" * 60)
    
    if checkpoint_path:
        print(f"Loading Jimbo's brain from: {checkpoint_path}")
        try:
            bot = load_policy_network(checkpoint_path, strict=False)
            print("[OK] Brain loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Could not load brain: {e}")
            print("Falling back to a fresh (random) brain.")
            bot = UniversalBlackjackHybridPolicyNetwork()
    else:
        print("Starting with a fresh (random) brain.")
        bot = UniversalBlackjackHybridPolicyNetwork()
    
    bot.eval()
    
    print("\n[CONFIG] Brain Architecture:")
    print(bot.get_config_summary())
    
    # Standard Blackjack scenario
    # Player has 16 (Hard), Dealer shows 10, No Usable Ace
    # This is a classic 'Hit or Stand' dilemma where basic strategy says Hit, 
    # but many players are afraid to bust.
    hand = (16, 10, False)
    print(f"\n[SCENARIO] Player: {hand[0]} | Dealer: {hand[1]} | Usable Ace: {hand[2]}")
    
    with torch.no_grad():
        # Policy network returns action probabilities [Stand, Hit]
        probs = bot(hand).squeeze()
        
    stand_prob = probs[0].item()
    hit_prob = probs[1].item()
    
    print(f"\n[DECISION] Analysis complete:")
    print(f"   - Probability Stand: {stand_prob:.2%}")
    print(f"   - Probability Hit:   {hit_prob:.2%}")
    
    decision = "HIT" if hit_prob > stand_prob else "STAND"
    confidence = max(stand_prob, hit_prob)
    
    print(f"\nJimbo says: {decision}! (Confidence: {confidence:.1%})")
    
    # Contextual commentary based on common Blackjack strategy
    if decision == "HIT":
        if hand[0] >= 12:
            print("\nNote: Jimbo is taking a calculated risk. Basic strategy suggests hitting on 16 vs 10.")
        else:
            print("\nNote: Jimbo is making a standard move.")
    else: # STAND
        if hand[0] < 12:
            print("\nNote: Jimbo is being extremely cautious (perhaps too much).")
        elif hand[0] >= 17:
            print("\nNote: Jimbo is following standard safety protocols.")
        else:
            print("\nNote: Jimbo is standing on a weak hand. This is the 'uncertainty tax' in action.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to a trained .pth model")
    args = parser.parse_args()
    
    run_demo(args.model)
