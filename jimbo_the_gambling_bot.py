import os
import sys
import torch
from pathlib import Path

# Ensure we can import the project package
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from blackjack_experiment.networks import UniversalBlackjackHybridPolicyNetwork

def run_demo():
    print("=" * 60)
    print(" JIMBO THE GAMBLING BOT - Quantum Hybrid Demo")
    print("=" * 60)
    
    # Initialize with default architecture
    bot = UniversalBlackjackHybridPolicyNetwork()
    bot.eval()
    
    print(bot.get_config_summary())
    
    # Standard Blackjack scenario
    # Player has 16 (Hard), Dealer shows 10, No Usable Ace
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
    
    if decision == "HIT" and hand[0] >= 12:
        print("\nNote: Jimbo is taking a risk. This is the 'uncertainty tax' in action.")
    elif decision == "STAND" and hand[0] < 12:
        print("\nNote: Jimbo is being very cautious.")

if __name__ == "__main__":
    run_demo()
