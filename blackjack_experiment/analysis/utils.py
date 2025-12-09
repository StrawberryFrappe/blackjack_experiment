"""
Shared utilities for Blackjack analysis.

This module provides common functions used across multiple analysis modules,
avoiding code duplication and ensuring consistency.
"""

import numpy as np
from typing import List, Tuple


def generate_all_blackjack_states() -> List[Tuple[int, int, int]]:
    """
    Generate all valid Blackjack states.
    
    Returns:
        List of (player_sum, dealer_card, usable_ace) tuples.
        Total: 18 player_sums (4-21) × 10 dealer_cards (1-10) × 2 ace states = 360 states
    """
    states = []
    for player_sum in range(4, 22):  # 4-21
        for dealer_card in range(1, 11):  # 1-10 (Ace=1)
            for usable_ace in [0, 1]:
                states.append((player_sum, dealer_card, usable_ace))
    return states


def create_blackjack_test_states(n_states: int = 1000) -> List[Tuple[int, int, int]]:
    """
    Create a list of representative blackjack test states.
    
    Args:
        n_states: Number of test states to generate
        
    Returns:
        List of (player_sum, dealer_card, usable_ace) tuples
    """
    test_states = []
    
    # Generate random valid blackjack states
    for _ in range(n_states):
        player_sum = np.random.randint(4, 22)  # Player can have 4-21
        dealer_card = np.random.randint(1, 11)  # Dealer shows 1(Ace)-10
        usable_ace = np.random.randint(0, 2)  # 0 or 1
        
        # Ensure valid state (usable ace only makes sense if sum >= 12)
        if usable_ace and player_sum < 12:
            usable_ace = 0
            
        test_states.append((player_sum, dealer_card, usable_ace))
    
    return test_states


def get_basic_strategy_action(player_sum: int, dealer_card: int, usable_ace: int) -> int:
    """
    Get the optimal basic strategy action for a given state.
    
    Basic strategy rules (simplified):
    - Always hit on 11 or below
    - Always stand on 17+ (hard) or 19+ (soft)
    - Against weak dealer (2-6): stand on 12-16
    - Against strong dealer (7-A): hit on 12-16
    - With usable ace: more aggressive hitting
    
    Args:
        player_sum: Player's hand total
        dealer_card: Dealer's visible card (1=Ace)
        usable_ace: Whether player has a usable ace (0 or 1)
    
    Returns:
        0 = Stand, 1 = Hit
    """
    # Always hit on 11 or below
    if player_sum <= 11:
        return 1
    
    # Always stand on 20-21
    if player_sum >= 20:
        return 0
    
    # With usable ace (soft hand)
    if usable_ace:
        # Soft 19+ (A,8 or A,9): Stand
        if player_sum >= 19:
            return 0
        # Soft 18 (A,7): Stand against 2-8, Hit against 9-A
        if player_sum == 18:
            return 0 if dealer_card <= 8 else 1
        # Soft 17 or below: Hit
        return 1
    
    # Hard hands
    # 17-19: Always stand
    if player_sum >= 17:
        return 0
    
    # 12-16: Depends on dealer
    if player_sum >= 12:
        # Weak dealer (2-6): Stand (let dealer bust)
        if 2 <= dealer_card <= 6:
            # Special case: hit 12 vs 2,3
            if player_sum == 12 and dealer_card in [2, 3]:
                return 1
            return 0
        # Strong dealer (7-A): Hit
        return 1
    
    # Should not reach here
    return 1
