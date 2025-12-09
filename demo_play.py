import os
import sys
import argparse
import tkinter as tk
from tkinter import filedialog
import gymnasium as gym
import torch
import time

# Add current directory to path so we can import blackjack_experiment
sys.path.append(os.getcwd())

from blackjack_experiment.networks.loader import load_policy_network
from blackjack_experiment.networks.base import encode_blackjack_state

def select_model_file():
    """Open a file dialog to select a trained model."""
    root = tk.Tk()
    root.withdraw() # Hide the main window
    
    initial_dir = os.path.join(os.getcwd(), "results")
    if not os.path.exists(initial_dir):
        initial_dir = os.getcwd()
        
    file_path = filedialog.askopenfilename(
        title="Select Trained Model",
        filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")],
        initialdir=initial_dir
    )
    return file_path

def play_game(model_path, max_games=10):
    """Run the Blackjack environment with the selected model."""
    print(f"Loading model from: {model_path}")
    try:
        policy_net = load_policy_network(model_path, strict=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Determine encoding
    if hasattr(policy_net, 'encoding'):
        encoding = policy_net.encoding
    else:
        encoding = 'one-hot'
    
    print(f"Model loaded. Using encoding: {encoding}")

    try:
        env = gym.make('Blackjack-v1', render_mode='human')
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Make sure gymnasium is installed: pip install gymnasium[toy_text]")
        return
    
    print(f"Starting game... Playing max {max_games} games. Press Ctrl+C to stop.")
    
    games_played = 0
    try:
        while max_games is None or games_played < max_games:
            games_played += 1
            print(f"\nGame {games_played}/{max_games if max_games else 'Inf'}")
            
            state, info = env.reset()
            terminated = False
            truncated = False
            
            print("\nNew Game")
            # Initial render
            env.render()
            time.sleep(1.0)
            
            while not (terminated or truncated):
                # Encode state
                # state is (player_sum, dealer_card, usable_ace)
                state_tensor = encode_blackjack_state(state, encoding=encoding)
                
                # Get action from model
                with torch.no_grad():
                    output = policy_net(state_tensor)
                    # Output is likely logits or probabilities. Argmax works for both.
                    action = torch.argmax(output).item()
                
                action_str = "HIT" if action == 1 else "STICK"
                print(f"State: {state}, Action: {action_str}")
                
                # Step environment
                state, reward, terminated, truncated, info = env.step(action)
                
                # Small delay for visibility
                time.sleep(1.0) 
                
            result = "WON" if reward > 0 else "LOST" if reward < 0 else "DRAW"
            print(f"Game Over. Result: {result} (Reward: {reward})")
            time.sleep(2.0) # Pause between games
            
        print(f"\nFinished playing {games_played} games.")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Blackjack demo with trained model")
    parser.add_argument("--games", type=int, default=10, help="Maximum number of games to play (default: 10)")
    args = parser.parse_args()

    print("Please select a model file from the dialog...")
    model_path = select_model_file()
    if model_path:
        play_game(model_path, max_games=args.games)
    else:
        print("No file selected.")
