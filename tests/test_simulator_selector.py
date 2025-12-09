"""Test script for simulator checkpoint selector."""

from blackjack_experiment.simulator import find_checkpoints

# Test checkpoint finding
print("Testing checkpoint finder...")
checkpoints = find_checkpoints('results')

print(f"\nFound {len(checkpoints)} checkpoints")
print("\nFirst 10 checkpoints:")
for i, (path, ep, wr, ts) in enumerate(checkpoints[:10], 1):
    print(f"{i:2}. [{ts}] {path:60} ep={ep:5}, wr={wr:.1f}%")

# Test grouping by timestamp
experiments = {}
for rel_path, ep, wr, timestamp in checkpoints:
    if timestamp not in experiments:
        experiments[timestamp] = []
    experiments[timestamp].append((rel_path, ep, wr))

print(f"\n\nFound {len(experiments)} experiments:")
for exp_name in sorted(experiments.keys(), reverse=True):
    checkpoints_in_exp = experiments[exp_name]
    best_wr = max(c[2] for c in checkpoints_in_exp)
    print(f"  {exp_name}: {len(checkpoints_in_exp)} checkpoints, best WR: {best_wr:.1f}%")
