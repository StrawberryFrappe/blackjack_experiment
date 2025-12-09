"""Quantum-first training logic."""

class QuantumFirstTrainer:
    """
    Trainer for Quantum-First strategy.
    
    Phase 1: Train only quantum weights (classical frozen)
    Phase 2: Train all weights (fine-tuning)
    """
    
    def __init__(self, quantum_first_episodes: int, normal_episodes: int, quantum_dropout_rate: float = 0.0):
        self.qf_episodes = quantum_first_episodes
        self.normal_episodes = normal_episodes
        self.quantum_dropout_rate = quantum_dropout_rate
        self.total_episodes = quantum_first_episodes + normal_episodes
        self._phase_switched = False

    def configure_network(self, policy_net):
        """Configure network for start of training."""
        # Initial configuration: freeze classical, set dropout
        if hasattr(policy_net, 'freeze_component'):
            policy_net.freeze_component('classical')
            print(f"[QF] Classical components frozen for first {self.qf_episodes} episodes")
            
        if hasattr(policy_net, 'set_quantum_dropout'):
            policy_net.set_quantum_dropout(self.quantum_dropout_rate)
            if self.quantum_dropout_rate > 0:
                print(f"[QF] Quantum dropout enabled: {self.quantum_dropout_rate:.0%}")

    def get_total_episodes(self):
        return self.total_episodes

    def on_episode(self, ep, policy_net):
        """Update network state based on current episode."""
        # Check for phase transition
        if ep >= self.qf_episodes and not self._phase_switched:
            print(f"\n[PHASE] Switching to NORMAL training (Ep {ep})")
            print("  - Unfreezing classical components")
            print("  - Disabling quantum dropout")
            
            if hasattr(policy_net, 'unfreeze_component'):
                policy_net.unfreeze_component('all')
            if hasattr(policy_net, 'set_quantum_dropout'):
                policy_net.set_quantum_dropout(0.0)
            
            self._phase_switched = True
