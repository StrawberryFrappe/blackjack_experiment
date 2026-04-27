"""Unified configuration for Blackjack Hybrid Quantum-Classical experiments.

Edit this file to change any aspect of the architecture or training.
All components import from here - no other config files exist.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class HybridConfig:
    """Variational Quantum Circuit (VQC) architecture.

    Tweak these to change the hybrid network topology.
    """
    # --- Quantum circuit ---
    n_qubits: int = 5
    n_layers: int = 4
    entanglement: str = 'linear'        # 'linear' | 'cyclic' | 'full' | 'none'
    measurement: str = 'pauli_z'        # 'pauli_z' | 'amplitude'
    data_reuploading: bool = True
    device: str = 'default.qubit'

    # --- Classical integration ---
    encoding: str = 'compact'           # 'compact' (3 raw features) | 'one-hot' (45)
    learnable_input_scaling: bool = False
    encoder_compression: str = 'full'   # 'full' | 'minimal' | 'direct' | 'shared'
    encoder_layers: Optional[List[int]] = None      # e.g. [32, 16] or None
    postprocessing_layers: Optional[List[int]] = None  # e.g. [8] or None

    # --- Encoding transform ---
    single_axis_encoding: bool = False
    encoding_transform: str = 'arctan'
    reuploading_transform: str = 'arctan'
    encoding_scale: float = 2.0


@dataclass
class ClassicalConfig:
    """Classical neural network architecture."""
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = 'tanh'


@dataclass
class AgentConfig:
    """A2C reinforcement learning hyperparameters."""
    learning_rate: float = 0.005
    gamma: float = 0.99
    entropy_coef: float = 0.02
    value_coef: float = 0.25
    max_grad_norm: float = 0.5
    n_steps: int = 5
    use_gae: bool = False
    gae_lambda: float = 0.95
    encoder_lr_scale: float = 1.0           # LR scale for hybrid encoder
    use_encoding_diversity: bool = False
    encoding_diversity_coef: float = 0.001


@dataclass
class TrainingConfig:
    """Training loop and experiment management settings."""
    n_episodes: int = 10000
    max_steps_per_episode: int = 200
    print_every: int = 100
    eval_every: int = 500
    eval_episodes: int = 10
    checkpoint_count: int = 180     # target number of saved checkpoints

    # Analysis
    enable_analysis: bool = True
    monitor_gradients: bool = True

    # Saving
    save_model: bool = True
    save_frequency: Optional[int] = None   # computed in __post_init__ if None

    # Microwaved encoder - freeze encoder for first `microwaved_fraction` of training
    microwaved_encoder: bool = False
    microwaved_fraction: float = 0.5

    def __post_init__(self):
        if self.save_frequency is None:
            if self.checkpoint_count > 0:
                self.save_frequency = max(1, self.n_episodes // self.checkpoint_count)
            else:
                self.save_frequency = 500


@dataclass
class Config:
    """Global experiment configuration - single entry point."""
    seed: int = 42
    network_type: Literal['classical', 'hybrid'] = 'hybrid'

    hybrid: HybridConfig = field(default_factory=HybridConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Environment
    env_name: str = 'Blackjack-v1'
    n_actions: int = 2

    def is_hybrid(self) -> bool:
        return self.network_type == 'hybrid'
