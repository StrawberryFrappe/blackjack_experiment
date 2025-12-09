"""
Configuration for Blackjack A2C experiments.

This module provides dataclasses for all experiment configuration:
- NetworkConfig: Network architecture settings
- AgentConfig: A2C agent hyperparameters
- TrainingConfig: Training loop settings
- Config: Complete experiment configuration
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List


@dataclass
class NetworkConfig:
    """Network configuration - just the type, actual architecture set in network file."""
    network_type: Literal['classical', 'minimal_classical', 'hybrid'] = 'classical'
    
    def is_hybrid(self) -> bool:
        """Check if network type is hybrid."""
        return self.network_type == 'hybrid'


@dataclass
class AgentConfig:
    """A2C agent configuration."""
    learning_rate: float = 0.005
    lr_policy: Optional[float] = None  # Alias for learning_rate
    lr_value: Optional[float] = None   # Separate value LR (defaults to learning_rate)
    encoder_lr_scale: float = 1.0      # Scaling factor for feature encoder LR (hybrid only)
    gamma: float = 0.99
    entropy_coef: float = 0.02
    value_coef: float = 0.25
    max_grad_norm: float = 0.5
    n_steps: int = 5
    use_gae: bool = False
    gae_lambda: float = 0.95
    
    def __post_init__(self):
        if self.lr_policy is not None:
            self.learning_rate = self.lr_policy
        if self.lr_value is None:
            self.lr_value = self.learning_rate


@dataclass
class TrainingConfig:
    """Training configuration."""
    n_episodes: int = 10000
    max_steps_per_episode: int = 200
    print_every: int = 100
    eval_every: int = 500
    eval_episodes: int = 10
    save_frequency: Optional[int] = None  # Auto-calculated if None
    checkpoint_count: int = 180  # Number of evenly-spaced checkpoints for time-lapse (10s @ 18fps)
    
    # Analysis
    enable_analysis: bool = True
    analysis_frequency: Optional[int] = None
    monitor_gradients: bool = True
    
    # Model saving
    save_model: bool = True
    model_path: str = "results/model.pth"
    
    # Microwaved encoder (freeze then unfreeze)
    microwaved_encoder: bool = False
    microwaved_fraction: float = 0.5
    
    def __post_init__(self):
        """Auto-calculate save_frequency if not provided."""
        if self.save_frequency is None and self.checkpoint_count > 0:
            # Calculate interval to produce exactly checkpoint_count checkpoints
            self.save_frequency = max(1, self.n_episodes // self.checkpoint_count)
        elif self.save_frequency is None:
            # Fallback to old default
            self.save_frequency = 500


@dataclass
class Config:
    """Complete experiment configuration."""
    seed: int = 42
    
    # Blackjack environment constants
    env_name: str = 'Blackjack-v1'
    n_actions: int = 2
    
    network: NetworkConfig = field(default_factory=NetworkConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Backward compatibility alias
ExperimentConfig = Config
