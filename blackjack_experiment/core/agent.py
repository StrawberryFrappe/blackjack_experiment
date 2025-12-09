"""
Advantage Actor-Critic (A2C) agent for Blackjack.

This module provides the A2CAgent class that implements:
- Policy network (actor) for action selection
- Value network (critic) for state value estimation
- Advantage estimation for variance reduction
- Entropy regularization for exploration
- Optional encoding diversity regularization for hybrid networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional, Dict, Any


class A2CAgent:
    """
    Advantage Actor-Critic (A2C) agent.
    
    A2C is an actor-critic method that uses:
    - A policy network (actor) to select actions
    - A value network (critic) to estimate state values
    - Advantage estimation to reduce variance
    - Entropy regularization for exploration
    """
    
    def __init__(
        self,
        policy_net: nn.Module,
        value_net: nn.Module,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 5,
        use_gae: bool = False,
        gae_lambda: float = 0.95,
        use_encoding_diversity: bool = False,
        encoding_diversity_coef: float = 0.001,
        seed: Optional[int] = None,
        encoder_lr_scale: float = 1.0
    ):
        """
        Initialize A2C agent.
        
        Args:
            policy_net: Policy network (actor)
            value_net: Value network (critic)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            entropy_coef: Coefficient for entropy bonus
            value_coef: Coefficient for value loss
            max_grad_norm: Maximum gradient norm for clipping
            n_steps: Number of steps for N-step TD
            use_gae: Whether to use Generalized Advantage Estimation
            gae_lambda: Lambda parameter for GAE
            use_encoding_diversity: Whether to use encoding diversity regularization
            encoding_diversity_coef: Coefficient for encoding diversity loss
            seed: Random seed
            encoder_lr_scale: Scaling factor for feature encoder learning rate (hybrid only)
        """
        self.policy_net = policy_net
        self.value_net = value_net
        
        # Check if policy network is hybrid
        self.is_hybrid_network = hasattr(policy_net, 'feature_encoder')
        
        # Separate optimizers for actor and critic
        if self.is_hybrid_network and encoder_lr_scale != 1.0:
            encoder_lr = learning_rate * encoder_lr_scale
            # Separate encoder parameters from the rest
            encoder_params = list(policy_net.feature_encoder.parameters())
            encoder_param_ids = set(id(p) for p in encoder_params)
            other_params = [p for p in policy_net.parameters() if id(p) not in encoder_param_ids]
            
            self.policy_optimizer = optim.Adam([
                {'params': other_params, 'lr': learning_rate},
                {'params': encoder_params, 'lr': encoder_lr}
            ])
        else:
            self.policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
            
        self.value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        
        # Encoding diversity regularization (for hybrid networks ONLY)
        if self.is_hybrid_network:
            self.use_encoding_diversity = use_encoding_diversity
            self.encoding_diversity_coef = encoding_diversity_coef
        else:
            self.use_encoding_diversity = False
            self.encoding_diversity_coef = 0.0
        
        # Store training configuration for reproducibility
        self.seed = seed
        self.training_config = {
            'learning_rate': learning_rate,
            'encoder_lr_scale': encoder_lr_scale,
            'gamma': gamma,
            'entropy_coef': entropy_coef,
            'value_coef': value_coef,
            'max_grad_norm': max_grad_norm,
            'n_steps': n_steps,
            'use_gae': use_gae,
            'gae_lambda': gae_lambda,
        }
        
        # Storage for N-step transitions
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.entropies = []
        self.dones = []
        
        # Storage for encoding diversity stats
        self.encoding_variances = []
        self.pairwise_distances = []
        
        # Metrics
        self.episode_count = 0
        self.step_count = 0
        self.update_count = 0
        
        # Running statistics
        self.reward_history = deque(maxlen=100)
        self.episode_metrics = []
        
        # Gradient monitoring (optional)
        self.gradient_monitor = None
    
    def set_gradient_monitor(self, monitor):
        """Set gradient monitor for tracking gradients during training."""
        self.gradient_monitor = monitor
    
    def select_action(self, state, training: bool = True) -> int:
        """
        Select action using policy network.
        
        Args:
            state: Current state (tuple for Blackjack)
            training: Whether we're in training mode
            
        Returns:
            Selected action (0=stick, 1=hit)
        """
        # Handle tuple states (Blackjack format)
        if isinstance(state, tuple):
            state_input = state
        else:
            state_input = torch.tensor([state], dtype=torch.long)
        
        # Get action probabilities
        if self.is_hybrid_network and self.use_encoding_diversity and training:
            action_probs, encoding_stats = self.policy_net(state_input, return_encoding_stats=True)
            self.encoding_variances.append(encoding_stats['encoding_variance'])
            self.pairwise_distances.append(encoding_stats['pairwise_distance'])
        else:
            action_probs = self.policy_net(state_input)
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        if training:
            value = self.value_net(state_input)
            
            self.states.append(state)
            self.actions.append(action.item())
            self.values.append(value.squeeze())
            self.log_probs.append(action_dist.log_prob(action))
            self.entropies.append(action_dist.entropy())
        
        return action.item()
    
    def step_update(self, reward: float, next_state, done: bool) -> Optional[Dict[str, float]]:
        """
        Store transition and potentially update.
        
        Args:
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Update metrics if update occurred, None otherwise
        """
        self.rewards.append(reward)
        self.dones.append(done)
        self.step_count += 1
        
        if len(self.rewards) >= self.n_steps or done:
            metrics = self._compute_and_update(next_state, done)
            
            if metrics:
                self.episode_metrics.append(metrics)
            
            # Clear buffers
            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []
            self.log_probs = []
            self.entropies = []
            self.dones = []
            self.encoding_variances = []
            self.pairwise_distances = []
            
            return metrics
        
        return None
    
    def _compute_and_update(self, next_state, done: bool) -> Dict[str, float]:
        """Compute advantages and perform update."""
        if len(self.rewards) == 0:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'total_loss': 0.0
            }
        
        # Bootstrap value for next state
        if done:
            next_value = torch.tensor(0.0)
        else:
            with torch.no_grad():
                if isinstance(next_state, tuple):
                    next_state_input = next_state
                else:
                    next_state_input = torch.tensor([next_state], dtype=torch.long)
                next_value = self.value_net(next_state_input).squeeze()
        
        # Compute returns and advantages
        if self.use_gae:
            returns, advantages = self._compute_gae(next_value)
        else:
            returns, advantages = self._compute_nstep_returns(next_value)
        
        # Convert to tensors
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)
        entropies = torch.stack(self.entropies)
        
        raw_advantages = advantages.clone()
        
        # Adaptive advantage normalization
        adv_magnitude = torch.abs(advantages).mean()
        
        if adv_magnitude > 1.0:
            normalized_advantages = advantages
        else:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 1e-6:
                normalized_advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            else:
                normalized_advantages = advantages * 10.0
        
        normalized_advantages = torch.clamp(normalized_advantages, -10.0, 10.0)
        
        # Compute losses
        policy_loss = -(log_probs * normalized_advantages.detach()).mean()
        value_loss = ((returns - values) ** 2).mean()
        entropy_loss = -entropies.mean()
        
        # Encoding diversity loss (hybrid networks only)
        encoding_diversity_loss = torch.tensor(0.0)
        if self.is_hybrid_network and self.use_encoding_diversity and len(self.encoding_variances) > 0:
            feature_dim = self.policy_net.feature_encoder[-2].out_features
            mean_variance = torch.stack(self.encoding_variances).mean()
            normalized_variance = mean_variance / (feature_dim / 12.0)
            normalized_variance = torch.clamp(normalized_variance, 1e-8, 1.0)
            encoding_diversity_loss = -torch.log(normalized_variance)
        
        # Combined loss
        total_loss = (policy_loss + 
                     self.value_coef * value_loss + 
                     self.entropy_coef * entropy_loss + 
                     self.encoding_diversity_coef * encoding_diversity_loss)
        
        # Update
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        
        # Capture gradients if monitor enabled
        gradient_stats = {}
        if self.gradient_monitor is not None:
            gradient_stats = self.gradient_monitor.capture_gradients()
        
        # Clip gradients
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        value_grad_norm = torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
        
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        self.update_count += 1
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'encoding_diversity_loss': encoding_diversity_loss.item() if self.is_hybrid_network and self.use_encoding_diversity else 0.0,
            'total_loss': total_loss.item(),
            'mean_value': values.mean().item(),
            'mean_advantage': raw_advantages.mean().item(),
            'advantage_std': raw_advantages.std().item(),
            'policy_grad_norm': policy_grad_norm.item() if isinstance(policy_grad_norm, torch.Tensor) else policy_grad_norm,
            'value_grad_norm': value_grad_norm.item() if isinstance(value_grad_norm, torch.Tensor) else value_grad_norm
        }
        
        if self.is_hybrid_network and self.use_encoding_diversity and len(self.encoding_variances) > 0:
            metrics['mean_encoding_variance'] = torch.stack(self.encoding_variances).mean().item()
            metrics['mean_pairwise_distance'] = torch.stack(self.pairwise_distances).mean().item()
        
        if gradient_stats:
            metrics.update({f'grad_{k}': v for k, v in gradient_stats.items()})
        
        return metrics
    
    def _compute_nstep_returns(self, next_value):
        """Compute N-step returns and advantages."""
        returns = []
        
        G = next_value.detach() if isinstance(next_value, torch.Tensor) else next_value
        for i in reversed(range(len(self.rewards))):
            done_mask = 1.0 - float(self.dones[i])
            G = self.rewards[i] + self.gamma * G * done_mask
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(self.values)
        advantages = returns - values.detach()
        
        return returns, advantages
    
    def _compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = []
        gae = 0
        
        values_tensor = torch.stack(self.values)
        values_detached = values_tensor.detach()
        next_value_detached = next_value.detach() if isinstance(next_value, torch.Tensor) else torch.tensor(next_value)
        
        for i in reversed(range(len(self.rewards))):
            done_mask = 1.0 - float(self.dones[i])
            
            if i == len(self.rewards) - 1:
                next_val = next_value_detached
            else:
                next_val = values_detached[i + 1]
            
            delta = self.rewards[i] + self.gamma * next_val * done_mask - values_detached[i]
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values_detached
        
        return returns, advantages
    
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """
        Update called at end of episode.
        Returns aggregated metrics from all step updates.
        """
        self.episode_count += 1
        
        if self.episode_metrics:
            metrics = {
                key: sum(m.get(key, 0) for m in self.episode_metrics) / len(self.episode_metrics)
                for key in self.episode_metrics[0].keys()
            }
        else:
            metrics = {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'total_loss': 0.0,
                'mean_value': 0.0,
                'mean_advantage': 0.0
            }
        
        self.episode_metrics = []
        self.reset_episode()
        
        return metrics
    
    def reset_episode(self):
        """Reset episode storage."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.entropies = []
        self.dones = []
    
    def save(self, filepath: str):
        """Save agent state including network configuration."""
        save_dict = {
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'update_count': self.update_count,
            'reward_history': list(self.reward_history),
            # Training configuration for reproducibility
            'seed': self.seed,
            'training_config': self.training_config,
        }
        
        # Save network configuration for reproducibility
        if self.is_hybrid_network:
            # Extract postprocessing hidden sizes by inspecting the sequential
            postproc_hidden = []
            postproc = getattr(self.policy_net, 'postprocessing', None)
            if postproc is not None:
                for layer in postproc:
                    if hasattr(layer, 'out_features'):
                        # All but the last Linear layer are hidden layers
                        postproc_hidden.append(layer.out_features)
                # Remove last layer (output layer)
                if postproc_hidden:
                    postproc_hidden = postproc_hidden[:-1]
            
            # Extract encoder hidden sizes
            encoder_hidden = []
            encoder = getattr(self.policy_net, 'feature_encoder', None)
            if encoder is not None:
                for layer in encoder:
                    if hasattr(layer, 'out_features'):
                        encoder_hidden.append(layer.out_features)
                # Remove last layer (output layer)
                if encoder_hidden:
                    encoder_hidden = encoder_hidden[:-1]
            
            save_dict['network_config'] = {
                'type': 'hybrid',
                # Quantum circuit configuration
                'n_qubits': getattr(self.policy_net, 'n_qubits', None),
                'n_layers': getattr(self.policy_net, 'n_layers', None),
                'encoding': getattr(self.policy_net, 'encoding', None),
                'entanglement_strategy': getattr(self.policy_net, 'entanglement_strategy', None),
                'measurement_mode': getattr(self.policy_net, 'measurement_mode', None),
                'data_reuploading': getattr(self.policy_net, 'data_reuploading', None),
                'device_name': getattr(self.policy_net, 'device_name', 'default.qubit'),
                # Encoding strategy (CRITICAL for reproducibility)
                'single_axis_encoding': getattr(self.policy_net, 'single_axis_encoding', True),
                'encoder_compression': getattr(self.policy_net, 'encoder_compression', 'minimal'),
                'encoding_transform': getattr(self.policy_net, 'encoding_transform', 'arctan'),
                'reuploading_transform': getattr(self.policy_net, 'reuploading_transform', 'arctan'),
                'encoding_scale': getattr(self.policy_net, 'encoding_scale', 2.0),
                # Classical components
                'learnable_input_scaling': getattr(self.policy_net, 'learnable_input_scaling', False),
                'encoder_layers': encoder_hidden if encoder_hidden else None,
                'postprocessing_layers': postproc_hidden if postproc_hidden else None,
                # Control modes (dropout, frozen encoder)
                'quantum_dropout_rate': getattr(self.policy_net, '_quantum_dropout_rate', 0.0),
                'frozen_components': list(getattr(self.policy_net, '_frozen_components', set())),
            }
        else:
            save_dict['network_config'] = {
                'type': 'classical',
                'hidden_sizes': getattr(self.policy_net, 'hidden_sizes', None),
                'activation': getattr(self.policy_net, 'activation_name', None),
            }
        
        torch.save(save_dict, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        
        if 'policy_optimizer_state_dict' in checkpoint:
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        if 'value_optimizer_state_dict' in checkpoint:
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        if 'episode_count' in checkpoint:
            self.episode_count = checkpoint['episode_count']
        if 'step_count' in checkpoint:
            self.step_count = checkpoint['step_count']
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']
        if 'reward_history' in checkpoint:
            self.reward_history = deque(checkpoint['reward_history'], maxlen=100)
