"""Control mechanisms: freeze, bypass, quantum dropout."""

import torch
import torch.nn as nn
import random
from typing import Union

from .config import BypassMode


class FreezeControls:
    """Mixin for freezing/unfreezing network components."""
    
    def freeze_component(self, component: str):
        """Freeze a component (stop gradient updates).
        
        Args:
            component: 'encoder', 'quantum', 'postprocessing', 'scaling', or 'classical'
                       'classical' freezes encoder + postprocessing + scaling
        """
        if component == 'classical':
            self.freeze_component('encoder')
            self.freeze_component('postprocessing')
            self.freeze_component('scaling')
            return
        
        self._frozen_components.add(component)
        
        if component == 'encoder':
            for p in self.feature_encoder.parameters():
                p.requires_grad = False
        elif component == 'quantum':
            self.weights.requires_grad = False
        elif component == 'postprocessing':
            for p in self.postprocessing.parameters():
                p.requires_grad = False
        elif component == 'scaling' and self.learnable_input_scaling:
            self.input_scale.requires_grad = False
            self.input_bias.requires_grad = False
    
    def unfreeze_component(self, component: str):
        """Unfreeze a component (enable gradient updates).
        
        Args:
            component: 'encoder', 'quantum', 'postprocessing', 'scaling', 'classical', or 'all'
        """
        if component == 'all':
            for c in ['encoder', 'quantum', 'postprocessing', 'scaling']:
                self.unfreeze_component(c)
            return
        
        if component == 'classical':
            self.unfreeze_component('encoder')
            self.unfreeze_component('postprocessing')
            self.unfreeze_component('scaling')
            return
        
        self._frozen_components.discard(component)
        
        if component == 'encoder':
            for p in self.feature_encoder.parameters():
                p.requires_grad = True
        elif component == 'quantum':
            self.weights.requires_grad = True
        elif component == 'postprocessing':
            for p in self.postprocessing.parameters():
                p.requires_grad = True
        elif component == 'scaling' and self.learnable_input_scaling:
            self.input_scale.requires_grad = True
            self.input_bias.requires_grad = True
    
    def get_frozen_components(self) -> set:
        """Get set of frozen component names."""
        return self._frozen_components.copy()


class BypassControls:
    """Mixin for bypass mode control."""
    
    def set_bypass_mode(self, mode: Union[str, BypassMode]):
        """Set bypass mode for quantum circuit.
        
        Args:
            mode: 'none', 'zeros', 'encoder', 'noise', or 'scrambler'
                - none: Normal quantum processing
                - zeros: Replace quantum output with zeros
                - encoder: Pass scaled encoder output directly (truncated/padded)
                - noise: Replace with random noise in [-1, 1]
                - scrambler: Freeze encoder with random weights (data scrambler)
        """
        if isinstance(mode, str):
            mode = BypassMode(mode.lower())
        self._bypass_mode = mode
        
        # If switching to scrambler mode, initialize random encoder
        if mode == BypassMode.SCRAMBLER:
            self.initialize_scrambler_mode()
    
    def initialize_scrambler_mode(self):
        """Initialize scrambler mode: randomize encoder weights and freeze them.
        
        This makes the encoder a 'scrambler' - it transforms game data randomly
        but consistently. Useful for testing if the network can learn despite
        noisy/scrambled input features.
        """
        # Reinitialize all encoder weights with random values
        for module in self.feature_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -1.0, 1.0)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.5, 0.5)
        
        # Freeze the encoder so these random weights stay constant
        self.freeze_component('encoder')
    
    def get_bypass_mode(self) -> BypassMode:
        """Get current bypass mode."""
        return self._bypass_mode
    
    def get_quantum_output_with_bypass(
        self,
        encoded: torch.Tensor,
        batch_size: int,
        quantum_output_dim: int
    ) -> torch.Tensor:
        """Get quantum output with bypass mode support."""
        if self._bypass_mode == BypassMode.NONE:
            return self._run_quantum(encoded)
        
        elif self._bypass_mode == BypassMode.ZEROS:
            return torch.zeros(batch_size, quantum_output_dim, 
                             device=encoded.device, dtype=encoded.dtype)
        
        elif self._bypass_mode == BypassMode.ENCODER:
            # Truncate or pad encoder output to match quantum output dim
            if encoded.shape[-1] >= quantum_output_dim:
                return encoded[:, :quantum_output_dim]
            else:
                padding = torch.zeros(batch_size, 
                                    quantum_output_dim - encoded.shape[-1],
                                    device=encoded.device, dtype=encoded.dtype)
                return torch.cat([encoded, padding], dim=-1)
        
        elif self._bypass_mode == BypassMode.NOISE:
            return torch.rand(batch_size, quantum_output_dim,
                            device=encoded.device, dtype=encoded.dtype) * 2 - 1
        
        elif self._bypass_mode == BypassMode.SCRAMBLER:
            # Scrambler mode: encoder has frozen random weights, pass quantum normally
            # The scrambling happens in the encoder itself (weights are frozen and random)
            return self._run_quantum(encoded)
        
        else:
            return self._run_quantum(encoded)


class QuantumDropout:
    """Mixin for quantum weight dropout."""
    
    def set_quantum_dropout(self, rate: float):
        """Set quantum dropout rate (random weight freezing per forward pass).
        
        This prevents classical components from learning to compensate for
        specific quantum behavior, forcing more robust encoder→quantum coupling.
        
        Args:
            rate: Fraction of quantum weights to freeze (0.0 to 1.0)
        """
        self._quantum_dropout_rate = max(0.0, min(1.0, rate))
    
    def get_quantum_dropout(self) -> float:
        """Get current quantum dropout rate."""
        return self._quantum_dropout_rate
    
    def apply_quantum_dropout(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply quantum dropout: create masked weights with some frozen.
        
        Args:
            weights: Original quantum weights
            
        Returns:
            Effective weights for this forward pass
        """
        if self._quantum_dropout_rate <= 0.0 or not self.training:
            return weights
        
        # Create mask: 1 = use weight, 0 = use detached (frozen) weight
        mask_shape = weights.shape
        n_weights = weights.numel()
        n_frozen = int(n_weights * self._quantum_dropout_rate)
        
        # Random indices to freeze
        frozen_indices = random.sample(range(n_weights), n_frozen)
        mask = torch.ones(n_weights, device=weights.device)
        mask[frozen_indices] = 0.0
        mask = mask.reshape(mask_shape)
        
        # Combine: gradient flows through mask=1, blocked for mask=0
        effective_weights = mask * weights + (1 - mask) * weights.detach()
        return effective_weights
