"""Control mechanisms: freeze, bypass, quantum dropout."""

import torch
import torch.nn as nn
import random
from typing import Union


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
