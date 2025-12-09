"""Analysis module for Blackjack RL experiments.

Provides core analysis capabilities:

1. **strategy.py** (BehaviorAnalyzer) - Decision heatmaps, basic strategy comparison
   Previously: decisions.py
   
2. **learning.py** (LearningAnalyzer) - When and how models learn strategy

3. **gradient_flow.py** (GradientFlowAnalyzer) - Gradient flow through quantum vs classical
   Previously: quantum.py / QuantumContributionAnalyzer
   
4. **signal_noise.py** (SignalNoiseAnalyzer) - Signal vs noise in quantum output  
   Previously: quantum_contribution.py / QuantumContributionAnalyzer (the OTHER one!)
   
5. **intermediate.py** (IntermediateOutputAnalyzer) - Stage-by-stage hybrid network analysis

6. **comparison.py** (ComparisonAnalyzer) - Compare behavior between models

7. **capacity.py** (CapacityAnalyzer) - Network capacity analysis and bottleneck detection

Shared utilities in utils.py: generate_all_blackjack_states, get_basic_strategy_action
"""

from .learning import LearningAnalyzer

# Strategy analysis (previously decisions.py)
from .strategy import BehaviorAnalyzer, compare_decisions, get_basic_strategy_action
# Backward compatibility aliases
from .strategy import DecisionAnalyzer, ModelAnalyzer

# Gradient flow analysis during training (previously quantum.py)
from .gradient_flow import (
    GradientFlowAnalyzer,
    analyze_model_quantum_contribution,
    compare_quantum_classical_performance,
)
# Backward compatibility aliases  
from .gradient_flow import QuantumContributionAnalyzer, HybridGradientAnalyzer

# Signal vs noise analysis (previously quantum_contribution.py)
from .signal_noise import (
    SignalNoiseAnalyzer,
    analyze_checkpoint as analyze_quantum_contribution,
)
# Note: SignalNoiseAnalyzer is also exported as QuantumContributionAnalyzer 
# for backward compat, but prefer using the new name

from .intermediate import IntermediateOutputAnalyzer, analyze_hybrid_intermediates
from .comparison import ComparisonAnalyzer, compare_models_from_checkpoints

# Network capacity analysis
from .capacity import CapacityAnalyzer, analyze_network_capacity

# Quantum-first experiments
from .experiments.qfirst import QuantumFirstTrainer

# Utilities
from .utils import generate_all_blackjack_states, create_blackjack_test_states

__all__ = [
    # Core analyzers (new names)
    'LearningAnalyzer',
    'BehaviorAnalyzer',
    'GradientFlowAnalyzer',
    'SignalNoiseAnalyzer',
    'IntermediateOutputAnalyzer',
    'ComparisonAnalyzer',
    
    # Backward compatibility aliases (deprecated names)
    'DecisionAnalyzer',  # → BehaviorAnalyzer
    'ModelAnalyzer',     # → BehaviorAnalyzer
    'QuantumContributionAnalyzer',  # → GradientFlowAnalyzer (in gradient_flow.py)
    'HybridGradientAnalyzer',       # → GradientFlowAnalyzer
    
    # Functions
    'compare_decisions',
    'get_basic_strategy_action',
    'analyze_model_quantum_contribution',
    'compare_quantum_classical_performance',
    'analyze_quantum_contribution',
    'analyze_hybrid_intermediates',
    'compare_models_from_checkpoints',
    
    # Quantum-first experiments
    'QuantumFirstTrainer',
    
    # Utilities
    'generate_all_blackjack_states',
    'create_blackjack_test_states',
]
