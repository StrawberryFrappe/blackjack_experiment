"""
Session management for training outputs.

This module provides utilities for organizing training outputs:
- SessionManager: Manages individual training session directories
- ComparisonSessionManager: Manages comparison experiments with multiple networks
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import torch


class SessionManager:
    """
    Manages training session directories and outputs.
    
    Each session has its own directory with:
    - Model checkpoints
    - Plots and visualizations
    - Gradient analysis data
    - Training logs
    - Session metadata (session_info.json)
    """
    
    def __init__(
        self,
        base_dir: str = "results",
        session_name: Optional[str] = None,
        network_type: Optional[str] = None,
        auto_timestamp: bool = False,
        agent_type: str = "a2c"
    ):
        """
        Initialize session manager.
        
        Args:
            base_dir: Base directory for all results
            session_name: Custom session name (auto-generated if None)
            network_type: Network type (e.g., 'classical', 'hybrid')
            auto_timestamp: Add timestamp to session name
            agent_type: Agent type (default: 'a2c')
        """
        self.base_dir = Path(base_dir)
        self.network_type = network_type
        self.agent_type = agent_type
        
        # Generate session name
        if session_name is None:
            session_name = f"a2c_{network_type}" if network_type else "training"
        
        if auto_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"{session_name}_{timestamp}"
        
        self.session_name = session_name
        self.session_dir = self.base_dir / session_name
        
        # Create directory structure
        self._create_structure()
        
        # Session metadata
        self.metadata = {
            'session_name': session_name,
            'network_type': network_type,
            'created_at': datetime.now().isoformat(),
            'loaded_from': None,
            'config': {}
        }
    
    def _create_structure(self):
        """Create session directory structure."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoints subdirectory for organized storage
        self.checkpoints_dir = self.session_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # All other outputs go directly in session directory
        self.models_dir = self.session_dir
        self.plots_dir = self.session_dir
        self.gradients_dir = self.session_dir
        self.logs_dir = self.session_dir
    
    def copy_existing_model(self, source_model_path: str) -> Path:
        """
        Copy an existing model to this session's directory.
        
        Args:
            source_model_path: Path to existing model checkpoint
            
        Returns:
            Path to the copied model
        """
        source_path = Path(source_model_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source model not found: {source_model_path}")
        
        dest_name = f"loaded_{source_path.name}"
        dest_path = self.models_dir / dest_name
        
        shutil.copy2(source_path, dest_path)
        
        self.metadata['loaded_from'] = str(source_path)
        self.save_metadata()
        
        print(f"[OK] Copied model to session: {dest_path}")
        return dest_path
    
    def get_model_path(self, name: str = "model", suffix: str = ".pth") -> Path:
        """Get path for saving a model."""
        return self.models_dir / f"{name}{suffix}"
    
    def get_checkpoint_path(self, episode: int, win_rate: Optional[float] = None) -> Path:
        """Get path for saving a checkpoint."""
        if win_rate is not None:
            filename = f"checkpoint_ep{episode}_wr{win_rate:.1f}.pth"
        else:
            filename = f"checkpoint_ep{episode}.pth"
        return self.checkpoints_dir / filename
    
    def get_final_model_path(self, win_rate: Optional[float] = None, 
                            avg_reward: Optional[float] = None) -> Path:
        """Get path for saving the final trained model."""
        parts = ["final"]
        if win_rate is not None:
            parts.append(f"wr{win_rate:.1f}")
        if avg_reward is not None:
            parts.append(f"reward{avg_reward:.2f}")
        filename = "_".join(parts) + ".pth"
        return self.models_dir / filename
    
    def get_plot_path(self, name: str, extension: str = ".png") -> Path:
        """Get path for saving a plot."""
        return self.plots_dir / f"{name}{extension}"
    
    def get_log_path(self, name: str = "training", extension: str = ".log") -> Path:
        """Get path for saving a log file."""
        return self.logs_dir / f"{name}{extension}"
    
    def get_gradient_dir(self) -> Path:
        """Get the gradients directory for this session."""
        return self.gradients_dir
    
    def add_config(self, config: Dict[str, Any]):
        """Add configuration information to session metadata."""
        self.metadata['config'].update(config)
        self.save_metadata()
    
    def add_results(self, results: Dict[str, Any]):
        """Add results information to session metadata."""
        if 'results' not in self.metadata:
            self.metadata['results'] = {}
        self.metadata['results'].update(results)
        self.save_metadata()
    
    def save_metadata(self):
        """Save session metadata to JSON file."""
        metadata_path = self.session_dir / "session_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load session metadata from JSON file."""
        metadata_path = self.session_dir / "session_info.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        return self.metadata
    
    def save_training_metadata(self, 
                              config: Any,
                              agent_info: Optional[Dict] = None,
                              network_info: Optional[Dict] = None,
                              seed: Optional[int] = None):
        """Save comprehensive training metadata for reproducibility."""
        import numpy as np
        import sys
        
        from .config import Config
        
        # Convert config to dict
        if isinstance(config, Config):
            # Build network config based on type
            network_config = {
                'network_type': config.network.network_type,
                'use_encoding_diversity': config.network.use_encoding_diversity,
                'encoding_diversity_coef': config.network.encoding_diversity_coef
            }
            
            # Add type-specific settings
            if config.network.network_type == 'hybrid':
                # For hybrid, use network_info if provided, else use config defaults
                if network_info:
                    network_config['n_qubits'] = network_info.get('n_qubits', config.network.n_qubits)
                    network_config['n_layers'] = network_info.get('n_layers', config.network.n_layers)
                else:
                    network_config['n_qubits'] = config.network.n_qubits
                    network_config['n_layers'] = config.network.n_layers
            else:
                # For classical, save hidden_sizes
                network_config['hidden_sizes'] = config.network.hidden_sizes
                network_config['activation'] = config.network.activation
            
            config_dict = {
                'seed': config.seed,
                'environment': {
                    'env_name': config.env_name,
                    'n_actions': config.n_actions
                },
                'network': network_config,
                'agent': {
                    'learning_rate': config.agent.learning_rate,
                    'gamma': config.agent.gamma,
                    'entropy_coef': config.agent.entropy_coef,
                    'value_coef': config.agent.value_coef,
                    'max_grad_norm': config.agent.max_grad_norm,
                    'n_steps': config.agent.n_steps,
                    'use_gae': config.agent.use_gae,
                    'gae_lambda': config.agent.gae_lambda
                },
                'training': {
                    'n_episodes': config.training.n_episodes,
                    'max_steps_per_episode': config.training.max_steps_per_episode,
                    'print_every': config.training.print_every,
                    'eval_every': config.training.eval_every,
                    'eval_episodes': config.training.eval_episodes,
                    'save_frequency': config.training.save_frequency,
                    'enable_analysis': config.training.enable_analysis,
                    'monitor_gradients': config.training.monitor_gradients
                }
            }
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {}
        
        training_metadata = {
            'timestamp': datetime.now().isoformat(),
            'seed': seed if seed is not None else config_dict.get('seed'),
            'config': config_dict,
            'agent_info': agent_info or {},
            'network_info': network_info or {},
            'environment': {
                'python_version': sys.version,
                'numpy_version': np.__version__,
                'torch_version': torch.__version__
            }
        }
        
        metadata_path = self.session_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        self.metadata['training_metadata'] = training_metadata
        self.save_metadata()
        
        print(f"[DATA] Training metadata saved: {metadata_path}")
    
    def get_session_summary(self) -> str:
        """Get a formatted summary of the session."""
        lines = [
            "=" * 70,
            f"SESSION: {self.session_name}",
            "=" * 70,
            f"Location: {self.session_dir}",
        ]
        
        if self.network_type:
            lines.append(f"Network: {self.network_type}")
        
        if self.metadata.get('loaded_from'):
            lines.append(f"Loaded from: {self.metadata['loaded_from']}")
        
        lines.extend([
            "",
            f"All outputs in: {self.session_dir}",
            "=" * 70
        ])
        
        return "\n".join(lines)
    
    def print_summary(self):
        """Print session summary."""
        print(self.get_session_summary())
    
    @staticmethod
    def list_sessions(base_dir: str = "results") -> List[Dict]:
        """List all existing sessions."""
        base_path = Path(base_dir)
        
        if not base_path.exists():
            return []
        
        sessions = []
        for item in base_path.iterdir():
            if item.is_dir():
                metadata_path = item / "session_info.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    sessions.append({
                        'path': item,
                        'name': item.name,
                        'metadata': metadata
                    })
        
        return sessions


class ComparisonSessionManager:
    """
    Manages comparison sessions for different network types.
    
    Structure:
    results/
        comparison_<timestamp>/
            a2c_<network_1>/
            a2c_<network_2>/
            comparison_results/
            comparison_info.json
    """
    
    def __init__(
        self,
        base_dir: str = "results",
        comparison_name: Optional[str] = None
    ):
        """
        Initialize comparison session manager.
        
        Args:
            base_dir: Base directory for all results
            comparison_name: Custom comparison name (auto-generated if None)
        """
        self.base_dir = Path(base_dir)
        
        if comparison_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_name = f"comparison_{timestamp}"
        
        self.comparison_name = comparison_name
        self.comparison_dir = self.base_dir / comparison_name
        
        # Create structure
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.comparison_dir / "comparison_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Track individual sessions
        self.sessions = {}
        
        # Metadata
        self.metadata = {
            'comparison_name': comparison_name,
            'created_at': datetime.now().isoformat(),
            'configurations': []
        }
    
    def create_session(self, network_type: str) -> SessionManager:
        """Create a session for a specific network type."""
        session_name = f"a2c_{network_type}"
        
        session = SessionManager(
            base_dir=str(self.comparison_dir),
            session_name=session_name,
            network_type=network_type,
            auto_timestamp=False
        )
        
        self.sessions[session_name] = session
        
        self.metadata['configurations'].append({
            'network_type': network_type,
            'session_name': session_name
        })
        self.save_metadata()
        
        return session
    
    def get_comparison_plot_path(self, name: str = "comparison") -> Path:
        """Get path for comparison plot."""
        return self.results_dir / f"{name}.png"
    
    def get_summary_path(self) -> Path:
        """Get path for comparison summary JSON."""
        return self.results_dir / "summary.json"
    
    def save_metadata(self):
        """Save comparison metadata."""
        metadata_path = self.comparison_dir / "comparison_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_summary(self, summary: Dict[str, Any]):
        """Save comparison summary."""
        summary_path = self.get_summary_path()
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def print_summary(self):
        """Print comparison summary."""
        lines = [
            "=" * 70,
            f"COMPARISON: {self.comparison_name}",
            "=" * 70,
            f"Location: {self.comparison_dir}",
            f"Networks: {len(self.sessions)}",
            ""
        ]
        
        for session_name, session in self.sessions.items():
            lines.append(f"  - {session_name}: {session.session_dir}")
        
        lines.extend([
            "",
            f"Results: {self.results_dir}",
            "=" * 70
        ])
        
        print("\n".join(lines))
