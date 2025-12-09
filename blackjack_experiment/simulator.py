"""
Interactive Blackjack State Simulator with Network Visualization.

A pygame-based tool to:
1. Select any Blackjack state (player_sum, dealer_card, usable_ace)
2. Visualize the model's decision and confidence
3. For hybrid networks: show outputs at each stage (encoder → quantum → action)

Usage:
    # Interactive mode (select from results folder)
    python -m blackjack_experiment.simulator
    
    # Direct mode with specific checkpoint
    python -m blackjack_experiment.simulator --checkpoint path/to/model.pth
    
    # Select from specific experiment
    python -m blackjack_experiment.simulator --dir results/comparison_20251205_142648
"""

import sys
from pathlib import Path

# Add parent directory to path to enable imports
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

import pygame
import torch
import numpy as np
from typing import Dict, Optional, Tuple
import argparse


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
GREEN = (34, 139, 34)
RED = (220, 20, 60)
BLUE = (65, 105, 225)
GOLD = (255, 215, 0)
PURPLE = (138, 43, 226)
CYAN = (0, 206, 209)
ORANGE = (255, 140, 0)


class BlackjackSimulator:
    """
    Interactive simulator for visualizing model decisions on Blackjack states.
    """
    
    def __init__(
        self,
        policy_net,
        network_type: str = 'classical',
        width: int = 1400,
        height: int = 850
    ):
        """
        Initialize simulator.
        
        Args:
            policy_net: Trained policy network
            network_type: 'classical' or 'hybrid'
            width: Window width
            height: Window height
        """
        pygame.init()
        pygame.display.set_caption("Blackjack RL State Simulator")
        
        self.screen = pygame.display.set_mode((width, height))
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        
        self.policy_net = policy_net
        self.policy_net.eval()
        self.network_type = network_type
        self.is_hybrid = network_type == 'hybrid' and hasattr(policy_net, 'forward_with_intermediates')
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 18)
        
        # State
        self.player_sum = 15
        self.dealer_card = 7
        self.usable_ace = 0
        
        # UI elements
        self.buttons = {}
        self._create_buttons()
        
        # For quantum contribution analysis (cached to avoid recomputing every frame)
        self.quantum_contribution = None
        self.last_analyzed_state = None
        
        # Cached analysis
        self.current_analysis = None
        self._update_analysis()
    
    def _create_buttons(self):
        """Create interactive buttons."""
        # Player sum buttons
        self.buttons['player_up'] = pygame.Rect(250, 100, 40, 30)
        self.buttons['player_down'] = pygame.Rect(250, 140, 40, 30)
        
        # Dealer card buttons
        self.buttons['dealer_up'] = pygame.Rect(250, 200, 40, 30)
        self.buttons['dealer_down'] = pygame.Rect(250, 240, 40, 30)
        
        # Usable ace toggle
        self.buttons['ace_toggle'] = pygame.Rect(250, 300, 80, 30)
        
        # Quick state buttons
        y_start = 380
        self.buttons['state_bust'] = pygame.Rect(50, y_start, 120, 30)
        self.buttons['state_low'] = pygame.Rect(180, y_start, 120, 30)
        self.buttons['state_mid'] = pygame.Rect(50, y_start + 40, 120, 30)
        self.buttons['state_high'] = pygame.Rect(180, y_start + 40, 120, 30)
    
    def _update_analysis(self):
        """Update model analysis for current state."""
        state = (self.player_sum, self.dealer_card, self.usable_ace)
        
        # Invalidate quantum contribution cache when state changes
        if self.last_analyzed_state != state:
            self.quantum_contribution = None
        
        with torch.no_grad():
            if self.is_hybrid:
                self.current_analysis = self.policy_net.forward_with_intermediates(state)
            else:
                action_probs = self.policy_net(state)
                self.current_analysis = {
                    'action_probs': action_probs
                }
    
    def _draw_state_controls(self):
        """Draw state selection controls."""
        # Title
        title = self.font_large.render("Blackjack State", True, WHITE)
        self.screen.blit(title, (50, 30))
        
        # Player sum
        player_label = self.font_medium.render(f"Player Sum: {self.player_sum}", True, WHITE)
        self.screen.blit(player_label, (50, 110))
        pygame.draw.rect(self.screen, GREEN, self.buttons['player_up'])
        pygame.draw.rect(self.screen, RED, self.buttons['player_down'])
        up_text = self.font_small.render("+", True, WHITE)
        down_text = self.font_small.render("-", True, WHITE)
        self.screen.blit(up_text, (self.buttons['player_up'].x + 15, self.buttons['player_up'].y + 5))
        self.screen.blit(down_text, (self.buttons['player_down'].x + 15, self.buttons['player_down'].y + 5))
        
        # Dealer card
        dealer_label = self.font_medium.render(f"Dealer Card: {self.dealer_card}", True, WHITE)
        self.screen.blit(dealer_label, (50, 210))
        pygame.draw.rect(self.screen, GREEN, self.buttons['dealer_up'])
        pygame.draw.rect(self.screen, RED, self.buttons['dealer_down'])
        self.screen.blit(up_text, (self.buttons['dealer_up'].x + 15, self.buttons['dealer_up'].y + 5))
        self.screen.blit(down_text, (self.buttons['dealer_down'].x + 15, self.buttons['dealer_down'].y + 5))
        
        # Usable ace
        ace_label = self.font_medium.render(f"Usable Ace: {'Yes' if self.usable_ace else 'No'}", True, WHITE)
        self.screen.blit(ace_label, (50, 310))
        ace_color = GOLD if self.usable_ace else GRAY
        pygame.draw.rect(self.screen, ace_color, self.buttons['ace_toggle'])
        toggle_text = self.font_small.render("Toggle", True, BLACK)
        self.screen.blit(toggle_text, (self.buttons['ace_toggle'].x + 10, self.buttons['ace_toggle'].y + 5))
        
        # Quick state buttons
        quick_label = self.font_medium.render("Quick States:", True, LIGHT_GRAY)
        self.screen.blit(quick_label, (50, 350))
        
        pygame.draw.rect(self.screen, PURPLE, self.buttons['state_bust'])
        pygame.draw.rect(self.screen, BLUE, self.buttons['state_low'])
        pygame.draw.rect(self.screen, CYAN, self.buttons['state_mid'])
        pygame.draw.rect(self.screen, GREEN, self.buttons['state_high'])
        
        self.screen.blit(self.font_small.render("Risky (16,10)", True, WHITE), (55, 385))
        self.screen.blit(self.font_small.render("Low (8,5)", True, WHITE), (190, 385))
        self.screen.blit(self.font_small.render("Mid (15,7)", True, WHITE), (60, 425))
        self.screen.blit(self.font_small.render("High (20,6)", True, WHITE), (185, 425))
    
    def _draw_decision_panel(self):
        """Draw the main decision panel."""
        panel_x = 350
        panel_y = 30
        panel_w = 400
        panel_h = 180
        
        # Panel background
        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x, panel_y, panel_w, panel_h), border_radius=10)
        pygame.draw.rect(self.screen, WHITE, (panel_x, panel_y, panel_w, panel_h), 2, border_radius=10)
        
        # Title
        title = self.font_large.render("MODEL DECISION", True, WHITE)
        self.screen.blit(title, (panel_x + 80, panel_y + 10))
        
        # Get action probabilities
        action_probs = self.current_analysis['action_probs'].squeeze().numpy()
        stick_prob = action_probs[0]
        hit_prob = action_probs[1]
        decision = "HIT" if hit_prob > stick_prob else "STICK"
        confidence = max(stick_prob, hit_prob) * 100
        
        # Decision
        decision_color = RED if decision == "HIT" else GREEN
        decision_text = self.font_large.render(decision, True, decision_color)
        self.screen.blit(decision_text, (panel_x + 150, panel_y + 55))
        
        # Confidence bar
        conf_label = self.font_small.render(f"Confidence: {confidence:.1f}%", True, LIGHT_GRAY)
        self.screen.blit(conf_label, (panel_x + 20, panel_y + 100))
        
        bar_x = panel_x + 20
        bar_y = panel_y + 125
        bar_w = panel_w - 40
        bar_h = 20
        pygame.draw.rect(self.screen, GRAY, (bar_x, bar_y, bar_w, bar_h))
        fill_w = int(bar_w * confidence / 100)
        pygame.draw.rect(self.screen, decision_color, (bar_x, bar_y, fill_w, bar_h))
        
        # Probabilities
        prob_text = self.font_small.render(f"STICK: {stick_prob*100:.1f}%  |  HIT: {hit_prob*100:.1f}%", True, LIGHT_GRAY)
        self.screen.blit(prob_text, (panel_x + 80, panel_y + 150))
    
    def _draw_hybrid_pipeline(self):
        """Draw the hybrid network pipeline visualization (vertical bars layout)."""
        if not self.is_hybrid:
            return
        
        # Pipeline flows top to bottom
        pipeline_x = 350
        start_y = 20
        
        # Title
        title = self.font_medium.render("HYBRID NETWORK PIPELINE", True, GOLD)
        self.screen.blit(title, (pipeline_x, start_y))
        
        encoder_out = self.current_analysis['feature_encoder_output'].squeeze().numpy()
        quantum_out = self.current_analysis['quantum_output'].squeeze().numpy()
        action_probs = self.current_analysis['action_probs'].squeeze().numpy()
        
        # Stage 1: Feature Encoder with vertical bars
        stage1_y = start_y + 30
        stage1_h = 200
        self._draw_vertical_bar_stage(
            pipeline_x, stage1_y, 700, stage1_h,
            "1. FEATURE ENCODER OUTPUT",
            encoder_out,
            BLUE
        )
        
        # Down arrow
        arrow_y = stage1_y + stage1_h + 5
        self._draw_down_arrow(pipeline_x + 350, arrow_y, 20)
        
        # Stage 2: Quantum Circuit Output with vertical bars
        stage2_y = arrow_y + 30
        stage2_h = 200
        self._draw_vertical_bar_stage(
            pipeline_x, stage2_y, 700, stage2_h,
            "2. QUANTUM CIRCUIT OUTPUT",
            quantum_out,
            PURPLE
        )
        
        # Down arrow
        arrow2_y = stage2_y + stage2_h + 5
        self._draw_down_arrow(pipeline_x + 350, arrow2_y, 20)
        
        # Stage 3: Action Probabilities
        stage3_y = arrow2_y + 30
        self._draw_action_stage_vertical(
            pipeline_x, stage3_y, 700, 120,
            "3. ACTION PROBABILITIES",
            action_probs
        )
        
        # Statistics panel on the far right
        self._draw_statistics_panel_vertical(1070, start_y + 30, encoder_out, quantum_out)
    
    def _draw_down_arrow(self, x, y, size):
        """Draw a downward pointing arrow."""
        pygame.draw.polygon(self.screen, WHITE, [
            (x, y + size),
            (x - size//2, y),
            (x + size//2, y)
        ])
        pygame.draw.line(self.screen, WHITE, (x, y - 5), (x, y), 3)
    
    def _draw_vertical_bar_stage(self, x, y, w, h, title, values, color):
        """Draw a pipeline stage with vertical bars (columns)."""
        # Background
        pygame.draw.rect(self.screen, DARK_GRAY, (x, y, w, h), border_radius=5)
        pygame.draw.rect(self.screen, color, (x, y, w, h), 2, border_radius=5)
        
        # Title
        title_text = self.font_small.render(title, True, color)
        self.screen.blit(title_text, (x + 10, y + 5))
        
        # Dimension info
        dim_text = self.font_tiny.render(f"dim={len(values)}", True, LIGHT_GRAY)
        self.screen.blit(dim_text, (x + w - 70, y + 8))
        
        # Calculate bar dimensions
        n_values = len(values)
        margin = 50
        bar_area_width = w - margin * 2
        bar_width = max(15, min(40, (bar_area_width - n_values * 2) // n_values))
        spacing = (bar_area_width - n_values * bar_width) // max(1, n_values - 1) if n_values > 1 else 0
        
        # Bar area
        bar_area_y = y + 35
        bar_area_h = h - 70
        center_y = bar_area_y + bar_area_h // 2  # Zero line
        
        # Normalize values
        val_abs_max = max(abs(values.min()), abs(values.max()), 0.001)
        
        for i, val in enumerate(values):
            bar_x = x + margin + i * (bar_width + spacing)
            
            # Background column
            pygame.draw.rect(self.screen, (40, 40, 40), 
                           (bar_x, bar_area_y, bar_width, bar_area_h))
            
            # Center line (zero)
            pygame.draw.line(self.screen, GRAY, 
                           (bar_x, center_y), (bar_x + bar_width, center_y), 1)
            
            # Value bar (from center)
            normalized = val / val_abs_max
            bar_h = int((bar_area_h // 2 - 5) * abs(normalized))
            
            if val >= 0:
                # Green bar going UP from center
                bar_color = (50, int(100 + 155 * abs(normalized)), 50)
                pygame.draw.rect(self.screen, bar_color,
                               (bar_x + 2, center_y - bar_h, bar_width - 4, bar_h))
            else:
                # Red bar going DOWN from center
                bar_color = (int(100 + 155 * abs(normalized)), 50, 50)
                pygame.draw.rect(self.screen, bar_color,
                               (bar_x + 2, center_y, bar_width - 4, bar_h))
            
            # Index label at bottom
            idx_text = self.font_tiny.render(f"{i}", True, LIGHT_GRAY)
            text_x = bar_x + (bar_width - idx_text.get_width()) // 2
            self.screen.blit(idx_text, (text_x, y + h - 18))
            
            # Value label at top
            val_text = self.font_tiny.render(f"{val:.2f}", True, color)
            text_x = bar_x + (bar_width - val_text.get_width()) // 2
            self.screen.blit(val_text, (text_x, bar_area_y - 15))
        
        # Draw zero line label
        zero_label = self.font_tiny.render("0", True, GRAY)
        self.screen.blit(zero_label, (x + 10, center_y - 6))
    
    def _draw_action_stage_vertical(self, x, y, w, h, title, probs):
        """Draw action probability stage with vertical bars."""
        pygame.draw.rect(self.screen, DARK_GRAY, (x, y, w, h), border_radius=5)
        pygame.draw.rect(self.screen, GREEN, (x, y, w, h), 2, border_radius=5)
        
        title_text = self.font_small.render(title, True, GREEN)
        self.screen.blit(title_text, (x + 10, y + 5))
        
        # Decision text
        decision = "HIT" if probs[1] > probs[0] else "STICK"
        decision_color = RED if decision == "HIT" else GREEN
        conf = max(probs) * 100
        
        decision_text = self.font_large.render(f"→ {decision}", True, decision_color)
        self.screen.blit(decision_text, (x + 20, y + 30))
        
        conf_text = self.font_medium.render(f"{conf:.1f}%", True, decision_color)
        self.screen.blit(conf_text, (x + 180, y + 35))
        
        # Vertical bars for STICK and HIT
        bar_area_x = x + 320
        bar_area_y = y + 25
        bar_area_h = h - 45
        bar_width = 80
        bar_spacing = 40
        
        for i, (label, prob, col) in enumerate([("STICK", probs[0], GREEN), ("HIT", probs[1], RED)]):
            bar_x = bar_area_x + i * (bar_width + bar_spacing)
            
            # Background
            pygame.draw.rect(self.screen, (40, 40, 40), 
                           (bar_x, bar_area_y, bar_width, bar_area_h))
            
            # Filled portion (from bottom)
            fill_h = int(bar_area_h * prob)
            pygame.draw.rect(self.screen, col,
                           (bar_x, bar_area_y + bar_area_h - fill_h, bar_width, fill_h))
            
            # Label and percentage
            label_text = self.font_small.render(label, True, WHITE)
            self.screen.blit(label_text, (bar_x + (bar_width - label_text.get_width()) // 2, y + h - 18))
            
            pct_text = self.font_small.render(f"{prob*100:.1f}%", True, col)
            self.screen.blit(pct_text, (bar_x + (bar_width - pct_text.get_width()) // 2, bar_area_y - 18))
        
        # Entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropy_text = self.font_tiny.render(f"Entropy: {entropy:.3f}", True, LIGHT_GRAY)
        self.screen.blit(entropy_text, (x + w - 120, y + h - 18))
    
    def _draw_statistics_panel_vertical(self, x, y, encoder_out, quantum_out):
        """Draw statistics panel for hybrid analysis (vertical layout on right side)."""
        panel_w = 250
        panel_h = 400
        
        pygame.draw.rect(self.screen, DARK_GRAY, (x, y, panel_w, panel_h), border_radius=5)
        pygame.draw.rect(self.screen, CYAN, (x, y, panel_w, panel_h), 2, border_radius=5)
        
        title = self.font_small.render("STATISTICS", True, CYAN)
        self.screen.blit(title, (x + 10, y + 8))
        
        # Encoder stats section
        enc_title = self.font_small.render("Encoder", True, BLUE)
        self.screen.blit(enc_title, (x + 10, y + 40))
        
        enc_stats = [
            f"Mean:  {encoder_out.mean():+.4f}",
            f"Std:   {encoder_out.std():.4f}",
            f"Min:   {encoder_out.min():+.4f}",
            f"Max:   {encoder_out.max():+.4f}",
            f"Norm:  {np.linalg.norm(encoder_out):.4f}",
        ]
        for i, stat in enumerate(enc_stats):
            self.screen.blit(self.font_tiny.render(stat, True, LIGHT_GRAY), (x + 15, y + 65 + i * 18))
        
        # Quantum stats section
        q_title = self.font_small.render("Quantum", True, PURPLE)
        self.screen.blit(q_title, (x + 10, y + 165))
        
        q_stats = [
            f"Mean:  {quantum_out.mean():+.4f}",
            f"Std:   {quantum_out.std():.4f}",
            f"Min:   {quantum_out.min():+.4f}",
            f"Max:   {quantum_out.max():+.4f}",
            f"Norm:  {np.linalg.norm(quantum_out):.4f}",
        ]
        for i, stat in enumerate(q_stats):
            self.screen.blit(self.font_tiny.render(stat, True, LIGHT_GRAY), (x + 15, y + 190 + i * 18))
        
        # Transformation analysis
        trans_title = self.font_small.render("Transform", True, GOLD)
        self.screen.blit(trans_title, (x + 10, y + 295))
        
        enc_var = encoder_out.var()
        q_var = quantum_out.var()
        ratio = q_var / (enc_var + 1e-10)
        
        trans_stats = [
            f"Enc Var:  {enc_var:.4f}",
            f"Q Var:    {q_var:.4f}",
            f"Ratio:    {ratio:.4f}",
        ]
        for i, stat in enumerate(trans_stats):
            self.screen.blit(self.font_tiny.render(stat, True, LIGHT_GRAY), (x + 15, y + 320 + i * 18))
        
        # Interpretation
        if ratio > 1.5:
            interp = "EXPANDS"
            interp_color = GREEN
        elif ratio < 0.5:
            interp = "COMPRESSES"
            interp_color = RED
        else:
            interp = "PRESERVES"
            interp_color = GOLD
        
        interp_text = self.font_small.render(f"Q {interp}", True, interp_color)
        self.screen.blit(interp_text, (x + 15, y + 375))
    
    def _draw_classical_info(self):
        """Draw info panel for classical networks."""
        if self.is_hybrid:
            return
        
        info_text = self.font_medium.render("Classical Network - No intermediate outputs available", True, GRAY)
        self.screen.blit(info_text, (350, 250))
    
    def _draw_instructions(self):
        """Draw usage instructions."""
        y = self.height - 80
        instructions = [
            "Controls: Click +/- to adjust state | Click Toggle for Ace | Click Quick States",
            "Arrow keys: ←/→ dealer card | ↑/↓ player sum | A to toggle ace",
            "Press Q to quit"
        ]
        for i, text in enumerate(instructions):
            inst_text = self.font_tiny.render(text, True, GRAY)
            self.screen.blit(inst_text, (20, y + i * 20))
    
    def handle_click(self, pos):
        """Handle mouse click."""
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                if name == 'player_up':
                    self.player_sum = min(21, self.player_sum + 1)
                elif name == 'player_down':
                    self.player_sum = max(4, self.player_sum - 1)
                elif name == 'dealer_up':
                    self.dealer_card = min(10, self.dealer_card + 1)
                elif name == 'dealer_down':
                    self.dealer_card = max(1, self.dealer_card - 1)
                elif name == 'ace_toggle':
                    self.usable_ace = 1 - self.usable_ace
                elif name == 'state_bust':
                    self.player_sum, self.dealer_card = 16, 10
                elif name == 'state_low':
                    self.player_sum, self.dealer_card = 8, 5
                elif name == 'state_mid':
                    self.player_sum, self.dealer_card = 15, 7
                elif name == 'state_high':
                    self.player_sum, self.dealer_card = 20, 6
                
                self._update_analysis()
                break
    
    def handle_key(self, key):
        """Handle keyboard input."""
        if key == pygame.K_UP:
            self.player_sum = min(21, self.player_sum + 1)
        elif key == pygame.K_DOWN:
            self.player_sum = max(4, self.player_sum - 1)
        elif key == pygame.K_RIGHT:
            self.dealer_card = min(10, self.dealer_card + 1)
        elif key == pygame.K_LEFT:
            self.dealer_card = max(1, self.dealer_card - 1)
        elif key == pygame.K_a:
            self.usable_ace = 1 - self.usable_ace
        
        self._update_analysis()
    
    def analyze_quantum_contribution(self, state: tuple) -> dict:
        """
        Analyze if quantum circuit actually affects the decision.
        Only runs when state changes (cached).
        """
        if not self.is_hybrid:
            return None
        
        # Return cached result if state hasn't changed
        if self.last_analyzed_state == state and self.quantum_contribution is not None:
            return self.quantum_contribution
        
        self.last_analyzed_state = state
            
        with torch.no_grad():
            # Get normal forward pass
            intermediates = self.policy_net.forward_with_intermediates(state)
            normal_probs = intermediates['action_probs'].squeeze().numpy()
            normal_action = int(np.argmax(normal_probs))
            
            encoder_out = intermediates['feature_encoder_output'].squeeze()
            quantum_out = intermediates['quantum_output'].squeeze()
            
            # ===== LINEARITY CHECK =====
            # Is quantum output approximately a linear function of encoder output?
            # If so, quantum circuit is just a "wire" and encoder talks directly to postprocessor
            encoder_np = encoder_out.numpy()
            quantum_np = quantum_out.numpy()
            
            # Compute correlation between encoder (reshaped) and quantum output
            # High correlation = quantum is linear passthrough
            enc_flat = encoder_np.flatten()
            q_flat = quantum_np.flatten()
            
            # Use first min(len) elements for correlation
            min_len = min(len(enc_flat), len(q_flat))
            if min_len > 1:
                corr_matrix = np.corrcoef(enc_flat[:min_len], q_flat[:min_len])
                linearity_corr = abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0
            else:
                linearity_corr = 0
            
            # Check if quantum output variance is similar to scaled encoder variance
            enc_var = np.var(enc_flat)
            q_var = np.var(q_flat)
            variance_ratio = q_var / (enc_var + 1e-10)
            
            # ===== BYPASS TESTS (simplified - no weight perturbation every frame) =====
            quantum_size = quantum_out.shape[-1]
            
            # BYPASS 1: Replace quantum output with zeros
            zero_quantum = torch.zeros(1, quantum_size)
            bypass_zero_logits = self.policy_net.postprocessing(zero_quantum)
            bypass_zero_probs = torch.softmax(bypass_zero_logits, dim=-1).squeeze().numpy()
            bypass_zero_action = int(np.argmax(bypass_zero_probs))
            
            # BYPASS 2: Replace quantum with scaled encoder (to test if postproc reads encoder signal)
            if len(enc_flat) >= quantum_size:
                bypass_encoder = torch.tensor(enc_flat[:quantum_size], dtype=torch.float32).unsqueeze(0)
            else:
                bypass_encoder = torch.zeros(1, quantum_size)
                bypass_encoder[0, :len(enc_flat)] = torch.tensor(enc_flat)
            # Scale to match quantum output range
            bypass_encoder = bypass_encoder * (quantum_np.std() / (enc_flat[:quantum_size].std() + 1e-10))
            bypass_encoder_logits = self.policy_net.postprocessing(bypass_encoder)
            bypass_encoder_probs = torch.softmax(bypass_encoder_logits, dim=-1).squeeze().numpy()
            bypass_encoder_action = int(np.argmax(bypass_encoder_probs))
            
            # Check if encoder-bypass gives SAME action as normal (encoder controls decision)
            encoder_controls = (bypass_encoder_action == normal_action)
            
            # ===== DETERMINE CONTRIBUTION TYPE =====
            is_linear_passthrough = linearity_corr > 0.7
            zeros_change_action = bypass_zero_action != normal_action
            
            if is_linear_passthrough and encoder_controls:
                contribution_type = "ENCODER→Q→POST (linear)"
                quantum_role = "Wire/Passthrough"
            elif is_linear_passthrough and not encoder_controls:
                contribution_type = "Q adds noise"
                quantum_role = "Distortion"
            elif zeros_change_action:
                contribution_type = "Q AFFECTS DECISION"
                quantum_role = "Active"
            else:
                contribution_type = "POST ignores Q"
                quantum_role = "Ignored"
            
            self.quantum_contribution = {
                'normal_action': normal_action,
                'normal_probs': normal_probs,
                'linearity': {
                    'correlation': linearity_corr,
                    'variance_ratio': variance_ratio,
                    'is_linear': is_linear_passthrough
                },
                'bypass_zero': {
                    'action': bypass_zero_action,
                    'probs': bypass_zero_probs,
                    'action_changed': zeros_change_action
                },
                'bypass_encoder': {
                    'action': bypass_encoder_action,
                    'probs': bypass_encoder_probs,
                    'encoder_controls': encoder_controls
                },
                'contribution_type': contribution_type,
                'quantum_role': quantum_role,
                'quantum_matters': zeros_change_action and not is_linear_passthrough
            }
            
            return self.quantum_contribution
    
    def draw_quantum_contribution_panel(self, contribution: dict):
        """Draw panel showing quantum contribution analysis."""
        panel_x = 1080
        panel_y = 450
        panel_w = 310
        panel_h = 400
        
        # Panel background
        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x, panel_y, panel_w, panel_h), border_radius=5)
        pygame.draw.rect(self.screen, GOLD, (panel_x, panel_y, panel_w, panel_h), 2, border_radius=5)
        
        # Title
        title = self.font_medium.render("QUANTUM ANALYSIS", True, GOLD)
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        y = panel_y + 45
        action_names = ['STICK', 'HIT']
        
        # Current decision
        normal_text = f"Decision: {action_names[contribution['normal_action']]}"
        self.screen.blit(self.font_small.render(normal_text, True, WHITE), (panel_x + 10, y))
        y += 30
        
        # === LINEARITY SECTION ===
        linearity = contribution['linearity']
        lin_title = self.font_small.render("LINEARITY CHECK:", True, CYAN)
        self.screen.blit(lin_title, (panel_x + 10, y))
        y += 22
        
        # Correlation bar
        corr = linearity['correlation']
        corr_text = f"Enc\u2192Q correlation: {corr:.2f}"
        corr_color = RED if corr > 0.7 else (ORANGE if corr > 0.4 else GREEN)
        self.screen.blit(self.font_tiny.render(corr_text, True, corr_color), (panel_x + 15, y))
        y += 18
        
        # Draw correlation bar
        bar_w = 180
        bar_h = 12
        pygame.draw.rect(self.screen, (40, 40, 40), (panel_x + 15, y, bar_w, bar_h))
        fill_w = int(bar_w * min(corr, 1.0))
        pygame.draw.rect(self.screen, corr_color, (panel_x + 15, y, fill_w, bar_h))
        # Threshold line at 0.7
        thresh_x = panel_x + 15 + int(bar_w * 0.7)
        pygame.draw.line(self.screen, WHITE, (thresh_x, y), (thresh_x, y + bar_h), 2)
        y += 20
        
        # Linear passthrough verdict
        if linearity['is_linear']:
            lin_verdict = "Q is LINEAR passthrough!"
            lin_color = RED
        else:
            lin_verdict = "Q is non-linear transform"
            lin_color = GREEN
        self.screen.blit(self.font_tiny.render(lin_verdict, True, lin_color), (panel_x + 15, y))
        y += 28
        
        # === BYPASS SECTION ===
        bypass_title = self.font_small.render("BYPASS TESTS:", True, CYAN)
        self.screen.blit(bypass_title, (panel_x + 10, y))
        y += 22
        
        # Zero bypass
        bz = contribution['bypass_zero']
        bz_action = action_names[bz['action']]
        bz_changed = bz['action_changed']
        bz_color = GREEN if bz_changed else GRAY
        bz_indicator = "<- Q matters!" if bz_changed else "(same)"
        self.screen.blit(self.font_tiny.render(f"Q=zeros: {bz_action} {bz_indicator}", True, bz_color), (panel_x + 15, y))
        y += 20
        
        # Encoder bypass
        be = contribution['bypass_encoder']
        be_action = action_names[be['action']]
        be_controls = be['encoder_controls']
        be_color = RED if be_controls else GREEN
        be_indicator = "<- Enc controls!" if be_controls else "(different)"
        self.screen.blit(self.font_tiny.render(f"Q=encoder: {be_action} {be_indicator}", True, be_color), (panel_x + 15, y))
        y += 28
        
        # === VERDICT ===
        verdict_title = self.font_small.render("VERDICT:", True, WHITE)
        self.screen.blit(verdict_title, (panel_x + 10, y))
        y += 25
        
        # Main contribution type
        contrib_type = contribution['contribution_type']
        role = contribution['quantum_role']
        
        if contribution['quantum_matters']:
            verdict_color = GREEN
        elif linearity['is_linear']:
            verdict_color = RED
        else:
            verdict_color = ORANGE
        
        self.screen.blit(self.font_medium.render(contrib_type, True, verdict_color), (panel_x + 10, y))
        y += 28
        
        role_text = f"Quantum role: {role}"
        self.screen.blit(self.font_small.render(role_text, True, verdict_color), (panel_x + 10, y))
        y += 30
        
        # Explanation
        explanations = []
        if linearity['is_linear'] and be_controls:
            explanations.append("* Encoder signal passes through Q")
            explanations.append("* Postproc reads encoder via Q")
            explanations.append("* Q circuit is redundant")
        elif not contribution['bypass_zero']['action_changed']:
            explanations.append("* Zeroing Q doesn't change action")
            explanations.append("* Postproc ignores Q values")
        elif contribution['quantum_matters']:
            explanations.append("* Q output affects decision")
            explanations.append("* Non-linear transformation")
        
        for exp in explanations[:3]:
            self.screen.blit(self.font_tiny.render(exp, True, LIGHT_GRAY), (panel_x + 10, y))
            y += 16

    def run(self):
        """Main simulator loop."""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    else:
                        self.handle_key(event.key)
            
            # Clear screen
            self.screen.fill(BLACK)
            
            # Draw UI
            self._draw_state_controls()
            
            if self.is_hybrid:
                self._draw_hybrid_pipeline()
                # Analyze and draw quantum contribution
                state = (self.player_sum, self.dealer_card, self.usable_ace)
                contribution = self.analyze_quantum_contribution(state)
                if contribution:
                    self.draw_quantum_contribution_panel(contribution)
            else:
                self._draw_decision_panel()
                self._draw_classical_info()
            
            self._draw_instructions()
            
            pygame.display.flip()
            self.clock.tick(30)
        
        pygame.quit()


def load_model(checkpoint_path: str, network_type: str = None):
    """Load a trained model from checkpoint.
    
    Uses network loader to extract config from checkpoint.
    network_type is ignored - type is determined from checkpoint.
    """
    from blackjack_experiment.networks.loader import load_policy_network, NetworkLoadError
    
    try:
        policy_net = load_policy_network(checkpoint_path, strict=False)
    except NetworkLoadError as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise
    
    return policy_net


def find_checkpoints(results_dir: str = 'results') -> list:
    """Find all checkpoint files in results directory.
    
    Returns:
        List of tuples: (relative_path, episode, win_rate, timestamp)
    """
    from pathlib import Path
    import re
    
    checkpoints = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return checkpoints
    
    # Find all .pth files (excluding Pre-Claude folder)
    for pth_file in results_path.rglob('*.pth'):
        # Skip files in Pre-Claude directory
        if 'Pre-Claude' in pth_file.parts:
            continue
        
        relative = pth_file.relative_to(results_path)
        
        # Extract episode and win rate from filename
        # Matches patterns like: checkpoint_ep1000_wr38.0.pth or final_wr38.0_reward-0.20.pth
        # Use word boundary to avoid matching the .pth extension
        match = re.search(r'ep(\d+)_wr([\d.]+?)\.pth', pth_file.name)
        if match:
            episode = int(match.group(1))
            win_rate = float(match.group(2))
        else:
            # Try to match final_wr pattern
            match = re.search(r'wr([\d.]+?)(?:_|\.pth)', pth_file.name)
            if match:
                episode = 999999  # Sort final checkpoints last within experiment
                win_rate = float(match.group(1))
            else:
                episode = 0
                win_rate = 0.0
        
        # Get experiment timestamp/name from parent dir
        # Look for timestamp pattern (YYYYMMDD_HHMMSS) or use directory name
        timestamp = ''
        parent_dir = pth_file.parent.name
        
        # First try to find timestamp in path parts
        for part in pth_file.parts:
            if re.match(r'\d{8}_\d{6}', part):
                timestamp = part
                break
        
        # If no timestamp found, use parent directory as experiment name
        if not timestamp:
            timestamp = parent_dir if parent_dir != 'results' else 'root'
        
        checkpoints.append((str(relative), episode, win_rate, timestamp))
    
    # Sort by timestamp (newest first), then by win rate (highest first)
    checkpoints.sort(key=lambda x: (x[3], -x[2]), reverse=True)
    return checkpoints


def select_checkpoint_gui(results_dir: str = 'results') -> str:
    """GUI file browser for checkpoint selection.
    
    Returns:
        Full path to selected checkpoint or None if cancelled
    """
    import tkinter as tk
    from tkinter import filedialog
    from pathlib import Path
    
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # Set initial directory
    initial_dir = Path(results_dir).absolute()
    if not initial_dir.exists():
        initial_dir = Path.cwd()
    
    # Open file dialog
    checkpoint_path = filedialog.askopenfilename(
        title="Select Model Checkpoint",
        initialdir=str(initial_dir),
        filetypes=[
            ("PyTorch Models", "*.pth"),
            ("All Files", "*.*")
        ]
    )
    
    root.destroy()
    
    if checkpoint_path:
        return str(checkpoint_path)
    return None


def select_checkpoint_interactive(results_dir: str = 'results') -> str:
    """Interactive terminal-based checkpoint selector.
    
    Returns:
        Full path to selected checkpoint
    """
    from pathlib import Path
    
    checkpoints = find_checkpoints(results_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {results_dir}/")
        return None
    
    # Group by experiment (timestamp)
    experiments = {}
    for rel_path, ep, wr, timestamp in checkpoints:
        if timestamp not in experiments:
            experiments[timestamp] = []
        experiments[timestamp].append((rel_path, ep, wr))
    
    print("\n" + "="*70)
    print("CHECKPOINT SELECTOR - Available Experiments")
    print("="*70)
    
    # Display experiments
    exp_list = sorted(experiments.keys(), reverse=True)
    for i, exp_name in enumerate(exp_list, 1):
        checkpoints_in_exp = experiments[exp_name]
        # Get experiment type from first checkpoint path
        first_path = checkpoints_in_exp[0][0]
        exp_type = 'unknown'
        if 'hybrid' in first_path.lower():
            exp_type = 'hybrid'
        elif 'classical' in first_path.lower():
            exp_type = 'classical'
        elif 'amplitude' in first_path.lower():
            exp_type = 'amplitude'
        elif 'bypass' in first_path.lower():
            exp_type = 'bypass'
        
        best_wr = max(c[2] for c in checkpoints_in_exp)
        print(f"{i:2}. [{exp_type:10}] {exp_name} ({len(checkpoints_in_exp)} checkpoints, best WR: {best_wr:.1f}%)")
    
    # Select experiment
    while True:
        try:
            choice = input(f"\nSelect experiment (1-{len(exp_list)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            exp_idx = int(choice) - 1
            if 0 <= exp_idx < len(exp_list):
                break
            print(f"Invalid selection. Enter 1-{len(exp_list)}")
        except ValueError:
            print("Invalid input. Enter a number.")
    
    selected_exp = exp_list[exp_idx]
    checkpoints_in_exp = experiments[selected_exp]
    
    print("\n" + "-"*70)
    print(f"Checkpoints in {selected_exp}:")
    print("-"*70)
    
    # Display checkpoints in selected experiment
    for i, (rel_path, ep, wr) in enumerate(checkpoints_in_exp, 1):
        filename = Path(rel_path).name
        print(f"{i:2}. {filename:50} (ep {ep:5}, WR {wr:5.1f}%)")
    
    # Select checkpoint
    while True:
        try:
            choice = input(f"\nSelect checkpoint (1-{len(checkpoints_in_exp)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            ckpt_idx = int(choice) - 1
            if 0 <= ckpt_idx < len(checkpoints_in_exp):
                break
            print(f"Invalid selection. Enter 1-{len(checkpoints_in_exp)}")
        except ValueError:
            print("Invalid input. Enter a number.")
    
    selected_checkpoint = checkpoints_in_exp[ckpt_idx][0]
    full_path = Path(results_dir) / selected_checkpoint
    
    print(f"\nSelected: {full_path}")
    return str(full_path)


def main():
    parser = argparse.ArgumentParser(
        description="Blackjack State Simulator - Interactive visualization of trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - select from available checkpoints
  python -m blackjack_experiment.simulator
  
  # Direct mode - load specific checkpoint
  python -m blackjack_experiment.simulator --checkpoint results/a2c_hybrid/final.pth
  
  # Select from specific experiment directory
  python -m blackjack_experiment.simulator --dir results/comparison_20251205_142648
        """
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (if not provided, GUI file browser will launch)')
    parser.add_argument('--dir', type=str, default='results',
                        help='Results directory to search for checkpoints (default: results)')
    parser.add_argument('--type', type=str, default=None, choices=['classical', 'hybrid'],
                        help='Network type (auto-detected from checkpoint)')
    parser.add_argument('--cli', action='store_true',
                        help='Use terminal-based selector instead of GUI file browser')
    
    args = parser.parse_args()
    
    # If no checkpoint provided, launch selector (GUI by default, CLI if --cli flag used)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if args.cli:
            # Use terminal-based selector
            checkpoint_path = select_checkpoint_interactive(args.dir)
        else:
            # Use GUI file browser (default)
            print("Opening file browser...")
            checkpoint_path = select_checkpoint_gui(args.dir)
        
        if checkpoint_path is None:
            print("No checkpoint selected. Exiting.")
            return
    
    print(f"\nLoading model from {checkpoint_path}...")
    policy_net = load_model(checkpoint_path, args.type)
    
    # Determine type from loaded network
    network_type = 'hybrid' if hasattr(policy_net, 'quantum_circuit') else 'classical'
    print(f"Network type: {network_type}")
    if network_type == 'hybrid':
        print(f"  n_qubits: {policy_net.n_qubits}, n_layers: {policy_net.n_layers}")
        if hasattr(policy_net, 'entanglement_strategy'):
            print(f"  entanglement: {policy_net.entanglement_strategy}")
        if hasattr(policy_net, 'learnable_input_scaling'):
            print(f"  input scaling: {policy_net.learnable_input_scaling}")
    
    print("\nStarting simulator...")
    print("Controls:")
    print("  +/- buttons: Adjust player sum and dealer card")
    print("  Toggle: Switch usable ace on/off")
    print("  Quick States: Load preset interesting states")
    print("  Q key: Quit")
    print()
    
    simulator = BlackjackSimulator(policy_net, network_type)
    simulator.run()


if __name__ == '__main__':
    main()
