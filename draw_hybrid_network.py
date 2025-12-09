import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pennylane as qml
import sys
import os

# Add current directory to path so we can import the package
sys.path.append(os.getcwd())

from blackjack_experiment.networks.hybrid.policy import UniversalBlackjackHybridPolicyNetwork

try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False
    print("torchviz not available - install with: pip install torchviz")

def draw_architecture(net, filename="network_architecture.png"):
    """
    Draws the network architecture in a classic 'neural network' style:
    Nodes (circles) and Edges (lines).
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Define the structure to visualize
    # We will visualize the flow: Input -> Encoder -> Quantum -> Output
    
    # 1. Input Layer (3 nodes for compact encoding)
    n_input = net.input_dim
    if n_input > 10: n_input = 5 # Truncate for visualization if too large
    
    # 2. Encoder Hidden Layer (Representing the classical processing)
    # We'll use 8 nodes to match the user's preferred aesthetic (3->8 connection)
    n_encoder = 8 
    
    # 3. Quantum Layer (Representing Qubits)
    n_quantum = net.n_qubits
    
    # 4. Output Layer (Actions)
    n_output = net.n_actions
    
    # List of layers: (Name, Number of Nodes)
    layers_config = [
        ("Input State", n_input),
        ("Classical\nEncoder", n_encoder),
        ("Quantum\nCircuit", n_quantum),
        ("Action\nProbs", n_output)
    ]
    
    # Layout settings
    left_margin = 0.1
    right_margin = 0.9
    layer_spacing = (right_margin - left_margin) / (len(layers_config) - 1)
    
    # Vertical centering
    y_center = 0.5
    v_spacing = 0.12  # Vertical spacing between nodes
    
    node_radius = 0.04
    
    # Store coordinates of nodes for each layer
    layer_coords = []
    
    for i, (name, n_nodes) in enumerate(layers_config):
        x = left_margin + i * layer_spacing
        
        # Calculate y positions to center the layer
        layer_height = (n_nodes - 1) * v_spacing
        start_y = y_center - layer_height / 2
        
        current_layer_nodes = []
        for j in range(n_nodes):
            y = start_y + j * v_spacing
            current_layer_nodes.append((x, y))
            
            # Draw Node (Circle)
            # White fill, dark grey outline
            circle = plt.Circle((x, y), node_radius, 
                              facecolor='white', edgecolor='#333333', 
                              linewidth=1.5, zorder=10)
            ax.add_patch(circle)
            
        layer_coords.append(current_layer_nodes)
        
        # Draw Layer Label
        ax.text(x, 0.1, name, ha='center', va='top', 
                fontsize=12, fontweight='bold', color='#333333')

    # Draw Connections (Edges)
    # Fully connected between adjacent layers
    for i in range(len(layer_coords) - 1):
        curr_layer = layer_coords[i]
        next_layer = layer_coords[i+1]
        
        for x1, y1 in curr_layer:
            for x2, y2 in next_layer:
                # Draw line
                ax.plot([x1, x2], [y1, y2], 
                       color='gray', linewidth=0.5, alpha=0.4, zorder=1)

    # Add title
    plt.text(0.5, 0.95, "Hybrid Network Architecture", 
             ha='center', va='center', fontsize=16, fontweight='bold', color='#333333')
             
    # Add parameter count annotation
    total_params = net.get_num_parameters()
    plt.text(0.5, 0.90, f"Total Parameters: {total_params}", 
             ha='center', va='center', fontsize=10, style='italic', color='#666666')

    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved architecture diagram to {filename}")
    plt.close()

def draw_quantum_circuit(net, filename="quantum_circuit.png"):
    print("Drawing quantum circuit...")
    n_qubits = net.n_qubits
    inputs = torch.randn(n_qubits, 2) 
    weights = net.weights
    
    # Use PennyLane's native drawer
    try:
        qnode = net.quantum_circuit
        fig, ax = qml.draw_mpl(qnode, style='pennylane', show_all_wires=True)(inputs, weights)
        plt.title(f"Quantum Circuit ({n_qubits} qubits, {net.n_layers} layers)", 
                  fontsize=14, pad=20)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved quantum circuit diagram to {filename}")
        plt.close()
    except Exception as e:
        print(f"Failed to draw quantum circuit: {e}")
        import traceback
        traceback.print_exc()

def draw_computational_graph(net, filename="hybrid_model_graph"):
    """Draw the computational graph of the hybrid model using torchviz."""
    if not TORCHVIZ_AVAILABLE:
        print("Skipping computational graph - torchviz not installed")
        return
    
    print("Drawing computational graph...")
    try:
        import os
        # Add Graphviz to PATH if it exists in known location
        graphviz_path = r"E:\stuff\Graphviz\bin"
        if os.path.exists(graphviz_path) and graphviz_path not in os.environ.get('PATH', ''):
            os.environ['PATH'] = os.environ.get('PATH', '') + os.pathsep + graphviz_path
        
        # Create a dummy input (batch of 1)
        dummy_state = (15, 10, 0)  # player_sum=15, dealer_card=10, usable_ace=False
        
        # Forward pass to build computation graph
        probs = net(dummy_state)
        
        # Create visualization
        dot = make_dot(probs, params=dict(net.named_parameters()), 
                      show_attrs=False, show_saved=False)
        
        # Improve readability with better layout
        dot.graph_attr.update({
            'rankdir': 'TB',  # Top to bottom
            'size': '20,30',  # Larger canvas
            'dpi': '300',  # Higher resolution
            'ranksep': '0.8',
            'nodesep': '0.5',
            'fontsize': '10',
            'bgcolor': 'white'
        })
        dot.node_attr.update({
            'style': 'filled',
            'fillcolor': 'lightblue',
            'fontsize': '10'
        })
        dot.edge_attr.update({
            'fontsize': '8'
        })
        
        # Try to render directly
        try:
            dot.format = 'png'
            dot.render(filename, cleanup=True)
            print(f"Saved computational graph to {filename}.png")
        except Exception as e:
            # Fallback: create a simple matplotlib visualization of the architecture
            print(f"Graphviz not available, creating simplified diagram...")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            ax.text(0.5, 0.9, "Hybrid Network Computational Graph", 
                   ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
            
            # List the network structure as text
            structure_text = f"""
Network Structure:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Blackjack State (player, dealer, ace)
    ↓
Encoder: {net.encoder_output_dim} neurons
    Parameters: {sum(p.numel() for p in net.feature_encoder.parameters())}
    ↓
Compression: → {net.quantum_input_dim} dimensions
    ↓
Quantum Circuit: {net.n_qubits} qubits, {net.n_layers} layers
    Parameters: {net.weights.numel()} (variational weights)
    ↓
Measurement: {net.quantum_output_dim} expectation values
    ↓
Postprocessing: {net.quantum_output_dim} → {net.n_actions}
    Parameters: {sum(p.numel() for p in net.postprocessing.parameters())}
    ↓
Softmax: Action Probabilities [Stand, Hit]

Total Parameters: {net.get_num_parameters()}
"""
            ax.text(0.5, 0.45, structure_text, ha='center', va='center', 
                   fontsize=10, family='monospace', transform=ax.transAxes)
            
            plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
            print(f"Saved simplified computational graph to {filename}.png")
            plt.close()
        
    except Exception as e:
        print(f"Failed to draw computational graph: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Initializing Hybrid Network...")
    net = UniversalBlackjackHybridPolicyNetwork()
    print(net.get_config_summary())
    
    draw_architecture(net)
    draw_quantum_circuit(net)
    draw_computational_graph(net)
