import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import networkx as nx

def create_activation_graphs():
    # Create directed graphs for each activation
    graphs = {}
    positions = {}
    
    # GLU Graph
    G_glu = nx.DiGraph()
    pos_glu = {
        'input': (0, 0),
        'x1': (2, 1),    # Gate part
        'x2': (2, -1),   # Value part
        'sigmoid': (4, 1),
        'multiply': (6, 0),
        'output': (8, 0)
    }
    
    G_glu.add_node('input', label='Input\n[batch, features]')
    G_glu.add_node('x1', label='Gate\n[batch, features/2]')
    G_glu.add_node('x2', label='Value\n[batch, features/2]')
    G_glu.add_node('sigmoid', label='Sigmoid\nσ(x1)')
    G_glu.add_node('multiply', label='Multiply\nx2 * σ(x1)')
    G_glu.add_node('output', label='Output\n[batch, features/2]')
    
    G_glu.add_edge('input', 'x1', label='Split')
    G_glu.add_edge('input', 'x2', label='Split')
    G_glu.add_edge('x1', 'sigmoid', label='σ')
    G_glu.add_edge('sigmoid', 'multiply', label='σ(x1)')
    G_glu.add_edge('x2', 'multiply', label='x2')
    G_glu.add_edge('multiply', 'output', label='Result')
    
    # ReLU Graph
    G_relu = nx.DiGraph()
    pos_relu = {
        'input': (0, 0),
        'relu': (4, 0),
        'output': (8, 0)
    }
    
    G_relu.add_node('input', label='Input\n[batch, features]')
    G_relu.add_node('relu', label='ReLU\nmax(0, x)')
    G_relu.add_node('output', label='Output\n[batch, features]')
    
    G_relu.add_edge('input', 'relu', label='x')
    G_relu.add_edge('relu', 'output', label='max(0, x)')
    
    # SiLU Graph
    G_silu = nx.DiGraph()
    pos_silu = {
        'input': (0, 0),
        'sigmoid': (3, 0),
        'multiply': (6, 0),
        'output': (9, 0)
    }
    
    G_silu.add_node('input', label='Input\n[batch, features]')
    G_silu.add_node('sigmoid', label='Sigmoid\nσ(x)')
    G_silu.add_node('multiply', label='Multiply\nx * σ(x)')
    G_silu.add_node('output', label='Output\n[batch, features]')
    
    G_silu.add_edge('input', 'sigmoid', label='x')
    G_silu.add_edge('input', 'multiply', label='x')
    G_silu.add_edge('sigmoid', 'multiply', label='σ(x)')
    G_silu.add_edge('multiply', 'output', label='x * σ(x)')
    
    graphs = {'glu': G_glu, 'relu': G_relu, 'silu': G_silu}
    positions = {'glu': pos_glu, 'relu': pos_relu, 'silu': pos_silu}
    
    return graphs, positions

def plot_activation_graphs():
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    graphs, positions = create_activation_graphs()
    
    for ax, (name, G) in zip(axes, graphs.items()):
        pos = positions[name]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=2000, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                             arrows=True, arrowsize=20, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Add title and remove axis
        ax.set_title(f'{name.upper()} Activation Function\nComputation Graph', pad=20)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('activation_computation_graphs.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_activation_functions():
    # Create input values
    x = np.linspace(-5, 5, 1000)
    
    # Compute activation outputs
    relu_output = np.maximum(0, x)
    sigmoid_x = 1 / (1 + np.exp(-x))
    silu_output = x * sigmoid_x
    glu_output = x * sigmoid_x  # For visualization, using same x for both parts
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot components
    plt.plot(x, x, 'k--', label='Identity', alpha=0.3)
    plt.plot(x, relu_output, 'b-', label='ReLU: max(0, x)', linewidth=2)
    plt.plot(x, silu_output, 'g-', label='SiLU: x * σ(x)', linewidth=2)
    plt.plot(x, glu_output, 'r-', label='GLU: x * σ(x)', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Input Value')
    plt.ylabel('Output Value')
    plt.title('Activation Functions Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add horizontal and vertical lines at zero
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Save the plot
    plt.savefig('activation_functions_plot.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Create both visualizations
    plot_activation_graphs()
    plot_activation_functions()
    print("Visualizations have been saved as 'activation_computation_graphs.png' and 'activation_functions_plot.png'") 