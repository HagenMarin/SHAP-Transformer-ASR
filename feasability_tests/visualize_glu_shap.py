import torch
import numpy as np
import matplotlib.pyplot as plt
from custom_shap_handlers import glu
import shap
from shap.explainers._deep.deep_pytorch import nonlinear_1d

def create_test_data():
    """Create test data for GLU visualization"""
    # Create input tensor with batch size 2 (one for input, one for reference)
    x = torch.randn(2, 160, 32, requires_grad=True)  # [batch, features, time]
    y = torch.randn(2, 80, 32)  # Output is half the feature size
    
    # Create a simple module to hold our tensors
    class TestModule:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    module = TestModule(x, y)
    return module, x, y

def compute_raw_gradients(module, x, y):
    """Compute raw gradients for GLU operation"""
    # Split input into gate and value
    x1, x2 = torch.chunk(x, 2, dim=1)
    
    # Compute GLU operation
    sigmoid_x1 = torch.sigmoid(x1)
    output = x2 * sigmoid_x1
    
    # Compute gradients
    grad_output = torch.ones_like(output)
    grad_x2 = grad_output * sigmoid_x1
    grad_x1 = grad_output * x2 * sigmoid_x1 * (1 - sigmoid_x1)
    
    # Combine gradients
    raw_grad = torch.cat([grad_x2, grad_x1], dim=1)
    return raw_grad

def compute_shap_values(module, x, y):
    """Compute SHAP values using our GLU handler"""
    # Create dummy grad_input and grad_output
    grad_input = [torch.ones_like(x)]
    grad_output = [torch.ones_like(y)]
    
    # Compute SHAP values using our handler
    shap_grads = glu(module, grad_input, grad_output)
    return shap_grads[0]

def plot_comparison(raw_grad, shap_grad, save_path="glu_gradient_comparison.png"):
    """Plot comparison between raw gradients and SHAP values"""
    plt.figure(figsize=(15, 10))
    
    # Plot raw gradients
    plt.subplot(2, 2, 1)
    # Take mean across batch dimension
    plt.imshow(raw_grad[0].mean(dim=0).detach().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Raw Gradients (Gate Part)')
    plt.xlabel('Time')
    plt.ylabel('Features')
    
    plt.subplot(2, 2, 2)
    plt.imshow(raw_grad[1].mean(dim=0).detach().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Raw Gradients (Value Part)')
    plt.xlabel('Time')
    plt.ylabel('Features')
    
    # Plot SHAP values
    plt.subplot(2, 2, 3)
    plt.imshow(shap_grad[0].mean(dim=0).detach().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('SHAP Values (Gate Part)')
    plt.xlabel('Time')
    plt.ylabel('Features')
    
    plt.subplot(2, 2, 4)
    plt.imshow(shap_grad[1].mean(dim=0).detach().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('SHAP Values (Value Part)')
    plt.xlabel('Time')
    plt.ylabel('Features')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_difference(raw_grad, shap_grad, save_path="glu_gradient_difference.png"):
    """Plot the difference between raw gradients and SHAP values"""
    plt.figure(figsize=(15, 10))
    
    # Compute differences (mean across batch dimension)
    diff_gate = raw_grad[0].mean(dim=0) - shap_grad[0].mean(dim=0)
    diff_value = raw_grad[1].mean(dim=0) - shap_grad[1].mean(dim=0)
    
    # Plot differences
    plt.subplot(2, 1, 1)
    plt.imshow(diff_gate.detach().numpy(), aspect='auto', cmap='RdBu')
    plt.colorbar()
    plt.title('Difference in Gate Part (Raw - SHAP)')
    plt.xlabel('Time')
    plt.ylabel('Features')
    
    plt.subplot(2, 1, 2)
    plt.imshow(diff_value.detach().numpy(), aspect='auto', cmap='RdBu')
    plt.colorbar()
    plt.title('Difference in Value Part (Raw - SHAP)')
    plt.xlabel('Time')
    plt.ylabel('Features')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_statistics(raw_grad, shap_grad, save_path="glu_gradient_statistics.png"):
    """Plot statistical comparison between raw gradients and SHAP values"""
    plt.figure(figsize=(15, 10))
    
    # Compute statistics (mean across batch dimension)
    raw_gate_stats = {
        'mean': raw_grad[0].mean().item(),
        'std': raw_grad[0].std().item(),
        'max': raw_grad[0].max().item(),
        'min': raw_grad[0].min().item()
    }
    
    raw_value_stats = {
        'mean': raw_grad[1].mean().item(),
        'std': raw_grad[1].std().item(),
        'max': raw_grad[1].max().item(),
        'min': raw_grad[1].min().item()
    }
    
    shap_gate_stats = {
        'mean': shap_grad[0].mean().item(),
        'std': shap_grad[0].std().item(),
        'max': shap_grad[0].max().item(),
        'min': shap_grad[0].min().item()
    }
    
    shap_value_stats = {
        'mean': shap_grad[1].mean().item(),
        'std': shap_grad[1].std().item(),
        'max': shap_grad[1].max().item(),
        'min': shap_grad[1].min().item()
    }
    
    # Plot statistics
    stats = ['mean', 'std', 'max', 'min']
    x = np.arange(len(stats))
    width = 0.35
    
    plt.subplot(2, 1, 1)
    plt.bar(x - width/2, [raw_gate_stats[s] for s in stats], width, label='Raw Gradients')
    plt.bar(x + width/2, [shap_gate_stats[s] for s in stats], width, label='SHAP Values')
    plt.title('Gate Part Statistics')
    plt.xticks(x, stats)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.bar(x - width/2, [raw_value_stats[s] for s in stats], width, label='Raw Gradients')
    plt.bar(x + width/2, [shap_value_stats[s] for s in stats], width, label='SHAP Values')
    plt.title('Value Part Statistics')
    plt.xticks(x, stats)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Create test data
    module, x, y = create_test_data()
    
    # Compute gradients
    raw_grad = compute_raw_gradients(module, x, y)
    shap_grad = compute_shap_values(module, x, y)
    
    # Split SHAP gradients into gate and value parts
    shap_gate, shap_value = torch.chunk(shap_grad, 2, dim=1)
    raw_gate, raw_value = torch.chunk(raw_grad, 2, dim=1)
    
    # Create visualizations
    plot_comparison(
        [raw_gate, raw_value],
        [shap_gate, shap_value],
        "glu_gradient_comparison.png"
    )
    
    plot_difference(
        [raw_gate, raw_value],
        [shap_gate, shap_value],
        "glu_gradient_difference.png"
    )
    
    plot_statistics(
        [raw_gate, raw_value],
        [shap_gate, shap_value],
        "glu_gradient_statistics.png"
    )
    
    print("Visualizations have been saved as:")
    print("- glu_gradient_comparison.png")
    print("- glu_gradient_difference.png")
    print("- glu_gradient_statistics.png")

if __name__ == "__main__":
    main() 