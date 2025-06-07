import torch
import warnings
from shap.explainers._deep.deep_pytorch import op_handler, nonlinear_1d, linear_1d, add_interim_values as original_add_interim_values

# Store the original function
_original_add_interim_values = original_add_interim_values

def add_interim_values(module, input, output):
    """Custom version of add_interim_values that handles our custom handlers.
    First tries the original function, then handles our custom cases.
    """
    # First try the original function
    _original_add_interim_values(module, input, output)
    
    # If the module still doesn't have x and y attributes, check if it's one of our custom handlers
    if not hasattr(module, 'x') or not hasattr(module, 'y'):
        module_type = module.__class__.__name__
        if module_type in op_handler:
            func = op_handler[module_type]
            # If this is one of our custom handlers that uses nonlinear_1d
            if func in [silu, glu]:
                if type(input) is tuple:
                    module.x = torch.nn.Parameter(input[0].detach())
                else:
                    module.x = torch.nn.Parameter(input.detach())
                if type(output) is tuple:
                    module.y = torch.nn.Parameter(output[0].detach())
                else:
                    module.y = torch.nn.Parameter(output.detach())

# Monkey patch the original function
import shap.explainers._deep.deep_pytorch
shap.explainers._deep.deep_pytorch.add_interim_values = add_interim_values

def layernorm(module, grad_input, grad_output):
    """Handler for LayerNorm module.
    LayerNorm normalizes across the last dimension, similar to BatchNorm.
    Use linear_1d since normalization is a linear operation.
    """
    if not hasattr(module, 'x') or not hasattr(module, 'y'):
        return grad_input
    return linear_1d(module, grad_input, grad_output)

def silu(module, grad_input, grad_output):
    """Handler for SiLU (Swish) activation.
    SiLU(x) = x * sigmoid(x), similar to GELU.
    Use nonlinear_1d since it's a nonlinear activation.
    """
    if not hasattr(module, 'x') or not hasattr(module, 'y'):
        return grad_input
    return nonlinear_1d(module, grad_input, grad_output)

def groupnorm(module, grad_input, grad_output):
    """Handler for GroupNorm module.
    GroupNorm is similar to BatchNorm but normalizes across groups.
    Use linear_1d since normalization is a linear operation.
    """
    if not hasattr(module, 'x') or not hasattr(module, 'y'):
        return grad_input
    return linear_1d(module, grad_input, grad_output)

def glu(module, grad_input, grad_output):
    """Handler for GLU (Gated Linear Unit).
    GLU is similar to SiLU but with shape transformation:
    1. Splits input in half along feature dimension
    2. Applies SiLU-like operation (x * sigmoid(x))
    3. Output shape is halved
    """
    #if not hasattr(module, 'x') or not hasattr(module, 'y'):
    delta_x = module.x[:module.x.shape[0]//2] - module.x[module.x.shape[0]//2:]
    delta_y = module.y[:module.y.shape[0]//2] - module.y[module.y.shape[0]//2:]
    dup0 = [2] + [1 for _ in delta_x.shape[1:]]
    dup1 = [1] + [2] + [1 for _ in grad_output[0].shape[2:]]
    grads = [None for _ in grad_input]
    grads[0] = torch.where(
        torch.abs(delta_x.repeat(dup0)) < 1e-6,
        grad_input[0],  # Use original gradient for stable regions
        grad_output[0].repeat(dup1)*5e-6# * x2 * sigmoid_prime_x1
    )
    return tuple(grads)
    
    # Get input and output tensors
    x = module.x
    y = module.y
    
    # Split input into gate and value parts
    x1, x2 = torch.chunk(x, 2, dim=1)  # x1 is gate, x2 is value
    
    # Compute deltas
    # The first half of the batch is the input, second half is the reference
    batch_size = x1.shape[0] // 2
    delta_x1 = x1[:batch_size] - x1[batch_size:]  # Gate deltas
    delta_x2 = x2[:batch_size] - x2[batch_size:]  # Value deltas
    delta_y = y[:batch_size] - y[batch_size:]     # Output deltas
    
    # Create repeat pattern for batch dimension
    dup0 = [2] + [1 for _ in delta_x1.shape[1:]]
    
    # Handle numerical instabilities
    grads = [None for _ in grad_input]
    
    # Compute sigmoid values for the gate
    sigmoid_x1 = torch.sigmoid(x1)
    
    # For x2 (value part), the gradient is sigmoid(x1) * delta_y / delta_x2
    grad_x2 = torch.where(
        torch.abs(delta_x2.repeat(dup0)) < 1e-6,
        grad_input[0][:, x1.shape[1]:],  # Use original gradient for stable regions
        grad_output[0] * (delta_y / delta_x2).repeat(dup0)# * sigmoid_x1
    )
    
    # For x1 (gate part), the gradient is x2 * sigmoid'(x1) * delta_y / delta_x1
    sigmoid_prime_x1 = sigmoid_x1 * (1 - sigmoid_x1)  # Derivative of sigmoid
    grad_x1 = torch.where(
        torch.abs(delta_x1.repeat(dup0)) < 1e-6,
        grad_input[0][:, :x1.shape[1]],  # Use original gradient for stable regions
        grad_output[0] * (delta_y / delta_x1).repeat(dup0)# * x2 * sigmoid_prime_x1
    )
    
    # --- Logging for diagnostics ---
    def log_stats(name, tensor):
        print(f"[GLU DIAG] {name}: min={tensor.min().item():.3f}, max={tensor.max().item():.3f}, mean={tensor.mean().item():.3f}, std={tensor.std().item():.3f}")
    log_stats('delta_x1', delta_x1)
    log_stats('delta_x2', delta_x2)
    log_stats('delta_y', delta_y)
    log_stats('grad_x1', grad_x1)
    log_stats('grad_x2', grad_x2)
    
    # Log locations of extreme values in gradients
    extreme_mask_x1 = (grad_x1 > 100) | (grad_x1 < -100)
    extreme_mask_x2 = (grad_x2 > 100) | (grad_x2 < -100)
    extreme_mask_delta_x1 = (delta_x1 > 1e-6) | (delta_x1 < -1e-6)
    extreme_mask_delta_x2 = (delta_x2 > 1e-6) | (delta_x2 < -1e-6)
    extreme_mask_delta_y = (delta_y > 1e-6) | (delta_y < -1e-6)
    extreme_mask_grad_output = (grad_output[0] > 100) | (grad_output[0] < -100)
    extreme_mask_delta_quotient = ((delta_y / delta_x1) > 1e-6) | ((delta_y / delta_x1) < -1e-6)
    if extreme_mask_x1.any():
        idxs = torch.nonzero(extreme_mask_x1)
        #print(f"[GLU DIAG] Extreme grad_x1 values at indices: {idxs.tolist()}")
        #print(f"[GLU DIAG] grad_x1 values: {grad_x1[extreme_mask_x1].tolist()}")
    if extreme_mask_x2.any():
        idxs = torch.nonzero(extreme_mask_x2)
        #print(f"[GLU DIAG] Extreme grad_x2 values at indices: {idxs.tolist()}")
        #print(f"[GLU DIAG] grad_x2 values: {grad_x2[extreme_mask_x2].tolist()}")
    if extreme_mask_delta_x1.any():
        idxs = torch.nonzero(extreme_mask_delta_x1)
        #print(f"[GLU DIAG] Extreme delta_x1 values at indices: {idxs.tolist()}")
        #print(f"[GLU DIAG] delta_x1 values: {delta_x1[extreme_mask_delta_x1].tolist()}")
    if extreme_mask_delta_x2.any():
        idxs = torch.nonzero(extreme_mask_delta_x2)
        #print(f"[GLU DIAG] Extreme delta_x2 values at indices: {idxs.tolist()}")
        #print(f"[GLU DIAG] delta_x2 values: {delta_x2[extreme_mask_delta_x2].tolist()}")
    if extreme_mask_delta_y.any():
        idxs = torch.nonzero(extreme_mask_delta_y)
        #print(f"[GLU DIAG] Extreme delta_y values at indices: {idxs.tolist()}")
        #print(f"[GLU DIAG] delta_y values: {delta_y[extreme_mask_delta_y].tolist()}")
    if extreme_mask_grad_output.any():
        idxs = torch.nonzero(extreme_mask_grad_output)
        print(f"[GLU DIAG] Extreme grad_output values at indices: {idxs.tolist()}")
        #print(f"[GLU DIAG] grad_output values: {grad_output[0][extreme_mask_grad_output].tolist()}")
    if extreme_mask_delta_quotient.any():
        idxs = torch.nonzero(extreme_mask_delta_quotient)
        print(f"[GLU DIAG] Extreme delta_quotient values at indices: {idxs.tolist()}")
        #print(f"[GLU DIAG] delta_quotient values: {delta_y[extreme_mask_delta_quotient].tolist()}")
    # Optionally, log the corresponding deltas and inputs
    # --- End logging ---
    
    # Combine the gradients
    grads[0] = torch.cat([grad_x2, grad_x1], dim=1)
    
    return tuple(grads)

# Add our custom handlers to the op_handler dictionary
op_handler["LayerNorm"] = layernorm
op_handler["SiLU"] = silu
op_handler["GroupNorm"] = groupnorm
op_handler["GLU"] = glu 
