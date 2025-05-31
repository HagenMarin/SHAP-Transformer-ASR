import torch
import warnings
from shap.explainers._deep.deep_pytorch import op_handler, nonlinear_1d, linear_1d

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
    GLU splits the input in half, applies sigmoid to one half, and multiplies with the other half.
    Use nonlinear_1d since it involves sigmoid and multiplication.
    """
    if not hasattr(module, 'x') or not hasattr(module, 'y'):
        return grad_input
    return nonlinear_1d(module, grad_input, grad_output)

# Add our custom handlers to the op_handler dictionary
op_handler["LayerNorm"] = layernorm
op_handler["SiLU"] = silu
op_handler["GroupNorm"] = groupnorm
op_handler["GLU"] = glu 