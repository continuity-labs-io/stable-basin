import torch
import torch.nn as nn

def register_nan_tripwire(model: nn.Module) -> list:
    """
    Attaches a forward hook to every submodule in the model to detect NaNs or Infs 
    in their outputs immediately after they are computed.
    
    This acts as a 'tripwire' during the forward pass, instantly raising an error 
    at the exact layer that generated the singularity, rather than waiting for 
    it to propagate and crash the loss or backward pass later.
    
    Args:
        model: The PyTorch neural network module to monitor.
        
    Returns:
        A list of hook handles. These can be used to remove the hooks later if needed 
        (e.g., `for handle in handles: handle.remove()`).
    """
    handles = []
    
    def check_tensor_health_hook(module, input, output):
        # Gracefully handle single tensor outputs by wrapping them in a tuple
        if isinstance(output, torch.Tensor):
            outputs = (output,)
        elif isinstance(output, tuple):
            outputs = output
        elif isinstance(output, list):
            outputs = tuple(output)
        elif isinstance(output, dict):
            outputs = tuple(output.values())
        else:
            # If the output is some other type (e.g. an object), we skip it
            return
            
        for idx, t in enumerate(outputs):
            if isinstance(t, torch.Tensor):
                # We only check floating point and complex tensors for NaNs/Infs
                if t.is_floating_point() or t.is_complex():
                    if torch.isnan(t).any():
                        raise RuntimeError(f"NaN detected in output {idx} of module: {module.__class__.__name__}")
                    if torch.isinf(t).any():
                        raise RuntimeError(f"Inf detected in output {idx} of module: {module.__class__.__name__}")

    for name, module in model.named_modules():
        # Attach the hook to every single sub-layer
        handle = module.register_forward_hook(check_tensor_health_hook)
        handles.append(handle)
        
    return handles
