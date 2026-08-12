import torch

class AttributionEngine:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        self._strategy = None
        
    def set_strategy(self, strategy_func):
        """
        Override the default attribution strategy.
        strategy_func signature: (model, x, target_time_step) -> torch.Tensor
        """
        self._strategy = strategy_func
        
    def reset_strategy(self):
        self._strategy = None
        
    def compute_attribution(self, model, x, target_time_step):
        if self._strategy is not None:
            return self._strategy(model, x, target_time_step)
            
        # Default: First-Order Taylor Decomposition
        x_req = x.clone().detach().requires_grad_(True)
        out = model(x_req)
        
        # Handle models that return tuples (like MeldEngine returning pred_t_plus_1, reconstructed_t, etc)
        predicted_state = out[0] if isinstance(out, tuple) else out
        
        target_state_sum = predicted_state[:, target_time_step, :].sum()
        gradients = torch.autograd.grad(target_state_sum, x_req, retain_graph=True)[0]
        
        return x_req.detach() * gradients
