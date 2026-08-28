Please fix a critical autograd NaN hazard in the `MeldLoss` class forward pass. 

Currently, `energy_expended` is calculated using
`torch.linalg.vector_norm(delta_y_flat, ord=2, dim=1, keepdim=True)`. In
PyTorch, the derivative of the L2 norm evaluates to x / ||x||. If the model
perfectly predicts the next state (`delta_y_flat` is exactly zero), the backward
pass will encounter a 0/0 division, immediately throwing a NaN gradient and
crashing the training loop.

Please replace the `torch.linalg.vector_norm` call with a manual Euclidean norm
calculation that includes a small epsilon for autograd stability. Use this exact
replacement: `torch.sqrt(torch.sum(delta_y_flat ** 2, dim=1, keepdim=True) +
1e-8)`
