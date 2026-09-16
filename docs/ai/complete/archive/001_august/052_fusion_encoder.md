# Role Instruction

You are an expert PyTorch ML Engineer and Core Architecture Designer. We are
building the neural network modules for a synthetic biological benchmark,
adhering to a strict separation of concerns based on the project's directory
structure. Ensure strict typing, modularity, and clean documentation.

# Task 1: The Fusion Encoder

Create `src/models/encoders/fusion.py`. Implement a PyTorch `nn.Module` named
`BiologicalCartridgeFusion`.

- **Init Args:** `d_cartridge` (int), `n_modalities` (int), `d_model` (int).
- **Layers:**
  - `W_cart`: `nn.Linear(d_cartridge, d_model, bias=False)` (Projects hardware
    data to dense space).
  - `W_gate`: `nn.Linear(n_modalities, d_model, bias=True)` (Routes the boolean
    mask to the latent subspace).
- **Forward Pass:** `forward(self, x_raw, mask)`
  - `latent_x = self.W_cart(x_raw)`
  - `latent_gate = torch.sigmoid(self.W_gate(mask))`
  - Return `(latent_x, latent_gate)`

# Task 2: The Baseline Control Model

Create `src/models/ssm/baseline_ssm.py`. Implement a `nn.Module` named
`BaselineSSM`. This is a standard continuous-time SSM that ingests data blindly.
It acts as our control group.

- **Init Args:** `d_model` (int).
- **Layers:**
  - `A_log`: `nn.Parameter(torch.log(torch.rand(d_model) * 0.5 + 0.1))`
    (Continuous decay rate).
  - `B_proj`: `nn.Linear(d_model, d_model, bias=False)` (Payload).
  - `dt_proj`: `nn.Linear(d_model, d_model)` (Baseline step size).
- **Forward Pass:** `forward(self, latent_x)`
  - Inputs: `latent_x` shape `[batch, seq_len, d_model]`.
  - Initialize
    `h_prev = torch.zeros(latent_x.size(0), latent_x.size(2), device=latent_x.device)`.
  - Calculate `A = -torch.exp(self.A_log)` (Must be strictly negative).
  - Create an empty list `hidden_states = []`.
  - Loop sequentially through time `t` from 0 to `seq_len - 1`:
    - `x_t = latent_x[:, t, :]`
    - `dt = torch.nn.functional.softplus(self.dt_proj(x_t))`
    - `B = self.B_proj(x_t)`
    - `A_bar = torch.exp(A * dt)`
    - `B_bar = (A_bar - 1.0) / (A - 1e-8) * B` (Epsilon prevents zero division).
    - `h_prev = A_bar * h_prev + B_bar`
    - `hidden_states.append(h_prev)`
  - Stack and return `hidden_states` as a tensor of shape
    `[batch, seq_len, d_model]`.

# Task 3: The Mask-Aware Innovation

Create `src/models/ssm/mask_aware_ssm.py`. Implement a `nn.Module` named
`MaskAwareSSM`. This model natively implements the "Physics Hack", modulating
time based on the latent gate.

- **Init Args:** `d_model` (int).
- **Layers:**
  - Identical to `BaselineSSM` (`A_log`, `B_proj`, `dt_proj`).
- **Forward Pass:** `forward(self, latent_x, latent_gate)`
  - Inputs: `latent_x` and `latent_gate`, both shape
    `[batch, seq_len, d_model]`.
  - Initialize
    `h_prev = torch.zeros(latent_x.size(0), latent_x.size(2), device=latent_x.device)`.
  - Calculate `A = -torch.exp(self.A_log)`.
  - Create an empty list `hidden_states = []`.
  - Loop sequentially through time `t` from 0 to `seq_len - 1`:
    - `x_t = latent_x[:, t, :]`
    - `g_t = latent_gate[:, t, :]`
    - `dt_base = torch.nn.functional.softplus(self.dt_proj(x_t))`
    - **THE PHYSICS HACK:** `dt_gated = dt_base * g_t + 1e-8`
    - `B = self.B_proj(x_t)`
    - `A_bar = torch.exp(A * dt_gated)`
    - `B_bar = (A_bar - 1.0) / (A - 1e-8) * B`
    - `h_prev = A_bar * h_prev + B_bar`
    - `hidden_states.append(h_prev)`
  - Stack and return `hidden_states` as a tensor of shape
    `[batch, seq_len, d_model]`.

# Task 4: Diagnostic Block

At the bottom of `src/models/ssm/mask_aware_ssm.py`, add a diagnostic
`if __name__ == '__main__':` block.

- Adjust `sys.path` dynamically (using `import sys, os`) so we can import
  `BiologicalCartridgeFusion` from `src.models.encoders.fusion` safely.
- Initialize
  `fusion = BiologicalCartridgeFusion(d_cartridge=30, n_modalities=2, d_model=64)`
  and `ssm = MaskAwareSSM(d_model=64)`.
- Generate mock data: `x_raw = torch.randn(2, 100, 30)` and
  `mask = torch.ones(2, 100, 2)`.
- Pass through Fusion: `latent_x, latent_gate = fusion(x_raw, mask)`.
- Pass through SSM: `h = ssm(latent_x, latent_gate)`.
- Print the shape of `h` (should be `[2, 100, 64]`) and a success message
  confirming the forward pass executed without exploding gradients or NaNs.
