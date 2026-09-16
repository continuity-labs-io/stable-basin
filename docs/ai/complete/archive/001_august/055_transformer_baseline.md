# Role Instruction

You are an expert PyTorch ML Engineer. We are finalizing our synthetic
benchmarking suite by adding a strictly Causal Transformer baseline. We must
prove how standard Attention mechanisms handle highly sparse, zero-padded
multimodal data compared to our SSMs. Ensure strict modularity.

# Task 1: The Transformer Baseline Module

Create the directory `src/models/attention/` if it does not exist, and create a
file `src/models/attention/baseline_transformer.py`. Implement a PyTorch
`nn.Module` named `BaselineTransformer`.

- **Imports:** `import torch`, `import torch.nn as nn`
- **Init Args:** `d_model` (int, default=64), `nhead` (int, default=4),
  `num_layers` (int, default=2), `max_len` (int, default=2000).
- **Layers:**
  - `self.d_model = d_model`
  - `self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)`
    (Learnable positional encoding).
  - `encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*4)`
  - `self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)`
- **Forward Pass:** `forward(self, latent_x)`
  - `batch, seq_len, d_model = latent_x.size()`
  - Add positional embeddings:
    `x = latent_x + self.pos_embedding[:, :seq_len, :]`
  - **CRITICAL CAUSALITY:** Create a causal mask to prevent looking into the
    future.
    `causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(latent_x.device)`
  - Pass through transformer:
    `out = self.transformer(x, mask=causal_mask, is_causal=True)`
  - Return `out`.

# Task 2: Update the Predictor Module

Modify `src/models/simulators/waddington_predictor.py`.

- **Imports:** Add
  `from src.models.attention.baseline_transformer import BaselineTransformer`.
- **Init:**
  - Update any docstrings to note `ssm_type` can now be `'baseline'`,
    `'mask_aware'`, or `'transformer'`.
  - Add the condition:
    ```python
    elif ssm_type == 'transformer':
        self.ssm = BaselineTransformer(d_model)
    ```
- **Forward Pass:**
  - Route the transformer exactly like the baseline SSM:
    ```python
    elif self.ssm_type == 'transformer':
        h = self.ssm(latent_x)
    ```

# Task 3: Update the Benchmarking Script

Modify `src/experiments/01_train_synthetic_benchmark.py` to include the
Transformer in the training and plotting.

- **Model Initialization:**
  - Add `model_transformer = WaddingtonPredictor("transformer").to(device)`
  - Add
    `optimizer_transformer = optim.AdamW(model_transformer.parameters(), lr=0.005)`
  - Add `transformer_loss_history = []`
- **Training Loop:**
  - Set `model_transformer.train()`.
  - Add the standard zero_grad, forward, loss, backward, step block for
    `model_transformer`.
  - Track `running_loss_transformer` and append its average to
    `transformer_loss_history`.
  - Update the epoch print statement to include
    `| Transformer Loss: {avg_loss_transformer:.3f}`.
- **Evaluation & Plotting:**
  - Set `model_transformer.eval()`.
  - Get
    `test_pred_transformer = model_transformer(test_x_raw, test_mask)[0].cpu().numpy()`
    inside the `torch.no_grad()` block.
  - **Top Subplot:** Add
    `ax1.plot(transformer_loss_history, "g:", linewidth=2, label="Transformer")`.
  - **Bottom Subplot:** Add
    `ax2.plot(test_pred_transformer, "g:", linewidth=2, label="Causal Transformer")`.
  - Adjust `figsize=(10, 10)` to accommodate the slightly larger plot with extra
    legend items cleanly.
