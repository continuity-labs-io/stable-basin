We are executing Phase 1: Model Consolidation & Registry Freeze.

We need to formalize the Universal Wrapper (SensorFusionPredictor) so it is the
absolute source of truth, and write a smoke_test.py to mathematically prove that
every architecture in our registry adheres to the exact same API and can train
without crashing.

Please execute the following steps:

1. Update src/harness/sensor_fusion_predictor.py We want to enforce a strict
   contract. Modify the forward method so it always returns (preds,
   hidden_states). Remove the return_hidden argument entirely.

Update the forward signature to: def forward(self, x_raw: torch.Tensor, mask:
Optional[torch.Tensor] = None):

Update the return statement at the end of forward to: return preds,
hidden_states

In get_hidden_states, update it to:

```Python
def get_hidden_states(self, x, mask=None):
    """Convenience method for cleanly extracting the thermodynamic manifold."""
    _, hidden_states = self.forward(x, mask=mask)
    return hidden_states
```

2. Create the Smoke Test (src/harness/smoke_test.py) Create a new script that
   acts as our CI/CD safety net. It will iterate through every model in our
   frozen registry, initialize it via SensorFusionPredictor, pass a synthetic
   multi-modal tensor with sparsity masks through it, and verify that the
   forward and backward passes execute cleanly without shape errors or NaNs.

```Python
import torch
import torch.nn.functional as F
import logging
from src.harness.sensor_fusion_predictor import SensorFusionPredictor, SSMType

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def run_smoke_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Running smoke tests on: {device}")

    # The Frozen Registry
    models_to_test = [
        SSMType.BASELINE,
        SSMType.FORWARD_FILL,
        SSMType.MASK_CONCAT,
        SSMType.TRANSFORMER,
        SSMType.MASK_AWARE,
        SSMType.MASK_AWARE_MAMBA,
        SSMType.GRU_D,
        SSMType.ODE_RNN
    ]

    batch_size = 2
    seq_len = 100
    modality_dims = [20, 10]
    out_dim = 1
    d_model = 64

    d_sensor_total = sum(modality_dims)

    # Synthetic Data
    x_raw = torch.randn(batch_size, seq_len, d_sensor_total, device=device)
    # 20% sparsity dropout
    mask = (torch.rand(batch_size, seq_len, len(modality_dims), device=device) > 0.2).float()
    y_true = torch.randn(batch_size, seq_len, out_dim, device=device)

    all_passed = True

    for ssm_type in models_to_test:
        try:
            logger.info(f"Testing {ssm_type.value}...")

            # 1. Initialization
            model = SensorFusionPredictor(
                ssm_type=ssm_type,
                modality_dims=modality_dims,
                d_model=d_model,
                out_dim=out_dim
            ).to(device)

            # 2. Forward Pass
            preds, hidden = model(x_raw, mask)

            assert preds.shape == (batch_size, seq_len, out_dim), f"Bad preds shape: {preds.shape}"
            assert hidden.shape == (batch_size, seq_len, d_model), f"Bad hidden shape: {hidden.shape}"
            assert not torch.isnan(preds).any(), "NaNs in predictions"
            assert not torch.isnan(hidden).any(), "NaNs in hidden states"

            # 3. Backward Pass
            loss = F.mse_loss(preds, y_true)
            loss.backward()

            # Check gradients
            has_grads = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        raise ValueError("NaNs in gradients")
                    has_grads = True

            assert has_grads, "No gradients computed"

            logger.info(f"  [PASS] {ssm_type.value} executed flawlessly.")

        except Exception as e:
            logger.error(f"  [FAIL] {ssm_type.value} threw an error: {e}")
            all_passed = False

    if all_passed:
        logger.info("ALL MODELS PASSED SMOKE TEST. The registry is frozen and stable.")
    else:
        logger.error("SOME MODELS FAILED. Do not proceed to Phase 2 until fixed.")

if __name__ == "__main__":
    run_smoke_test()
```

3. Run the Smoke Test Execute python -m src.harness.smoke_test to verify the
   registry is solid.
