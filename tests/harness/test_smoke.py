import pytest
import torch
import torch.nn.functional as F
import logging
from src.harness.sensor_fusion_predictor import SensorFusionPredictor, SSMType

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

@pytest.mark.integration
def test_smoke():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # The Frozen Registry
    models_to_test = [
        SSMType.ZERO_PADDED_SSM,
        SSMType.FORWARD_FILL_SSM,
        SSMType.MASK_CONCAT_SSM,
        SSMType.CAUSAL_TRANSFORMER,
        SSMType.MASR_SSM,
        SSMType.MASR_MAMBA,
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
            logger.info(f"{ssm_type.value}")

            # 1. Initialization
            model = SensorFusionPredictor(
                ssm_type=ssm_type,
                modality_dims=modality_dims,
                d_model=d_model,
                out_dim=out_dim
            ).to(device)

            # 2. Forward Pass
            preds, hidden, reconstructed_t = model(x_raw, mask)

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

            logger.info(f"  Pass")

        except Exception as e:
            logger.error(f"  Fail: {e}")
            all_passed = False

    if all_passed:
        logger.info("Smoke test passed.")
    else:
        logger.error("Smoke test failed.")
        assert all_passed, "Smoke test failed for one or more models."

