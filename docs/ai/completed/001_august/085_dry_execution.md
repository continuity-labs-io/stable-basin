We are executing Phase 2: Harness Unification & Configuration (DRY Execution).

Our goal is to extract the duplicated PyTorch training logic into a unified StableBasinTrainer, and drive our execution runner using a declarative YAML configuration file instead of argparse.

Please execute the following steps carefully:

1. Install PyYAML
Add pyyaml to requirements.txt (and environment.yml if present).

2. Create the Unified Trainer (src/harness/trainer.py)
Create this file to handle all optimization logic so our runner scripts don't have to reinvent the wheel. It must support both our loss patterns.

```python
import time
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class StableBasinTrainer:
    def __init__(self, model, optimizer, device, loss_type="residual_mse", clip_grad_norm=1.0):
        """
        loss_type: 'direct_mse' (predicts y_true directly) or 'residual_mse' (predicts Delta X)
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_type = loss_type
        self.clip_grad_norm = clip_grad_norm

    def train_epoch(self, dataloader, epoch, use_wandb=False):
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for b_idx, batch in enumerate(dataloader):
            x_raw = batch["x_raw"].to(self.device)
            mask = batch["mask"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # The universal contract from Phase 1
            preds, _ = self.model(x_raw, mask)
            
            if self.loss_type == "direct_mse":
                y_true = batch["y_true"].to(self.device)
                loss = F.mse_loss(preds, y_true)
            elif self.loss_type == "residual_mse":
                # Predict the temporal derivative Delta X
                pred_delta = preds[:, :-1, :] - x_raw[:, :-1, :]
                true_delta = x_raw[:, 1:, :] - x_raw[:, :-1, :]
                loss = F.mse_loss(pred_delta, true_delta)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            loss.backward()
            
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
            self.optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s")
        
        if use_wandb:
            import wandb
            wandb.log({"train_loss": avg_loss, "epoch": epoch, "epoch_time": epoch_time})
            
        return avg_loss

    def fit(self, dataloader, epochs, use_wandb=False):
        loss_history = []
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(dataloader, epoch, use_wandb)
            loss_history.append(loss)
        return loss_history
```

3. Create the YAML Configuration (configs/clinical_diagnostic.yaml)
Create a configs directory at the root of the project, and add clinical_diagnostic.yaml:
```yaml
experiment_name: "Clinical Diagnostic (Pharmacological Shock)"
models:
  - "meld"
  - "baseline"
  - "transformer"
  - "mask_aware"
data:
  condition: "50uM"
  seq_len: 2000
  input_dim: 1024
training:
  epochs: 15
  d_model: 256
  lr: 0.001
  burn_in_frames: 500
evaluation:
  ksm_threshold: 0.85
  png_prefix: "06_clinical_diagnostic_dashboard"
  csv_prefix: "06_clinical_diagnostic_metrics"
```

4. Refactor src/harness/clinical_diagnostic_runner.py
Update the script to use the new architecture:

Replace all argparse flags except --config. Load the YAML using import yaml and config = yaml.safe_load(open(args.config)).

After loading the dataset (the telemetry tensor), encapsulate the model instantiation, training, dynamic crash detection, and diagnostic generation into a helper function: def run_diagnostic_for_model(model_type, config, telemetry, mask, device):.

Inside this function, use the new StableBasinTrainer for the burn-in training. To make it compatible without writing a formal PyTorch Dataset, simply pack your training data into a dummy list: dataloader = [{"x_raw": x_train, "mask": mask_train}] and call trainer.fit(dataloader, epochs).

In main(), iterate over config["models"] to run run_diagnostic_for_model() sequentially. Suffix the output filenames dynamically using the prefixes from the config (e.g., f"{config['evaluation']['png_prefix']}_{model_type}.png").

5. Clean Up the Makefile
Update the clinical-diagnostic target in the Makefile to run:
python -m src.harness.clinical_diagnostic_runner --config configs/clinical_diagnostic.yaml
(You can delete the old clinical-diagnostic-all target, as the YAML configuration inherently handles iterating over all models now).

6. Verification
Run make clinical-diagnostic to ensure the refactored script executes cleanly end-to-end and still produces the expected JSON, CSV, and PNG artifacts for all 4 models.
