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
            if wandb.run is not None:
                wandb.log({"train_loss": avg_loss, "epoch": epoch, "epoch_time": epoch_time})
            
        return avg_loss

    def fit(self, dataloader, epochs, use_wandb=False):
        loss_history = []
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(dataloader, epoch, use_wandb)
            loss_history.append(loss)
        return loss_history
