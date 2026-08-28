import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.config import settings


class MeldLoss(nn.Module):
    """
    Composite loss function for the state-space training loop.
    Incorporates Next-Frame Forecasting, Lipschitz continuous penalty, and Time-Reversal Error.
    Backcronym: multimodal evaluation of latent dynamics.
    """

    def __init__(
        self,
        alpha=settings.MELD_ALPHA,
        beta=settings.MELD_BETA,
        gamma=settings.MELD_GAMMA,
        L=settings.LIPSCHITZ_CONSTANT,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.L = L

    def forward(self, state_t, target_t_plus_1, pred_t_plus_1, reconstructed_t, delta_x):
        """
        Calculates the composite loss.

        Args:
            state_t: Tensor of shape (batch_size, ...) representing the actual state at time t.
            target_t_plus_1: Tensor of shape (batch_size, ...) representing the ground-truth state at time t+1.
            pred_t_plus_1: Tensor of shape (batch_size, ...) representing the predicted state at time t+1.
            reconstructed_t: Tensor of shape (batch_size, ...) representing the reconstructed state at time t.
            delta_x: Tensor of shape (batch_size, 1) representing the magnitude of the perturbation/time step.

        Returns:
            Tuple containing:
            - L_total: A scalar tensor representing the total weighted loss.
            - metrics: A dictionary of individual detached loss components for telemetry logging.
        """
        # 1. Next-Frame Forecasting (L_forecast)
        l_forecast = F.mse_loss(pred_t_plus_1, target_t_plus_1)

        # 2. Steady-State Flux Penalty (formerly Lipschitz Penalty)
        # Biology is an open thermodynamic system fed by microfluidics.
        # We penalize the AI if it hallucinates a state transition whose required activation 
        # energy exceeds the continuous glucose perfusion rate. 
        # ΔATP_internal = Energy_imported - Energy_expended
        
        # Calculate the predicted state change (activation energy required): Δy = pred_t_plus_1 - state_t
        delta_y = pred_t_plus_1 - state_t

        # Calculate the L2 norm of the predicted state change per sample across all non-batch dimensions.
        # This represents the Energy_expended for the state transition.
        batch_size = delta_y.size(0)
        delta_y_flat = delta_y.view(batch_size, -1)
        energy_expended = torch.sqrt(torch.sum(delta_y_flat ** 2, dim=1, keepdim=True) + 1e-8)  # shape (batch_size, 1)

        # The glucose perfusion rate provides continuous energy flux: Energy_imported = L * delta_x
        # where L is the perfusion rate constant (steady-state flux).
        energy_imported = self.L * delta_x

        # Penalize if required activation energy exceeds the glucose perfusion rate:
        # max(0, Energy_expended - Energy_imported)
        flux_violations = F.relu(energy_expended - energy_imported)

        # Mean across the batch (kept as l_lipschitz for backward compatibility with telemetry)
        l_lipschitz = flux_violations.mean()

        # 3. Time-Reversal Error (L_reverse)
        # Reversible processes produce no net entropy.
        # When a cell undergoes an irreversible phase transition, information is permanently erased,
        # dissipating heat according to Landauer's limit.
        # reverse-prediction head attempts to invert the trajectory back to time t.
        # The magnitude of l_reverse quantifies the thermodynamic irreversibility of the transition.
        l_reverse = F.mse_loss(reconstructed_t, state_t)

        # Total Loss
        l_total = self.alpha * l_forecast + self.beta * l_lipschitz + self.gamma * l_reverse

        # Telemetry Dictionary
        metrics = {
            "forecast_loss": l_forecast.detach().item(),
            "lipschitz_penalty": l_lipschitz.detach().item(),
            "reverse_loss": l_reverse.detach().item(),
        }

        return l_total, metrics


class TopoContrastiveLoss(nn.Module):
    """
    Contrastive loss to align topological shape of continuous LFP standing waves
    with visual stimuli (stimulus embeddings), similar to CLIP.
    """

    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, lfp_latents, vision_latents):
        # L2-normalize both sets of latent vectors along the feature dimension
        lfp_latents = F.normalize(lfp_latents, p=2, dim=1)
        vision_latents = F.normalize(vision_latents, p=2, dim=1)

        # Calculate cosine similarity matrix
        logits = (lfp_latents @ vision_latents.T) * self.logit_scale.exp()

        # Labels for symmetric Cross-Entropy loss (InfoNCE)
        batch_size = lfp_latents.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        # Loss for LFP predicting image (rows) and image predicting LFP (columns)
        loss_lfp_to_vision = F.cross_entropy(logits, labels)
        loss_vision_to_lfp = F.cross_entropy(logits.T, labels)

        # Average loss
        loss = (loss_lfp_to_vision + loss_vision_to_lfp) / 2

        # Telemetry Dictionary
        metrics = {
            "contrastive_loss": loss.detach().item(),
            "logit_scale": self.logit_scale.detach().item(),
        }

        return loss, metrics
