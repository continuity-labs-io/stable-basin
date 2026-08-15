I am upgrading the MeldLoss class in src/models/losses/meld_loss.py of my Stable Basin biological physics engine. I need to add a 'Mitotic Isochoric (Constant Volume) Constraint' based on in vivo LLSM observations.Assume the 100-D latent optical vector ($\Sigma$) predicted by the model (pred_t_plus_1) can be passed through a frozen linear readout layer self.volume_readout to extract a 1-D 'Volume' scalar. The remaining 99 dimensions represent morphological shape.Write a new custom loss penalty called l_isochoric.

1 It must heavily penalize any change in Volume(pred_t_plus_1) compared to Volume(state_t) (the absolute volume must remain constant, $dV/dt = 0$).

2 It must allow high variance / large L2 norm changes in the 99 shape dimensions without penalty (allowing the cell to 'bleb').

3 Integrate this l_isochoric penalty into the forward pass alongside the existing l_forecast and l_lipschitz, with a new weighting parameter delta=10.0.

4 Ensure it logs to the metrics dictionary for telemetry.
