# Open vs. Closed Systems (The Steady-State Flux)

**Status:** Completed

## 1. The Conceptual Shift
Biology is an open thermodynamic system fed by microfluidics. It does not drain a static ATP battery; it maintains steady-state flux.

## 2. The Narrative
The loss function does not penalize the cell for spending energy; it penalizes the AI if it hallucinates an electrical/transcriptomic state transition whose required activation energy exceeds the glucose perfusion rate of the bioreactor. $\Delta \text{ATP}_{internal} = \text{Energy}_{imported} - \text{Energy}_{expended}$.

## 3. The Repo Fix (Implementation Plan)
We must update the `Lipschitz_penalty`.

### Action Items
- [x] Update the `Lipschitz_penalty` function to reflect an open thermodynamic system based on the glucose perfusion rate vs required activation energy.

### Targeted Files
- `src/models/losses/meld_loss.py`
