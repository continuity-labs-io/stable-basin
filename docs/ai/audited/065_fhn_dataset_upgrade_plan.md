# Dataset Upgrade: Coupled Biological Oscillator

The current Waddington dataset is too simple (a step-function phase transition). To truly stress-test our Sensor Fusion architecture and State-Space Models, we should upgrade the data generation to a true multidimensional dynamical system that mathematically requires continuous integration to solve.

## Proposed System: The FitzHugh-Nagumo (FHN) Oscillator

The FHN model is a classic 2D simplification of the Hodgkin-Huxley model, widely used to model biological spiking (e.g., neurons, cardiac cells). 

It consists of two coupled variables:
1. **$v$ (Fast Variable)**: The membrane potential (spiking).
2. **$w$ (Slow Variable)**: The recovery variable.

The system is defined by the ODEs:
$$ \frac{dv}{dt} = v - \frac{v^3}{3} - w + I_{ext} $$
$$ \frac{dw}{dt} = \frac{v + a - b \cdot w}{\tau} $$

**Crucially, we will set the time-scale separation parameter ($\tau$) to $10^4$**. This makes the FHN system a highly *stiff* relaxation oscillator, where the slow variable evolves $10,000\times$ slower than the fast variable. This will severely punish architectures that cannot learn multiple discrete temporal scales simultaneously.

### How this maps to Sensor Fusion

We will generate the sequences using Euler integration and map them to our modalities as follows:

- **Target (`y_true`)**: The fast spiking variable $v(t)$.
- **Modality 0 (20D, Continuous)**: A random projection of the slow variable $w(t)$. This gives the model continuous background information, but *no direct access* to $v$.
- **Modality 1 (10D, Sparse)**: A random projection of $v(t)$, but **masked out 95% of the time**.

### Why this is a brilliant benchmark:
To accurately predict $v(t)$ during the 95% of the time it is masked, the neural network **MUST** learn the underlying ODEs. It has to combine its continuous observation of $w(t)$ (from Modality 0) with its memory of the last seen $v(t)$ (from Modality 1) and continuously integrate the system forward in time. 

This guarantees that:
1. Simple feed-forward networks will fail completely.
2. The model is forced to perform true **Sensor Fusion** (combining $w$ and sparse $v$).
3. It proves the value of state-space memory for biological ODE integration.

## Proposed Code Changes

### 1. `src/data/waddington_dataset.py`
- We will rewrite `SyntheticWaddingtonDataset` to run a loop of Euler integration over the FHN equations to generate $v(t)$ and $w(t)$.
- We will redefine `self.W_0` (1 -> 20) and `self.W_1` (1 -> 10) to act as the projection matrices.
- The `__getitem__` function will yield `modality_0 = w @ W_0 + noise` and `modality_1 = (v @ W_1 + noise) * mask_1`.

## User Feedback Required
Does this FHN biological oscillator setup sound exactly like the kind of continuous integration problem you want to benchmark? If you approve, I will rewrite the dataset class!
