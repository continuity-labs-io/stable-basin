# Proposal: Hardware-Efficient Discretization via Taylor Series Approximation

## Background: The Transcendental Compute Bottleneck
Right now, our State-Space Models (`masr_mamba`, `masr_ssm`) are dramatically slower than the Transformer. This is slowing down the density sweep experiment in particualr. The primary bottleneck is the continuous-time **Zero-Order Hold (ZOH)** discretization step that runs at every single timestep:

```python
A_bar = torch.exp(A * dt)
B_bar = torch.expm1(A * dt) / A * B
```

The issue is that `torch.exp` and `torch.expm1` are **transcendental functions**. On modern GPU architectures (like NVIDIA A100s or H100s), hardware is optimized for Fused Multiply-Add (FMA) instructions (like matrix multiplications). Transcendental operations have significantly higher clock-cycle latency, lower throughput, and consume a massive portion of the SM (Streaming Multiprocessor) resources, especially when executed billions of times across the sequence.

## The Solution: Low-Order Taylor Series Expansion
Because the time-step $\Delta t$ is typically very small in high-frequency sensory telemetry (and $A$ is strictly bounded to be negative for stability), the value $x = A \Delta t$ is extremely close to zero. 

This allows us to completely bypass the transcendental `exp` instructions by substituting them with a low-order **Taylor Series Approximation**.

### Mathematical Derivation
The Taylor series for $e^x$ around 0 is:
$$ e^x \approx 1 + x + \frac{x^2}{2} + \frac{x^3}{6} $$

Therefore, we can approximate the `expm1(x)` term ($e^x - 1$) as:
$$ \text{expm1}(x) \approx x + \frac{x^2}{2} + \frac{x^3}{6} $$

When we plug this into the $B\_bar$ discretization equation:
$$ \bar{B} = \frac{\text{expm1}(A \Delta t)}{A} \cdot B \approx \frac{A \Delta t + \frac{(A \Delta t)^2}{2} + \frac{(A \Delta t)^3}{6}}{A} \cdot B $$

The $A$ cancels out perfectly, eliminating the dangerous division by $A$ and leaving us with:
$$ \bar{B} \approx \left( \Delta t + \frac{A \Delta t^2}{2} + \frac{A^2 \Delta t^3}{6} \right) \cdot B $$

### Performance Impact
By implementing this:
1. **Transcendental Elimination:** We remove `torch.exp` and `torch.expm1` entirely from the integration loop.
2. **Division Elimination:** We eliminate the division by $A$, bypassing the need for `A_safe` clamping entirely (removing another mathematical hazard).
3. **FMA Optimization:** The entire discretization step is reduced to 2 or 3 standard multiplication/addition operations, which map perfectly onto GPU tensor cores.

This optimization is foundational to pushing continuous-time State-Space Models to compete with Transformers in raw wall-clock training time. 

## Open Questions
- Do you want to implement a 2nd-order or 3rd-order Taylor approximation? (2nd order is faster, 3rd order is more numerically stable at larger $\Delta t$ steps).
- Would you like me to implement this approximation in just the `masr_mamba` module, or across all the SSM architectures to bring the overall sweep time down?
