Context: We are upgrading the biological phase structure dataloader in Project
MELD. Currently, the `align_to_master_clock` method uses a SciPy cubic spline
(`interp1d`) to map 1-minute slow morphological frames onto a fast 500Hz master
clock. Biologically, cells do not deform like simple polynomials.

Task: Please replace the V1 lazy interpolation with a Continuous-Time Neural ODE
using the `torchdiffeq` library.

1. Create a lightweight PyTorch module named
   `MorphologicalVectorField(nn.Module)` containing a 2-layer MLP. This will
   parameterize the continuous biological velocity dz(t)/dt = f(z(t), t).
2. Inside `align_to_master_clock`, instantiate this vector field and use
   `torchdiffeq.odeint` to project the 100-D latent shape vectors (Sigma)
   continuously across the master clock timeline.
3. Ensure the integration time steps strictly correspond to the 500Hz
   micro-bursts, preserving our O(N) memory constraints by only calculating the
   exact evaluation points needed.
4. Use standard PyTorch tensors and avoid altering the existing DataFrame output
   format.
5. Include a regression test that demonstrates the Neural ODE provides smoother
   transitions between time steps than the original cubic spline.

Dependencies

- torchdiffeq
