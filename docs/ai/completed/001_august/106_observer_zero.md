We are shifting from theoretical biophysics to executable simulations. I need to
build "Observer Zero," a 2D continuous-time physics simulation in pure PyTorch
to visualize spontaneous phase-locking, the emergence of a Markov Blanket, and
self-sustaining vortices from noise.

Please write a self-contained PyTorch script in
src/models/vessel/observer_zero.py that implements a Reaction-Diffusion system
(specifically, a FitzHugh-Nagumo activator-inhibitor model) with stochastic
noise.

Requirements:

The Substrate: Create a 2D grid (e.g., 128x128 or 256x256 tensors) for two
coupled state variables: u (the activator/internal state) and v (the
inhibitor/boundary state). Initialize both entirely with high-frequency random
uniform noise. The Physics (Continuous-Time Update): Define a recurrent update
step using a small continuous Δt. The equations should follow the general form:

u(t+Δt) = u(t) + Δt · [D_u ∇² u + u - u³ - v + σdW] v(t+Δt) = v(t) + Δt · [D_v
∇² v + ε(u - γv)]

The Kinematics (Coupling): Implement the spatial Laplacian ∇² using a 2D PyTorch
convolution (a 3x3 Laplacian kernel). This allows neighboring pixels to share
their state, acting as the biological gap-junction grid. The Noise: Inject a
small amount of ongoing Brownian noise (σdW) at each integration step to
simulate thermodynamic entropy. The Simulation Loop & Animation: Use
matplotlib.animation.FuncAnimation to run the integration loop. In each frame,
execute several micro-steps of the continuous-time update to speed up the visual
evolution. Ensure the driver script is distinct from the model and ensure the
drive is put in src/demo/vessel/ The Output: Render a live heatmap of the u
state. The visual output must clearly show the initial chaos settling into
organized, stable geometric spirals/vortices (the attractors/Markov blankets)
pushing back against the background static.

Ensure all mathematical variables in the comments use standard Unicode
characters (e.g., Δt, σdW, ∇²). Write the code so it executes perfectly on a
local CPU/Metal backend without Triton. Include a **main** block so the script
runs the animation directly.
