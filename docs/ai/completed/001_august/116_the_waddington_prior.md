# The Waddington Prior

A rigorous visualization of the mathematical definition of aging as the thermodynamic flattening of a biological Prior. Implementation in src/demo/vessel/waddington_prior.py that simulates Fristonian Non-Equilibrium Steady State (NESS) dynamics via matrix multiplication.

## Requirements:

### The Mathematical Setup:

Define a 2D state vector x. Set the Mean mu = torch.tensor([0.0, 0.0]).

Define a Solenoidal matrix Q = torch.tensor([[0.0, 1.0], [-1.0, 0.0]]).

Define a Dissipation matrix Gamma = torch.tensor([[0.1, 0.0], [0.0, 0.1]]).

The Two Conditions (Youth vs. Aging):

Define Pi_youth as a matrix with high eigenvalues (e.g., [[5.0, 0.0], [0.0, 5.0]]). This creates a steep geometric basin.

Define Pi_aging as a matrix with low eigenvalues (e.g., [[0.5, 0.0], [0.0, 0.5]]). This creates a flattened, shallow basin.

The Continuous-Time Update Loop:

Implement the discrete Euler-Maruyama integration step using explicit PyTorch matrix multiplication (@):
x = x - dt * ((Q - Gamma) @ Pi @ (x - mu)) + torch.sqrt(2 * Gamma * dt) @ torch.randn(2)

Run this update loop for both the Youth state and the Aging state simultaneously. Inject the exact same random noise seed into both steps so the only difference in behavior is caused by the Precision matrix Pi.

The Visualization & Telemetry:

Use matplotlib.animation.FuncAnimation to create a 1x2 subplot figure (Left: Youth, Right: Aging).

Geometry: In the background of each subplot, plot the contour lines of the Potential Energy basin: U(x) = 0.5 * (x - mu)^T @ Pi @ (x - mu). The Youth contour should be tightly packed (steep); the Aging contour should be wide and spread out (flat).

Particle: Plot the live trajectory of the state x as a moving dot orbiting the basin, leaving a short fading trail behind it.

Telemetry: Overlay live text on the screen for each panel displaying:
a) Trace(Pi) (Precision Metric)
b) U(x) (Current Potential Energy / Surprisal)
c) ||x - mu|| (Distance from Target Mean)

Ensure all variables in the comments use standard Unicode. The script must execute cleanly on CPU and include a __main__ block to render the animation.
