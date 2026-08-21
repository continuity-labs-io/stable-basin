We are upgrading our continuous-time simulation from a reactive "Observer" to an "Active Inference Agent" embodying Fristonian Free Energy minimization and chemotaxis.

Please write a new PyTorch script in src/models/vessel/active_inference_agent.py. Build upon the previous Reaction-Diffusion architecture, but implement the following thermodynamic agency mechanics:

The Environment (Nutrient Field): Introduce a static or slowly diffusing 2D Gaussian nutrient field (a "food source") placed off-center in the grid.
Metabolic Decay: Introduce a global decay parameter to the u (activator) tensor. The internal energy of the agent's geometric structures must slowly drain over time (Δt).
Sensation (The Gradient): At each step, calculate the spatial gradient (∇) of the external nutrient field that physically intersects with the agent's current high-activation u pixels.
Active Inference (Chemotaxis): Implement a feedback mechanism where the agent minimizes its prediction error (the dropping metabolic energy) by coupling the nutrient gradient to its spatial diffusion. The Laplacian D_u parameter should become spatially anisotropic, heavily favoring diffusion in the direction of the highest nutrient concentration.
The Visual Result: When running the Matplotlib animation, the user should observe a stable Reaction-Diffusion cluster (the Agent) begin to metabolically fade, sense the external Gaussian nutrient field, and actively deform its topological boundary to "crawl" up the gradient toward the food source to replenish its energy.

Ensure the equations remain continuous-time and execute natively in PyTorch. Include a __main__ block to run the animation.

