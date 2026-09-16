Upgrade the "Observer Zero" simulation from a passively stable system to one
that demonstrates Active Coherence Maintenance via external perturbation. I want
to interactively inject a "wound" (a sensation) into the Markov Blanket and
watch the internal geometry dynamically deform to repair it.

Please modify the existing src/models/vessel/observer_zero.py script with the
following additions:

Interactive Event Handling: Connect a button_press_event listener to the
Matplotlib figure. The Sensation Injection: When the user clicks the plot,
capture the (x, y) data coordinates. Map those coordinates to the underlying
PyTorch u and v tensors. The Wound Mechanism: At the location of the click,
inject a massive localized spike of energy. Specifically, force a circular or
square region (e.g., radius of 5 to 10 pixels) in the u tensor to a high
activation value (e.g., u = 3.0) and the v tensor to a low value to simulate a
sudden, violent external stimulus disrupting the local topology. Continuous
Integration: Ensure the FuncAnimation continues running seamlessly after the
click. The visual output must allow the user to watch the stable geometric
spirals violently deform upon impact, and then observe the Reaction-Diffusion
physics actively repairing the boundary over subsequent integration steps (Δt).

Ensure the coordinate mapping correctly translates Matplotlib data coordinates
to the PyTorch tensor indices. Keep the core continuous-time physics intact.
