Load src/demo/01_hardware_scaling_proof.py.

Import SpectralMetrics.

Compute the Power Spectral Density (PSD) of the hidden_states over a sliding window leading up to the EVENT_FRAME (the 50μM Diazepam shock).

Update the plot_dashboard function to accept PSD data and generate a 4-panel figure.

The 4th panel should plot the PSD, visually demonstrating the collapse of the primary resonant peaks and the elevation of the broadband 1/f noise floor as the pharmacological shock induces friction (Γ) failure.
