The Continuous-Time Spectrogram (Matrix A Initializer): In src/models/ssm/physics, create a PyTorch module BiologicalSpectrogramInit. This function must initialize the State Space Model's Matrix A using log-spaced negative real eigenvalues. Explicitly map these initialization frequencies to target biological timescales ranging from high-frequency electrophysiology (e.g., 20,000 Hz) to slow morphological drift (e.g., 0.0001 Hz). Output the matrix in the parameterization A = -exp(A_log) to guarantee system stability. Document the math clearly using standard Unicode symbols.

include tests
