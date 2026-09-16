Update StateSpaceEngine. Initialize a Mamba-2 backbone using the 'mamba-ssm'
package. The module should accept continuous input tensors of shape (batch,
time_steps, num_neurons), project them into a hidden state dimension, pass them
through the Mamba-2 blocks, and apply a final linear layer with a Softplus
activation to predict non-negative spike rates for the next time step.

Expose a method to return the internal hidden state for downstream visualization
of the model's memory retention.
