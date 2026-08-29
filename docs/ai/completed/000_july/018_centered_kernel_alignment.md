Please update the `ThermodynamicMetrics` class to include a new method called
`calculate_cka(self, z_seq1, z_seq2)`. This method calculates the Linear
Centered Kernel Alignment (CKA) to prove that the geometric shape of the
biological manifold is preserved across multi-day recordings, even in the
presence of representational drift.

Implement this exclusively using pure PyTorch tensor operations to avoid
introducing heavy external dependencies.

Follow these steps for the implementation:

1. Validate that `z_seq1` and `z_seq2` share the same temporal length (n). If
   not, trim them to the minimum length using
   `min_steps = min(z_seq1.shape[0], z_seq2.shape[0])`.
2. Compute the linear Gram matrices: K = z_seq1 @ z_seq1.T and L = z_seq2 @
   z_seq2.T.
3. Center the Gram matrices. Create a centering matrix H = I - (1/n) using
   `torch.eye(n)` and `torch.ones(n, n) / n`. Then, calculate the centered
   matrices: K_c = H @ K @ H and L_c = H @ L @ H.
4. Compute the Hilbert-Schmidt Independence Criterion (HSIC) as
   `torch.trace(K_c @ L_c)`.
5. Return the normalized CKA score: HSIC(K_c, L_c) / torch.sqrt(HSIC(K_c, K_c)
   \* HSIC(L_c, L_c)).
