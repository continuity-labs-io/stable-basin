# Control Group Parity Symmetrical Hands

### Prompt 5: The Control Group Parity (The Symmetrical Hands)

**Target File:** `tests/models/ssm/test_masr_mamba_rigor.py` **Context Files:**
`src/models/ssm/masr_mamba.py`

**Instructions:** Create an architectural parity test named
`test_mask_aware_control_group_parity`.

1. Initialize two instances of `MaskAwareMamba`: one with `mask_aware=True` and
   one with `mask_aware=False`.
2. Assert that both models possess the exact same sequential projection depth by
   verifying their `input_proj` modules contain both `nn.LayerNorm` and
   `nn.GELU`.
3. Pass a clean, fully-observed dummy input through both models (ensuring shapes
   align properly for both conditions).
4. Verify that both forward passes complete successfully and output tensors of
   the exact same dimensional shape. The control group must be proven to be a
   structurally valid, perfectly matched peer!
