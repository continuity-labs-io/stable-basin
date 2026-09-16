# Bridge Goodfire SPD with MELD Mamba

This plan outlines the architecture to run the Stochastic Parameter
Decomposition (SPD) on the custom `StateSpaceEngine` trained inside
`1_hssm_demo.py`.

## User Review Required

> [!IMPORTANT] The Goodfire SPD library uses `einops` for decompositions on
> linear and embedding matrices. Decomposing 1D Convolutions requires a new
> structural component `Conv1dComponent`. I've outlined the math and PyTorch
> module structure below. Please verify if the proposed stochastic mask
> integration for Conv1d aligns with your mathematical expectations.

## Proposed Changes

### Mamba End-to-End Core

#### [MODIFY] 1_hssm_demo.py

Export the trained `StateSpaceEngine` (Fusion Core) weights.

- Modify the `__main__` block.
- After `train_orthogonal_veto` completes, extract the `mamba` engine.
- Create the output directory `output/spd/` if it doesn't exist.
- Save the weights:
  `torch.save(mamba.state_dict(), "output/spd/meld_mamba.pth")`.

---

### Configuration

#### [MODIFY] spd_mamba_config.yaml

Update the SPD config to target our local Mamba architecture rather than the
Hugging Face pre-trained model.

- Set `pretrained_model_class: src.models.state_space_engine.StateSpaceEngine`
- Set `pretrained_model_path: output/spd/meld_mamba.pth`
- Set `pretrained_model_name_hf: null`
- Set `tokenizer_name: null`
- Set `target_module_patterns:` to `["mamba.conv1d", "forward_predictor"]`
  (Targeting Mamba's internal continuous convolution and our local predictor).

---

### Interpretability Pipeline

#### [MODIFY] spd_interpreter.py

Implement the Goodfire patch to support 1D Convolutions.

- **Define `Conv1dComponent`**: Create a custom `nn.Module` mimicking
  `LinearComponent` that parameterizes the convolution weights as $W = B^T A^T$,
  reshapes to the proper Conv1d Kernel size, and supports the stochastic `.mask`
  property.
- **Dynamic Monkey-Patch**: Hook `ComponentModel.create_target_components` so
  when it encounters an `nn.Conv1d` layer (like inside Mamba), it replaces it
  with our new `Conv1dComponent` instead of raising a ValueError.

## Verification Plan

### Manual Verification

1. I will run `python src/demo/1_hssm_demo.py` to train the custom Mamba engine
   and export the weights to `output/spd/meld_mamba.pth`.
2. I will run `python src/metrics/spd_interpreter.py`. We will watch the
   Goodfire API boot up, load the local `.pth` file, replace `mamba.conv1d` with
   our custom stochastic block, and run the decomposition!
