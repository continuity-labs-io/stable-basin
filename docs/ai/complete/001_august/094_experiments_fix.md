Step 1: Establish the Single Source of Truth

In src/harness/sensor_fusion_predictor.py, please update the SSMType Enum to
serve as our absolute single source of truth for the entire repository. Rename
the enum members and their string values to match our manuscript exactly:
ZERO_PADDED_SSM = 'zero_padded_ssm', FORWARD_FILL_SSM = 'forward_fill_ssm',
MASK_CONCAT_SSM = 'mask_concat_ssm', CAUSAL_TRANSFORMER = 'causal_transformer',
MASR = 'masr', MASR_MAMBA = 'masr_mamba', GRU_D = 'gru_d', and ODE_RNN =
'ode_rnn'. Remove any legacy names like 'meld' or 'baseline'. Then, update the
**init** and forward logic inside SensorFusionPredictor to route strictly using
these new Enum values.

Step 2: Fix the Diagnostic Runner & The Bug

In src/harness/clinical_diagnostic_runner.py, please remove the custom
ModelAdapter class entirely, as well as the hardcoded if/elif instantiation
block inside evaluate_model. Instead, import SensorFusionPredictor and SSMType.
Instantiate the model cleanly using SensorFusionPredictor(ssm_type=model_type,
modality_dims=[input_dim], d_model=d_model, out_dim=input_dim). This
automatically grants the runner access to the Forward-Fill and Mask-Concat
baselines, and it fixes the critical bug where mask_aware=False was being passed
to the Mamba model.

Step 3: Update MambaLRP References (Crucial Follow-up)

Because we just swapped ModelAdapter for SensorFusionPredictor in the diagnostic
runner, we need to update the LRP engine. In src/metrics/mamba_lrp.py, please
update the _lrp_linear backward propagation to reference
self.model.fusion.W_cart (instead of self.model.input_proj) and
self.model.readout (instead of self.model.forward_head), ensuring the dimensions
align with the SensorFusionPredictor wrapper.

Step 4: Audit Configs and Makefiles

Please audit the Makefile, configs/clinical_diagnostic.yaml,
configs/sensor_fusion.yaml, src/harness/smoke_test.py, and
src/harness/sensor_fusion_sweep.py. Ensure that any lists of models, CLI
arguments, or plotting dictionaries strictly use the newly defined string values
from our SSMType Enum (e.g., 'masr_mamba', 'zero_padded_ssm'), completely
scrubbing the legacy strings.
