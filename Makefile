MODELS ?= zero_padded_ssm forward_fill_ssm mask_concat_ssm causal_transformer masr_ssm masr_mamba gru_d ode_rnn

.PHONY: sensor-fusion-baseline sensor-fusion-extrapolation sensor-fusion-loss-ablation sensor-fusion-imputation sensor-fusion-all

sensor-fusion-loss-ablation:
	python -m src.harness.sensor_fusion_sweep \
		--config configs/sensor_fusion.yaml \
		--task loss_ablation

sensor-fusion-baseline:
	python -m src.harness.sensor_fusion_sweep \
		--config configs/sensor_fusion.yaml \
		--task baseline

sensor-fusion-extrapolation:
	python -m src.harness.sensor_fusion_sweep \
		--config configs/sensor_fusion.yaml \
		--task extrapolation

sensor-fusion-density-sweep:
	python -m src.harness.sensor_fusion_sweep \
		--config configs/sensor_fusion.yaml \
		--task density_sweep

sensor-fusion-all: sensor-fusion-baseline sensor-fusion-extrapolation sensor-fusion-density-sweep sensor-fusion-loss-ablation

.PHONY: clinical-diagnostic

clinical-diagnostic:
	python -m src.harness.clinical_diagnostic_runner --config configs/clinical_diagnostic.yaml

.PHONY: lint-pytorch preflight

lint-pytorch:
	@echo "Running TorchFix..."
	@echo "TorchFix will catch deprecated PyTorch symbols, missing autograd contexts, and dangerous in-place operations that break backpropagation."
	torchfix src/ tests/

preflight: lint-pytorch
	pytest
