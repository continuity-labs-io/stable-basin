MODELS ?= zero_padded_ssm forward_fill_ssm mask_concat_ssm causal_transformer masr_ssm masr_mamba gru_d ode_rnn

.PHONY: sensor-fusion-baseline sensor-fusion-extrapolation sensor-fusion-imputation sensor-fusion-all

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

sensor-fusion-all: sensor-fusion-baseline sensor-fusion-extrapolation sensor-fusion-density-sweep

.PHONY: clinical-diagnostic


clinical-diagnostic:
	python -m src.harness.clinical_diagnostic_runner --config configs/clinical_diagnostic.yaml

.PHONY: preflight docker-build

preflight:
	@echo "Running Preflight Smoke Tests..."
	pytest tests/harness/test_smoke.py
	@echo "Smoke tests passed! The registry is stable."

# docker-build:
# 	@echo "Building Stable Basin Docker Image..."
# 	docker build -t stable-basin:latest .
