MODELS ?= baseline forward_fill mask_concat transformer mask_aware mask_aware_mamba gru_d ode_rnn

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

.PHONY: clinical-autopsy

EPOCHS ?= 15

clinical-autopsy:
	python -m src.harness.clinical_autopsy_runner --config configs/clinical_autopsy.yaml

.PHONY: preflight docker-build

preflight:
	@echo "Running Preflight Smoke Tests..."
	python -m src.harness.smoke_test
	@echo "Smoke tests passed! The registry is stable."

docker-build:
	@echo "Building Stable Basin Docker Image..."
	docker build -t stable-basin:latest .
