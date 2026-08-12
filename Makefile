MODELS ?= baseline forward_fill mask_concat transformer mask_aware mask_aware_mamba gru_d ode_rnn

.PHONY: sensor-fusion-baseline sensor-fusion-extrapolation sensor-fusion-imputation sensor-fusion-all

sensor-fusion-baseline:
	python -m src.harness.sensor_fusion_runner \
		--task-name "Baseline (Interpolation)" \
		--epochs 50 \
		--train-seq-len 500 \
		--test-seq-len 500 \
		--models $(MODELS) \
		--csv-name 01_baseline_interpolation.csv \
		--png-name 01_baseline_interpolation.png

sensor-fusion-extrapolation:
	python -m src.harness.sensor_fusion_runner \
		--task-name "Extrapolation Test" \
		--epochs 40 \
		--train-seq-len 500 \
		--test-seq-len 5000 \
		--models $(MODELS) \
		--csv-name 02_extrapolation_test.csv \
		--png-name 02_extrapolation_test.png

sensor-fusion-density-sweep:
	python -m src.harness.density_sweep_runner \
		--epochs 10 \
		--train-seq-len 500 \
		--test-seq-len 2000 \
		--densities 0.1 0.05 0.02 0.01 0.005 0.001 \
		--seeds 42 100 256 512 1024 \
		--models $(MODELS) \
		--csv-name 03_sensor_density_sweep.csv \
		--png-name 03_sensor_density_sweep.png

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
