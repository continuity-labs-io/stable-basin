.PHONY: sensor-fusion-synthetic sensor-fusion-extrapolation sensor-fusion-imputation sensor-fusion-all

sensor-fusion-synthetic:
	python -m src.harness.sensor_fusion_runner \
		--task-name "Synthetic Benchmark" \
		--epochs 50 \
		--train-seq-len 500 \
		--test-seq-len 500 \
		--models baseline forward_fill mask_concat transformer mask_aware gru_d ode_rnn \
		--csv-name 01_synthetic_benchmark.csv \
		--png-name 01_synthetic_benchmark.png

sensor-fusion-extrapolation:
	python -m src.harness.sensor_fusion_runner \
		--task-name "Extrapolation Stress Test" \
		--epochs 40 \
		--train-seq-len 500 \
		--test-seq-len 2000 \
		--models baseline forward_fill mask_concat transformer mask_aware gru_d ode_rnn \
		--csv-name 02_extrapolation_stress_test.csv \
		--png-name 02_extrapolation_stress_test.png

sensor-fusion-imputation:
	python -m src.harness.sensor_fusion_runner \
		--task-name "Imputation Baseline Comparison" \
		--epochs 40 \
		--train-seq-len 500 \
		--test-seq-len 2000 \
		--models baseline forward_fill mask_concat transformer mask_aware gru_d ode_rnn \
		--csv-name 03_imputation_baseline_comparison.csv \
		--png-name 03_imputation_baseline_comparison.png

sensor-fusion-sparsity-sweep:
	python -m src.harness.sparsity_sweep_runner \
		--epochs 10 \
		--train-seq-len 500 \
		--test-seq-len 2000 \
		--sparsities 0.1 0.05 0.02 0.01 0.005 0.001 \
		--seeds 42 100 256 512 1024 \
		--models baseline forward_fill mask_concat gru_d ode_rnn mask_aware \
		--csv-name 04_sparsity_sweep.csv \
		--png-name 04_sparsity_sweep.png

sensor-fusion-all: sensor-fusion-synthetic sensor-fusion-extrapolation sensor-fusion-imputation sensor-fusion-sparsity-sweep
