.PHONY: sensor-fusion-synthetic sensor-fusion-extrapolation sensor-fusion-imputation sensor-fusion-all

sensor-fusion-synthetic:
	python -m src.harness.sensor_fusion_runner \
		--task-name "Synthetic Benchmark" \
		--epochs 50 \
		--train-seq-len 500 \
		--test-seq-len 500 \
		--models baseline mask_aware transformer \
		--csv-name 02_benchmark.csv \
		--png-name 02_benchmark_results.png

sensor-fusion-extrapolation:
	python -m src.harness.sensor_fusion_runner \
		--task-name "Extrapolation Stress Test" \
		--epochs 40 \
		--train-seq-len 500 \
		--test-seq-len 2000 \
		--models baseline mask_aware transformer \
		--csv-name 03_stress_test.csv \
		--png-name 03_extrapolation_results.png

sensor-fusion-imputation:
	python -m src.harness.sensor_fusion_runner \
		--task-name "Imputation Baseline Comparison" \
		--epochs 40 \
		--train-seq-len 500 \
		--test-seq-len 2000 \
		--models baseline forward_fill mask_concat transformer mask_aware \
		--csv-name 04_arxiv_results.csv \
		--png-name 04_arxiv_money_chart.png

sensor-fusion-all: sensor-fusion-synthetic sensor-fusion-extrapolation sensor-fusion-imputation
