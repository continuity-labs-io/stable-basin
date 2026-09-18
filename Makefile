MODELS ?= zero_padded_ssm forward_fill_ssm mask_concat_ssm causal_transformer masr_ssm masr_mamba gru_d ode_rnn

.PHONY: baseline extrapolation loss-ablation imputation all-experiments

loss-ablation:
	python -m src.harness.sensor_fusion_sweep \
		--config configs/baseline_experiments.yaml \
		--task loss_ablation

baseline:
	python -m src.harness.sensor_fusion_sweep \
		--config configs/baseline_experiments.yaml \
		--task baseline

extrapolation:
	python -m src.harness.sensor_fusion_sweep \
		--config configs/baseline_experiments.yaml \
		--task extrapolation

density-sweep:
	python -m src.harness.sensor_fusion_sweep \
		--config configs/baseline_experiments.yaml \
		--task density_sweep

ssm-experiments: baseline extrapolation density-sweep loss-ablation

.PHONY: clinical-diagnostic

clinical-diagnostic:
	python -m src.harness.clinical_diagnostic_runner --config configs/clinical_diagnostic.yaml

.PHONY: lint-pytorch preflight audit

lint-pytorch:
	@echo "Running TorchFix..."
	@echo "TorchFix will catch deprecated PyTorch symbols, missing autograd contexts, and dangerous in-place operations that break backpropagation."
	-LIBCST_PARSER_TYPE=pure torchfix -j 1 src/ tests/

preflight: lint-pytorch
	pytest

audit:
	python tools/multi_agent_auditor.py --mode "repo"

.PHONY: worm-gait-ebm coverage

coverage:
	@echo "Running unit tests with coverage analysis..."
	pytest --cov=src --cov-report=term-missing tests/

worm-gait-ebm:
	python -m src.benchmarks.06_worm_gait_aging_ebm --config configs/worm_gait_ebm.yaml

.PHONY: worm-gait-intervention

worm-gait-intervention:
	python -m src.benchmarks.07_worm_gait_intervention --config configs/worm_gait_intervention.yaml
