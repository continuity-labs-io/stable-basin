MODELS ?= zero_padded_ssm forward_fill_ssm mask_concat_ssm causal_transformer masr_ssm masr_mamba gru_d ode_rnn

.PHONY: baseline extrapolation loss-ablation density-sweep ssm-experiments clinical-diagnostic worm-gait-ebm worm-gait-intervention worm-gait-experiments lint-pytorch preflight audit coverage paper

# ==========================================
# General Software Engineering Tools
# ==========================================

lint-pytorch:
	@echo "Running TorchFix..."
	@echo "TorchFix will catch deprecated PyTorch symbols, missing autograd contexts, and dangerous in-place operations that break backpropagation."
	-LIBCST_PARSER_TYPE=pure torchfix -j 1 src/ tests/

preflight: lint-pytorch
	pytest

audit:
	python tools/multi_agent_auditor.py --mode "repo"

coverage:
	@echo "Running unit tests with coverage analysis..."
	pytest --cov=src --cov-report=term-missing tests/

# The PAPER variable specifies the subfolder within the paper/ directory to compile.
# You can override it from the CLI, e.g.: make paper PAPER=my_future_paper
PAPER ?= sharpening_the_tack

paper:
	python tools/compile_paper.py --paper $(PAPER)
	cd paper/$(PAPER) && pdflatex $(PAPER).tex


# ==========================================
# Scientific Experiments (SSM Suite)
# ==========================================

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

clinical-diagnostic:
	python -m src.harness.clinical_diagnostic_runner --config configs/clinical_diagnostic.yaml

ssm-experiments: baseline extrapolation density-sweep loss-ablation clinical-diagnostic


# ==========================================
# Scientific Experiments (Worm Gait Suite)
# ==========================================

.PHONY: worm-gait-baseline
worm-gait-baseline:
	python -m src.benchmarks.worm_gait.01_worm_gait_baseline_metrics

.PHONY: worm-gait-infer-lambda
worm-gait-infer-lambda:
	python -m src.benchmarks.worm_gait.02_infer_biological_lambda

.PHONY: worm-gait-ssm
worm-gait-ssm:
	python -m src.benchmarks.worm_gait.03_worm_gait_aging_ssm

.PHONY: worm-gait-transformer
worm-gait-transformer:
	python -m src.benchmarks.worm_gait.04_worm_gait_aging_transformer

.PHONY: worm-gait-optune
worm-gait-optune:
	python -m src.benchmarks.worm_gait.05_worm_gait_optune_ebm_architecture

.PHONY: worm-gait-ebm
worm-gait-ebm:
	python -m src.benchmarks.worm_gait.06_worm_gait_aging_ebm --config configs/worm_gait_ebm.yaml

.PHONY: worm-gait-intervention
worm-gait-intervention:
	python -m src.benchmarks.worm_gait.07_worm_gait_intervention --config configs/worm_gait_intervention.yaml

.PHONY: worm-gait-sweep
worm-gait-sweep:
	python -m src.benchmarks.worm_gait.08_worm_gait_lambda_sweep --config configs/worm_gait_intervention.yaml

.PHONY: worm-gait-pharmacology
worm-gait-pharmacology:
	python -m src.benchmarks.worm_gait.09_pharmacological_translation --config configs/worm_gait_intervention.yaml

.PHONY: worm-gait-animate
worm-gait-animate:
	python -m src.benchmarks.worm_gait.10_animate_worm_gait

.PHONY: worm-gait-experiments
worm-gait-experiments: worm-gait-baseline worm-gait-infer-lambda worm-gait-ssm worm-gait-transformer worm-gait-optune worm-gait-ebm worm-gait-intervention worm-gait-sweep worm-gait-pharmacology worm-gait-animate

.PHONY: reproduce-paper
reproduce-paper: worm-gait-experiments paper
