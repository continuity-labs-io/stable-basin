MODELS ?= zero_padded_ssm forward_fill_ssm mask_concat_ssm causal_transformer masr_ssm masr_mamba gru_d ode_rnn

.PHONY: baseline extrapolation loss-ablation density-sweep ssm-experiments clinical-diagnostic aging-resilience-ebm aging-resilience-intervention aging-resilience-experiments lint-pytorch preflight audit coverage paper

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
	python -m tools.code_auditor.multi_agent_auditor --mode "repo"

coverage:
	@echo "Running unit tests with coverage analysis..."
	pytest --cov=src --cov-report=term-missing tests/

# The PAPER variable specifies the subfolder within the paper/ directory to compile.
# You can override it from the CLI, e.g.: make paper PAPER=my_future_paper
PAPER ?= sharpening_the_tack

paper:
	python -m tools.paper.compile_paper --paper $(PAPER)
	cd paper/$(PAPER) && pdflatex -interaction=batchmode $(PAPER).tex


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

.PHONY: aging-resilience-baseline
aging-resilience-baseline:
	python -m src.benchmarks.aging_resilience.01_baseline_metrics

.PHONY: aging-resilience-ssm
aging-resilience-ssm:
	python -m src.benchmarks.aging_resilience.02_aging_ssm

.PHONY: aging-resilience-transformer
aging-resilience-transformer:
	python -m src.benchmarks.aging_resilience.03_aging_transformer

.PHONY: aging-resilience-optune
aging-resilience-optune:
	python -m src.benchmarks.aging_resilience.04_optune_ebm_architecture

.PHONY: aging-resilience-ebm
aging-resilience-ebm:
	python -m src.benchmarks.aging_resilience.05_aging_ebm --config configs/aging_resilience.yaml

.PHONY: aging-resilience-infer-lambda
aging-resilience-infer-lambda:
	python -m src.benchmarks.aging_resilience.06_infer_biological_lambda

.PHONY: aging-resilience-intervention
aging-resilience-intervention:
	python -m src.benchmarks.aging_resilience.07_intervention --config configs/aging_resilience.yaml

.PHONY: aging-resilience-sweep
aging-resilience-sweep:
	python -m src.benchmarks.aging_resilience.08_lambda_sweep --config configs/aging_resilience.yaml

.PHONY: aging-resilience-pharmacology
aging-resilience-pharmacology:
	python -m src.benchmarks.aging_resilience.09_pharmacological_translation --config configs/aging_resilience.yaml

.PHONY: aging-resilience-animate
aging-resilience-animate:
	python -m src.benchmarks.aging_resilience.10_animate

.PHONY: aging-resilience-null-control
aging-resilience-null-control:
	python -m src.benchmarks.aging_resilience.11_null_control --config configs/aging_resilience.yaml

.PHONY: aging-resilience-experiments
aging-resilience-experiments: aging-resilience-baseline aging-resilience-ssm aging-resilience-transformer aging-resilience-optune aging-resilience-ebm aging-resilience-infer-lambda aging-resilience-intervention aging-resilience-sweep aging-resilience-pharmacology aging-resilience-animate aging-resilience-null-control

.PHONY: reproduce-paper
reproduce-paper: aging-resilience-experiments paper
