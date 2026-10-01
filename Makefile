MODELS ?= zero_padded_ssm forward_fill_ssm mask_concat_ssm causal_transformer masr_ssm masr_mamba gru_d ode_rnn
DATASET ?= echo_resilience

.PHONY: baseline extrapolation loss-ablation density-sweep ssm-experiments ksm-threshold echo-resilience-ebm echo-resilience-lambda-comparison echo-resilience-experiments lint-pytorch preflight audit coverage paper

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

n ?= 1
N ?= $(n)

.PHONY: prompts
prompts:
	python -m tools.make_prompts -n $(N)

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

ksm-threshold:
	python -m src.harness.ksm_threshold_runner --config configs/ksm_threshold.yaml

quickstart:
	WANDB_MODE=disabled python -m src.harness.ksm_threshold_runner --config configs/ksm_threshold.yaml --use_synthetic --no-ray

ssm-experiments: baseline extrapolation density-sweep loss-ablation ksm-threshold

# ==========================================
# Demos and Tutorials
# ==========================================

.PHONY: demo-observer
demo-observer:
	python -m examples.01_simulate_observer_zero --save output/observer_demo.mp4

.PHONY: notebooks
notebooks:
	jupyter notebook notebooks/


# ==========================================
# Lifespan Suite (Track 4)
# ==========================================

.PHONY: lifespan
lifespan:
	python -m src.benchmarks.lifespan.run --suite configs/lifespan/suite.yaml $(if $(ONLY),--only $(ONLY))

.PHONY: lifespan-test
lifespan-test:
	pytest tests/lifespan -v

# ==========================================
# Parked: track 3 (Echo)
# ==========================================
# The Aging Resilience suite abstracts the workflow so it can run across 
# multiple diverse datasets. Use the convenience targets below to run the 
# full pipeline end-to-end on a specific dataset.

.PHONY: run-worm-gait
run-worm-gait:
	$(MAKE) echo-resilience-experiments DATASET=echo_resilience

.PHONY: run-killifish
run-killifish:
	$(MAKE) echo-resilience-experiments DATASET=killifish_experiments

.PHONY: run-catnap
run-catnap:
	$(MAKE) echo-resilience-experiments DATASET=catnap_experiments

.PHONY: run-synthetic-aging
run-synthetic-aging:
	WANDB_MODE=disabled $(MAKE) echo-resilience-experiments DATASET=synthetic_aging

# ------------------------------------------
# Granular Pipeline Steps
# ------------------------------------------
# You can also run individual steps for a specific dataset like so:
# make echo-resilience-ebm DATASET=killifish_experiments


.PHONY: echo-resilience-baseline
echo-resilience-baseline:
	python -m src.benchmarks.echo_resilience.01_baseline_metrics --config configs/$(DATASET).yaml

.PHONY: echo-resilience-ssm
echo-resilience-ssm:
	python -m src.benchmarks.echo_resilience.02_aging_ssm --config configs/$(DATASET).yaml

.PHONY: echo-resilience-transformer
echo-resilience-transformer:
	python -m src.benchmarks.echo_resilience.03_aging_transformer --config configs/$(DATASET).yaml

.PHONY: echo-resilience-optune
echo-resilience-optune:
	python -m src.benchmarks.echo_resilience.04_optune_ebm_architecture --config configs/$(DATASET).yaml

.PHONY: echo-resilience-ebm
echo-resilience-ebm:
	python -m src.benchmarks.echo_resilience.05_aging_ebm --config configs/$(DATASET).yaml

.PHONY: echo-resilience-fit-lambda
echo-resilience-fit-lambda:
	python -m src.benchmarks.echo_resilience.06_fit_lambda --config configs/$(DATASET).yaml

.PHONY: echo-resilience-lambda-comparison
echo-resilience-lambda-comparison:
	python -m src.benchmarks.echo_resilience.07_lambda_comparison --config configs/$(DATASET).yaml

.PHONY: echo-resilience-sweep
echo-resilience-sweep:
	python -m src.benchmarks.echo_resilience.08_lambda_sweep --config configs/$(DATASET).yaml

.PHONY: echo-resilience-lambda-argmax
echo-resilience-lambda-argmax:
	python -m src.benchmarks.echo_resilience.09_lambda_argmax --config configs/$(DATASET).yaml

.PHONY: echo-resilience-animate
echo-resilience-animate:
	python -m src.benchmarks.echo_resilience.10_animate --config configs/$(DATASET).yaml

.PHONY: echo-resilience-null-control
echo-resilience-null-control:
	python -m src.benchmarks.echo_resilience.11_null_control --config configs/$(DATASET).yaml

.PHONY: echo-resilience-experiments
echo-resilience-experiments: echo-resilience-baseline echo-resilience-ssm echo-resilience-transformer echo-resilience-optune echo-resilience-ebm echo-resilience-fit-lambda echo-resilience-lambda-comparison echo-resilience-sweep echo-resilience-lambda-argmax echo-resilience-animate echo-resilience-null-control

.PHONY: reproduce-paper
reproduce-paper: echo-resilience-experiments paper

