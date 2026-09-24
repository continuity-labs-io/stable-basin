# Computational Biology Paper Blueprint: IMRAD Structure

This document is the architectural blueprint for drafting research papers in the Stable Basin repository. It transforms paper writing from a freeform creative exercise into a structured data-entry task. 

Future automated generation tools (e.g., Jinja2 templates via `make paper`) should map JSON metrics and PNG figures directly into this skeleton.

## 1. Abstract
*Strictly 150-250 words. No citations.*
* **Context:** 1-2 sentences establishing the broad biological problem (e.g., aging, resilience).
* **The Gap:** 1 sentence defining what is mathematically or computationally missing in the field.
* **Our Method:** 2 sentences describing the novel architecture/algorithm built to solve this.
* **Validation:** 1-2 sentences explaining how the method was proven (e.g., synthetic positive control, ablation).
* **Results:** 1-2 sentences highlighting the concrete numbers (e.g., False Positive Rate, optimal dose, effect sizes).

## 2. Introduction
* **Background:** Introduce the core biological concept (e.g., Waddington landscapes, phase transitions).
* **The Problem:** Explain why current computational methods (e.g., standard RNNs, static modeling) fail to capture these dynamics accurately.
* **Our Contribution:** A clear, bulleted list of the 3 specific contributions of this paper:
  1. The mathematical/theoretical framework (e.g., Friston Blankets, EBMs).
  2. The computational tool/architecture introduced.
  3. The empirical proof/dataset evaluation.

## 3. Prior Work (or Related Work)
* **Biological Modeling:** What other prominent papers have modeled this specific biological phenomenon?
* **Computational Methods:** Who else has used similar computational architectures (e.g., State Space Models, Energy-Based Models) in biology?
* **Differentiation:** Explicitly state how this approach is mathematically or conceptually different from the prior work.

## 4. Methods
* **Data Source & Preprocessing:** Where did the data come from? How was it normalized? Explicitly delineate between real biological data and synthetic/augmented data.
* **Network Architecture:** Detail the specific neural network blocks used (e.g., `PrecisionWeightedEBM`, `MaskAwareMamba`).
* **Physics Engine / Integration:** Describe the numerical integration scheme (e.g., Euler-Maruyama) and thermodynamic rules applied (e.g., Overdamped Langevin).
* **Evaluation Metrics:** Mathematically define the metrics used to score the model (e.g., Koopman Stability Metric, Energy Distance $R(\lambda)$).

## 5. Results
*Each subsection should reference at least one Figure or Table.*
* **5.1 Baseline Detection:** Did the baseline metrics successfully catch the injected anomaly or phase transition? 
* **5.2 Ablation Study:** Contrast the failure of the control architecture (e.g., `IdentityPrecisionEBM`) against the success of the proposed architecture. *(Reference Figure 1)*.
* **5.3 Thermodynamic Intervention:** Detail the application of the control parameter (e.g., $\lambda$). Describe the resulting shift in the energy landscape or distance metric. *(Reference Figure 2)*.
* **5.4 Dose-Response Dynamics:** Describe the curve of the intervention parameter and identify the optimal dose discovered. *(Reference Figure 3)*.

## 6. Discussion
* **Interpretation:** What do these results mean for the broader field of computational biology and the specific biological problem?
* **Limitations:** Acknowledge weaknesses transparently (e.g., "This study relied on a synthetic positive control, not real longitudinal aging data"). Reviewers require this.
* **Future Work:** Outline the immediate next steps required to scale this work (e.g., expanding to omics-level systems, acquiring longitudinal data).
