# Open Challenges

*Stable Basin is an open-source biological physics engine. The core math is written, but we are looking for engineers to take ownership of specific infrastructure nodes. If you build it, you will receive lead-contributor status and potential co-authorship on resulting papers.*

---

### The RL / Game Dev Quest: "The Rejuvenation Gymnasium"

* **Target:** Reinforcement Learning (RL) Engineers, AI Agent Builders.
* **Where it lives in the code:** The intersection of `rejuvenation_controller.py` (which uses a hardcoded threshold `if ksm_score < 0.85`) and the terminal game `11_ratchet_simulator.py`.
* **The Problem:** Aging and therapy dosing is a sequential decision-making problem, but currently, it is driven by basic heuristics and text prompts.
* **The Pitch:** *"Help us turn biological age reversal into an AI benchmark. We need an RL engineer to wrap our continuous physics engine and Ratchet Simulator into a standard Farama `Gymnasium` (OpenAI Gym) environment. Define the states (KSM/CSD) and actions (IV Flow/Therapy Power) so the global AI community can train PPO or SAC agents to autonomously discover optimal longevity protocols."*


### V2 State Vector: The "Quintet" Tensor

The Current fused state vector combines Optical, RNA, and Electrophysiological data, but lacks single-cell epigenetic and in-line electrochemical data. This task builds the dataloaders for single-cell epigenetic
clocks (Gamma) and in-line electrochemical sensors (Mu). Ensure the state space models can gracefully handle the massive `NaN`gaps of hourly epigenetic reads.
