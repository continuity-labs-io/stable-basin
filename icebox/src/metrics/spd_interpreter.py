# src/metrics/spd_interpreter.py
"""
SPD Interpreter for Mamba / H-SSM

OVERVIEW & SUPPORT:
This script monkey-patches the Goodfire `spd` (Stochastic Parameter Decomposition) library
so that it can execute on non-Transformer architectures, specifically `Mamba` and `StateSpaceEngine`.
It successfully intercepts the internal optimization loops, overrides HF-specific assertions,
dynamically unfolds 1D continuous convolutions (like Mamba's `conv1d`) into pseudo-linear layers,
and permits the SPD pipeline to run end-to-end (reaching Step 0) without crashing due to architecture mismatches.

LIMITATIONS:
1. `Total Loss: nan` at Step 0: The mask generation relies on Singular Value Decomposition (SVD).
   When initializing the `A` and `B` basis matrices for `Conv1d` weights, SVD currently produces `nan`s
   (likely due to numerical instability with grouped depthwise convolution dimensions or empty weights),
   which subsequently poisons the masks and the faithfulness loss.
2. Brittle Patching: The Goodfire library heavily hardcodes variables such as `d_in`, expects HF Tokenizers,
   and enforces strict output shapes from standard transformers. The deep runtime monkey-patching used here
   is extremely fragile and tightly coupled to the specific version of `spd`.

NEXT STEPS FOR H-SSM INTERPRETABILITY:
- Transition away from trying to force Mamba/H-SSM through the Goodfire LLM-centric experimental pipeline.
- Develop a clean, standalone minimal script that natively implements the SPD mathematical objective
  (sparse dictionary learning / mask optimization) directly on the specific `Conv1d` and `Linear` layers of our H-SSM.
- Isolate the `calc_causal_importances` and `calc_faithfulness_loss` logic so they natively handle 3D sequence
  tensors without relying on PyTorch's `F.unfold` hacks or bypassing Hugging Face assertions.
"""

import os

os.environ["WANDB_MODE"] = "disabled"

import argparse
from pathlib import Path

# Import the main execution loop directly from the installed spd library
from spd.experiments.lm.lm_decomposition import main as run_lm_decomposition


def main():
    # Resolve the absolute path to your config
    default_config = "configs/spd_mamba_config.yaml"

    parser = argparse.ArgumentParser(description="Run SPD on Mamba for MELD Interpretability")
    parser.add_argument("--config", type=str, default=default_config)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find SPD config at {config_path}")

    out_dir = "output/spd"
    os.makedirs(out_dir, exist_ok=True)

    # Monkey-patch Goodfire's Path to reroute the hardcoded output directory
    import spd.experiments.lm.lm_decomposition as lm_module
    import pathlib

    _orig_Path = pathlib.Path

    def patched_Path(*args, **kwargs):
        p = _orig_Path(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], str) and args[0].endswith("lm_decomposition.py"):

            class MockFile:
                @property
                def parent(self):
                    class MockParent:
                        def __truediv__(self, other):
                            if other == "out":
                                return _orig_Path(out_dir)
                            return _orig_Path(args[0]).parent / other

                    return MockParent()

            return MockFile()
        return p

    lm_module.Path = patched_Path

    # Bug in Goodfire's LM pipeline: model_path=None is hardcoded!
    # We patch load_pretrained to inject the model path.
    import spd.utils as spd_utils

    _orig_load_pretrained = spd_utils.load_pretrained

    def patched_load_pretrained(path_to_class, model_path=None, model_name_hf=None, **kwargs):
        actual_path = os.path.join(out_dir, "meld_mamba.pth")
        return _orig_load_pretrained(path_to_class, actual_path, model_name_hf, **kwargs)

    spd_utils.load_pretrained = patched_load_pretrained
    lm_module.load_pretrained = patched_load_pretrained

    logger.info("[*] Booting Stochastic Parameter Decomposition via Goodfire API...")
    logger.info(f"[*] Config: {config_path.name}")
    logger.info(f"[*] Output Dir: {out_dir}")
    # Execute the decomposition natively!
    run_lm_decomposition(config_path_or_obj=config_path)


# --- 1D Convolution Patch for Mamba ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import fnmatch
from spd.models.component_model import ComponentModel
from spd.models.components import LinearComponent, EmbeddingComponent
from spd.module_utils import init_param_


class Conv1dComponent(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups,
        C,
        bias_tensor,
        stride=1,
        padding=0,
        dilation=1,
    ):
        super().__init__()
        self.C = C
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        d_in = (in_channels // groups) * self.kernel_size
        d_out = out_channels

        self.A = nn.Parameter(torch.empty(d_in, C))
        self.B = nn.Parameter(torch.empty(C, d_out))
        self.bias = nn.Parameter(bias_tensor.clone()) if bias_tensor is not None else None

        init_param_(self.A, fan_val=d_out, nonlinearity="linear")
        init_param_(self.B, fan_val=C, nonlinearity="linear")

        self.mask = None

    @property
    def weight(self):
        w = torch.einsum("ic,co->oi", self.A, self.B)
        return w.view(self.out_channels, self.in_channels // self.groups, self.kernel_size)

    def forward(self, x):
        if self.mask is None:
            return F.conv1d(
                x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )

        mask = self.mask
        if mask.ndim == 3:  # (Batch, Seq, C)
            mask = mask.mean(dim=1)

        batch_size = x.shape[0]
        W_batch = torch.einsum("ic,co,bc->boi", self.A, self.B, mask)
        W_batch = W_batch.reshape(
            batch_size * self.out_channels, self.in_channels // self.groups, self.kernel_size
        )

        x_reshaped = x.reshape(1, batch_size * self.in_channels, -1)

        out = F.conv1d(
            x_reshaped,
            W_batch,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * batch_size,
        )

        out = out.view(batch_size, self.out_channels, -1)
        if self.bias is not None:
            out += self.bias.view(1, -1, 1)
        return out


_original_create = ComponentModel.create_target_components


def _patched_create(self, target_module_patterns, C):
    components = {}
    matched_patterns = set()

    for name, module in self.model.named_modules():
        for pattern in target_module_patterns:
            if fnmatch.fnmatch(name, pattern):
                matched_patterns.add(pattern)
                safe_name = name.replace(".", "-")
                if isinstance(module, nn.Linear):
                    d_out, d_in = module.weight.shape
                    components[safe_name] = LinearComponent(
                        d_in=d_in, d_out=d_out, C=C, bias=module.bias
                    )
                elif isinstance(module, nn.Embedding):
                    components[safe_name] = EmbeddingComponent(
                        vocab_size=module.num_embeddings, embedding_dim=module.embedding_dim, C=C
                    )
                elif isinstance(module, nn.Conv1d):
                    components[safe_name] = Conv1dComponent(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size,
                        groups=module.groups,
                        C=C,
                        bias_tensor=module.bias,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                    )
                else:
                    raise ValueError(
                        f"Module '{name}' matched pattern '{pattern}' but type {type(module)} is not supported."
                    )
                break

    unmatched_patterns = set(target_module_patterns) - matched_patterns
    if unmatched_patterns:
        raise ValueError(f"Unmatched patterns: {sorted(unmatched_patterns)}")
    if not components:
        raise ValueError(
            f"No modules found matching target_module_patterns: {target_module_patterns}"
        )

    return nn.ModuleDict(components)


# Apply patch
ComponentModel.create_target_components = _patched_create

import einops
import spd.models.component_model as component_model_module

_orig_init_As_and_Bs_ = component_model_module.init_As_and_Bs_


def _patched_init_As_and_Bs_(model, components) -> None:
    for param_name, component in components.items():
        A = component.A
        B = component.B
        target_weight = model.model.get_parameter(param_name + ".weight")
        if isinstance(component, EmbeddingComponent):
            target_weight = target_weight.T  # (d_out d_in)
        elif isinstance(component, Conv1dComponent):
            # Flatten Conv1d weight (out_channels, in_channels//groups, kernel_size) to (d_out, d_in)
            target_weight = target_weight.reshape(component.out_channels, -1)

        # Make A and B have unit norm in the d_in and d_out dimensions
        A.data[:] = torch.randn_like(A.data)
        B.data[:] = torch.randn_like(B.data)
        A.data[:] = A.data / A.data.norm(dim=-2, keepdim=True)
        B.data[:] = B.data / B.data.norm(dim=-1, keepdim=True)

        # Calculate inner products
        C_norms = einops.einsum(A, B, target_weight, "d_in C, C d_out, d_out d_in -> C")
        # Scale B by the inner product.
        B.data[:] = B.data * C_norms.unsqueeze(-1)


component_model_module.init_As_and_Bs_ = _patched_init_As_and_Bs_

import spd.models.component_utils as component_utils_module

_orig_calc_causal_importances = component_utils_module.calc_causal_importances


def _patched_calc_causal_importances(pre_weight_acts, As, gates, detach_inputs=False):
    causal_importances = {}
    causal_importances_upper_leaky = {}

    for param_name in pre_weight_acts:
        acts = pre_weight_acts[param_name]

        if not acts.dtype.is_floating_point:
            component_act = As[param_name][acts]
        else:
            if acts.dim() == 3 and "conv1d" in param_name:
                import torch.nn.functional as F

                # acts: (batch, in_channels, seq)
                # unfold to get sliding windows of size kernel_size=4
                acts_unfolded = F.unfold(acts.unsqueeze(-1), kernel_size=(4, 1), padding=(3, 0))
                # acts_unfolded: (batch, channels * 4, seq_out)
                batch, _, seq_out = acts_unfolded.shape
                # reshape to (batch, channels, 4, seq_out)
                acts_unfolded = acts_unfolded.view(batch, -1, 4, seq_out)
                # we want to project onto A: (kernel_size=4, C)
                # sum over channels for the overall component activation
                component_act = einops.einsum(
                    acts_unfolded, As[param_name], "b c k s, k C -> b s C"
                )
                # slice seq_out to match original seq len if necessary
                component_act = component_act[:, :512, :]
            else:
                component_act = einops.einsum(acts, As[param_name], "... d_in, d_in C -> ... C")

        gate_input = component_act.detach() if detach_inputs else component_act

        if isinstance(gates[param_name], component_utils_module.GateMLP):
            causal_importances[param_name] = gates[param_name](gate_input)
            causal_importances_upper_leaky[param_name] = gates[param_name](gate_input)
        else:
            causal_importances[param_name] = component_act * gates[param_name](gate_input)
            causal_importances_upper_leaky[param_name] = component_act * gates[param_name](
                gate_input
            )

    return causal_importances, causal_importances_upper_leaky


component_utils_module.calc_causal_importances = _patched_calc_causal_importances

import spd.run_spd as run_spd_module

run_spd_module.init_As_and_Bs_ = _patched_init_As_and_Bs_
run_spd_module.calc_causal_importances = _patched_calc_causal_importances

import spd.losses as losses_module


def _patched_calc_faithfulness_loss(components, target_model, n_params, device):
    target_params = {}
    component_params = {}
    for comp_name, component in components.items():
        component_params[comp_name] = component.weight
        submodule = target_model.get_submodule(comp_name)

        import torch.nn as nn

        if not hasattr(nn.Module, "set_submodule"):

            def _set_submodule(self, target, module):
                atoms = target.split(".")
                name = atoms.pop(-1)
                mod = self
                for item in atoms:
                    mod = getattr(mod, item)
                setattr(mod, name, module)

            nn.Module.set_submodule = _set_submodule

        assert isinstance(submodule, (nn.Linear, nn.Embedding, nn.Conv1d))
        target_params[comp_name] = submodule.weight

        assert component_params[comp_name].shape == target_params[comp_name].shape

    faithfulness_loss = torch.tensor(0.0, device=device)
    for name in component_params:
        faithfulness_loss = (
            faithfulness_loss + ((target_params[name] - component_params[name]) ** 2).sum()
        )
    return faithfulness_loss / n_params


losses_module.calc_faithfulness_loss = _patched_calc_faithfulness_loss
run_spd_module.calc_faithfulness_loss = _patched_calc_faithfulness_loss

import spd.utils as utils_module

import logging

logger = logging.getLogger(__name__)
_orig_calc_kl_divergence_lm = utils_module.calc_kl_divergence_lm


def _patched_calc_kl_divergence_lm(pred, target):
    if isinstance(pred, tuple):
        pred = pred[0]
    if isinstance(target, tuple):
        target = target[0]
    return _orig_calc_kl_divergence_lm(pred, target)


utils_module.calc_kl_divergence_lm = _patched_calc_kl_divergence_lm
losses_module.calc_kl_divergence_lm = _patched_calc_kl_divergence_lm
run_spd_module.calc_kl_divergence_lm = _patched_calc_kl_divergence_lm

# --- End 1D Convolution Patch ---

if __name__ == "__main__":
    main()
