# Codebase Topology Analysis for `/Users/ry/gh/stable-basin/src`

This report ranks the modules by their interconnectedness, helping to identify core 'Hubs' (highly imported), heavy 'Consumers' (importing many things), and isolated code.

## Top 20 Hubs (High In-Degree for `src`)
Modules that provide core value and are depended upon by many others.

| Module | In-Degree (Times Imported) |
|---|---|
| `src` | 9 |
| `src.config` | 5 |
| `src.models.ssm.physics` | 3 |
| `src.data` | 2 |
| `src.harness` | 2 |
| `src.harness.sensor_fusion_predictor` | 2 |
| `src.metrics` | 2 |
| `src.metrics.attribution_engine` | 2 |
| `src.utils` | 2 |
| `src.utils.device` | 2 |
| `src.data.ephys` | 1 |
| `src.data.ephys.pharma_shock_dataset` | 1 |
| `src.harness.trainer` | 1 |
| `src.metrics.diagnostic_engine` | 1 |
| `src.metrics.mamba_lrp` | 1 |
| `src.metrics.metrics` | 1 |
| `src.models` | 1 |
| `src.models.attention` | 1 |
| `src.models.attention.baseline_transformer` | 1 |
| `src.models.encoders` | 1 |

## Top 20 Consumers (High Out-Degree)
Modules that orchestrate or combine many different components (more bug-prone).

| Module | Out-Degree (Imports Made) |
|---|---|
| `src.harness.clinical_diagnostic_runner` | 26 |
| `src.harness.sensor_fusion_sweep` | 23 |
| `src.harness.sensor_fusion_predictor` | 15 |
| `src.metrics.metrics` | 8 |
| `src.data.sim2real.phase_structure_dataloader` | 7 |
| `src.data.sim2real.human_telemetry_dataloader` | 6 |
| `src.data.waddington_dataset` | 6 |
| `src.models.losses.meld_loss` | 6 |
| `src.data.ephys.pharma_shock_dataset` | 5 |
| `src.metrics.diagnostic_engine` | 5 |
| `src.models.encoders.topo_encoder` | 5 |
| `src.data.sim2real.bioelectric_dataloader` | 4 |
| `src.data.sim2real.epigenetic_entropy_dataloader` | 4 |
| `src.harness.trainer` | 4 |
| `src.models.encoders.gevi_encoder` | 4 |
| `src.models.encoders.spatial_compressor` | 4 |
| `src.models.ssm.masr_mamba` | 4 |
| `src.data.sim2real.gevi_dataloader` | 3 |
| `src.data.sim2real.multimodal_bio_dataloader` | 3 |
| `src.data.sim2real.neocortical_assembloid_dataloader` | 3 |

## Entry Points
Known entry points and runner scripts that are not imported by other internal modules.

| Module |
|---|
| `src.demo` |
| `src.harness.clinical_diagnostic_runner` |
| `src.harness.sensor_fusion_sweep` |
| `src.harness.summarize_results` |

## Isolated / Dead Code (In-Degree = 0)
Internal modules that are NEVER imported by anything else in the package. These are prime candidates for deletion or iceboxing (unless they are top-level entry point scripts).

| Module |
|---|
| `src.data.sim2real` |
| `src.data.sim2real.bioelectric_dataloader` |
| `src.data.sim2real.epigenetic_entropy_dataloader` |
| `src.data.sim2real.gevi_dataloader` |
| `src.data.sim2real.human_telemetry_dataloader` |
| `src.data.sim2real.multimodal_bio_builder` |
| `src.data.sim2real.multimodal_bio_dataloader` |
| `src.data.sim2real.neocortical_assembloid_dataloader` |
| `src.data.sim2real.phase_structure_dataloader` |
| `src.data.sim2real.rna_dataloader` |
| `src.models.encoders.gevi_encoder` |
| `src.models.encoders.spatial_compressor` |
| `src.models.encoders.topo_encoder` |
| `src.models.losses` |
| `src.models.losses.meld_loss` |
| `src.models.ssm.triton_fused_scan` |
