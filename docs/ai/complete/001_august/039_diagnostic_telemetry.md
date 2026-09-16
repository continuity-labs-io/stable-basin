Context: We are debugging the continuous-time Mamba-2 engine and the PyDMD
metrics in `src/demo/8_ephys_demo.py`. The visual dashboard is obscuring
potential scaling or mathematical collapse issues in the underlying tensors. We
need to implement a rigorous, programmatic diagnostic logging system to audit
the math directly in the terminal output.

Task: Implement a `DiagnosticLogger` in `8_ephys_demo.py`.

1. At the start of the script, set up Python's `logging` module to output to
   both the console (INFO) and a detailed log file named
   `output/ephys_diagnostic.log` (DEBUG).
2. **Data Ingestion Audit:** After the `.brw` dataset is loaded and the first
   `batch` tensor is extracted, log the exact `shape`, `dtype`, `min()`,
   `max()`, `mean()`, and `std()` of the raw tensor to prove the Z-score
   normalization worked and the data is not a flatline.
3. **Burn-in Audit:** During the 10-iteration training loop, log not just the
   total loss, but the exact gradient norm (`torch.nn.utils.clip_grad_norm_`
   returns this) to monitor if the model is exploding or vanishing.
4. **The Crash Audit:** After the Waddington crash is injected (T=5000), log the
   `mean` and `std` of the tensor _before_ the crash (e.g., frames 4000-4999)
   versus _after_ the crash (frames 5000-5999) to verify that the mathematical
   "flatline" was injected correctly.
5. **PyDMD Eigenvalue Audit:** Modify `ThermodynamicMetrics.calculate_ksm`
   temporarily (or add a debug flag) to log the exact complex eigenvalues
   (`dmd.eigs`) and the extracted `max_eig` magnitude for the specific sliding
   window immediately _before_ the crash and the sliding window immediately
   _after_ the crash. We need to prove PyDMD is actually calculating rank
   collapse.
6. **LRP Audit:** Log the total sum of the `relevance_tensor` output by
   `MambaLRPEpsilon` to verify that relevance is actively being conserved across
   the sequence and not just dropping to zero.

Ensure all logging uses `logger.info()` or `logger.debug()` in accordance with
`002_logging_standards.md`.
