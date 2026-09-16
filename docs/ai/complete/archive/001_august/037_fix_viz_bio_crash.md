Context: We need to refine the `8_ephys_demo.py` dashboard. The raw telemetry
currently looks like a solid block of color before the crash, and the PyDMD KSM
metric oscillates instead of collapsing after the crash.

Task: Update `8_ephys_demo.py`.

1. In the "Simulating Waddington Crash" section, change the crash logic. Instead
   of injecting random noise (`torch.randn_like(...) * 10.0`), we want to
   simulate a biological "flatline" (necrosis/cell death). Multiply the tensor
   after `EVENT_FRAME` by a tiny decay factor (e.g., `0.001`). This will cause
   the PyDMD matrix rank to collapse, ensuring the KSM metric plummets and STAYS
   low after the crash.
2. In `plot_ephys_dashboard`, adjust the `vmax` and `vmin` for Panel 1 (Raw
   Telemetry). Calculate the 95th percentile of the absolute values of the
   pre-crash telemetry (`raw_sub[:event_frame]`) and use that as the symmetric
   `vmax` limit for the `imshow` plot. This will prevent the baseline biological
   noise from washing out into a solid color block.
