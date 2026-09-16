Context: We need to build a visual proof-of-concept demonstrating why standard
OS software clocks destroy continuous biological causality, necessitating our
IEEE 1588 PTP hardware-sync strategy.

Task: Create a Python script `src/modules/ptp_sync/01_jitter_simulator.py`.

1. Use the `threading` module to simulate two independent biological sensors
   running concurrently over a 0.5-second window.
2. Sensor A (Optical): A thread generating a continuous 100Hz smooth sine wave.
3. Sensor B (Electrical): A thread generating a 20,000Hz baseline with sharp,
   periodic action potential spikes injected exactly every 0.1 seconds.
4. The Hack (OS Jitter): Both threads must use `time.time()` to timestamp their
   data points as they generate them, simulating how standard user-space
   software records data. Add random `time.sleep()` micro-delays (e.g., 1-5
   milliseconds) to the threads to simulate unpredictable OS scheduler
   preemption and CPU load.
5. In the main thread, collect the data and align the two signals on a shared
   timeline based on their recorded `time.time()` timestamps.
6. Generate a 2-panel Matplotlib dashboard (save to
   `output/10_ptp_jitter_demo.png`).
   - Top Panel: The Ground Truth (how the signals actually align in physical
     time).
   - Bottom Panel: The Software Clock Reality. Plot the signals aligned by their
     `time.time()` stamps. It should visually demonstrate the 20kHz spike
     shifting chaotically before and after the 100Hz optical wave due to OS
     jitter, proving that true causality (dx/dt) is destroyed.
