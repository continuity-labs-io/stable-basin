import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import os

# Physical parameters
DURATION = 0.5
OPTICAL_FREQ = 100
ELEC_SAMPLE_RATE = 20000
OPTICAL_SAMPLE_RATE = 1000
SPIKE_INTERVAL = 0.1

sensor_a_data = []  # (phys_t, sw_t, val)
sensor_b_data = []  # (phys_t, sw_t, val)


def generate_sensor_a():
    """Simulates a 100Hz optical sensor."""
    chunk_size = 5  # Generate in chunks (5ms)
    interval = 1.0 / OPTICAL_SAMPLE_RATE
    num_samples = int(DURATION * OPTICAL_SAMPLE_RATE)

    start_time_sw = time.time()

    for i in range(0, num_samples, chunk_size):
        chunk_samples = min(chunk_size, num_samples - i)
        phys_t = (i + np.arange(chunk_samples)) * interval
        val = np.sin(2 * np.pi * OPTICAL_FREQ * phys_t)

        # Simulate OS scheduler preemption/CPU load
        time.sleep(np.random.uniform(0.001, 0.005))

        sw_t = time.time() - start_time_sw
        # Interpolate software timestamps for the chunk assuming uniform sampling within the chunk
        sw_t_arr = sw_t - np.flip(np.arange(chunk_samples) * interval)

        for j in range(chunk_samples):
            sensor_a_data.append((phys_t[j], sw_t_arr[j], val[j]))


def generate_sensor_b():
    """Simulates a 20kHz electrical sensor with periodic spikes."""
    chunk_size = 100  # Generate in chunks (5ms)
    interval = 1.0 / ELEC_SAMPLE_RATE
    num_samples = int(DURATION * ELEC_SAMPLE_RATE)

    start_time_sw = time.time()

    for i in range(0, num_samples, chunk_size):
        chunk_samples = min(chunk_size, num_samples - i)
        phys_t = (i + np.arange(chunk_samples)) * interval
        val = np.random.normal(0, 0.1, chunk_samples)  # 20kHz baseline

        for j in range(chunk_samples):
            # Inject sharp spike every SPIKE_INTERVAL
            if (i + j) % int(SPIKE_INTERVAL * ELEC_SAMPLE_RATE) == 0:
                val[j] = 5.0

        # Simulate OS scheduler preemption/CPU load
        time.sleep(np.random.uniform(0.001, 0.005))

        sw_t = time.time() - start_time_sw
        sw_t_arr = sw_t - np.flip(np.arange(chunk_samples) * interval)

        for j in range(chunk_samples):
            sensor_b_data.append((phys_t[j], sw_t_arr[j], val[j]))


def main():
    print("Starting simulation of independent biological sensors...")

    # Run threads concurrently
    t1 = threading.Thread(target=generate_sensor_a)
    t2 = threading.Thread(target=generate_sensor_b)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Simulation complete. Generating visual proof-of-concept...")

    # Sort data in case threads appended out of strict physical time order
    # (though within each thread it's ordered, but just to be safe for unpacking)
    sensor_a_data.sort(key=lambda x: x[0])
    sensor_b_data.sort(key=lambda x: x[0])

    phys_t_a, sw_t_a, val_a = zip(*sensor_a_data, strict=False)
    phys_t_b, sw_t_b, val_b = zip(*sensor_b_data, strict=False)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # --- Top Panel: Ground Truth ---
    axs[0].plot(phys_t_a, val_a, label="Sensor A (100Hz Smooth Sine)", color="blue", linewidth=1.5)
    axs[0].plot(
        phys_t_b,
        val_b,
        label="Sensor B (20kHz with Action Potentials)",
        color="red",
        alpha=0.7,
        linewidth=1.0,
    )
    axs[0].set_title("Ground Truth: Continuous Biological Causality (Physical Time)", fontsize=14)
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlabel("Time (s)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, linestyle="--", alpha=0.6)
    axs[0].set_xlim(0, DURATION)

    # Highlight the alignment in Ground Truth
    for t in np.arange(0, DURATION, SPIKE_INTERVAL):
        axs[0].axvline(x=t, color="green", linestyle=":", alpha=0.5)

    # --- Bottom Panel: Software Clock Reality ---
    axs[1].plot(sw_t_a, val_a, label="Sensor A (Software Clock)", color="blue", linewidth=1.5)
    axs[1].plot(
        sw_t_b, val_b, label="Sensor B (Software Clock)", color="red", alpha=0.7, linewidth=1.0
    )
    axs[1].set_title(
        "Software Clock Reality: Jitter Destroys Causality (time.time() Alignment)", fontsize=14
    )
    axs[1].set_ylabel("Amplitude")
    axs[1].set_xlabel("Time (s)")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle="--", alpha=0.6)
    axs[1].set_xlim(0, DURATION)

    # Highlight the misalignment in Software Clock
    # Spikes might shift relative to the sine wave
    for t in np.arange(0, DURATION, SPIKE_INTERVAL):
        axs[1].axvline(x=t, color="green", linestyle=":", alpha=0.5)

    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    out_path = "output/10_ptp_jitter_demo.png"
    plt.savefig(out_path, dpi=150)
    print(f"Successfully generated dashboard: {out_path}")


if __name__ == "__main__":
    main()
