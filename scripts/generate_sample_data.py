"""
IntelliPQ - Sample Data Generator
Generates sample CSV files for testing file upload in the Streamlit application.
Uses only Python standard library so it runs in any environment.
"""

import os
import math
import csv
import random


def generate_samples(output_dir: str = "data/sample_waveforms"):
    os.makedirs(output_dir, exist_ok=True)
    timesteps = 1000
    duration = 1.0
    dt = duration / timesteps
    freq = 50.0

    def write_csv(filename, values):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_seconds", "voltage_pu"])
            for idx, val in enumerate(values):
                t = round(idx * dt, 4)
                writer.writerow([t, round(val, 6)])

    # 1. Normal Sample (Amplitude ~ 1.0 pu, 50 Hz, low noise)
    norm_vals = []
    for i in range(timesteps):
        t = i * dt
        noise = random.gauss(0, 0.005)
        norm_vals.append(1.0 * math.sin(2 * math.pi * freq * t) + noise)
    write_csv("normal_sample.csv", norm_vals)

    # 2. Voltage Sag Sample (Amplitude drops to 0.65 pu between 0.2s and 0.8s)
    sag_vals = []
    for i in range(timesteps):
        t = i * dt
        amp = 0.65 if (0.2 <= t <= 0.8) else 1.0
        noise = random.gauss(0, 0.005)
        sag_vals.append(amp * math.sin(2 * math.pi * freq * t) + noise)
    write_csv("voltage_sag_sample.csv", sag_vals)

    # 3. Voltage Swell Sample (Amplitude rises to 1.35 pu between 0.2s and 0.8s)
    swell_vals = []
    for i in range(timesteps):
        t = i * dt
        amp = 1.35 if (0.2 <= t <= 0.8) else 1.0
        noise = random.gauss(0, 0.005)
        swell_vals.append(amp * math.sin(2 * math.pi * freq * t) + noise)
    write_csv("voltage_swell_sample.csv", swell_vals)

    # 4. Harmonic Distortion Sample (3rd and 5th harmonic injected)
    harm_vals = []
    for i in range(timesteps):
        t = i * dt
        v = (
            1.0 * math.sin(2 * math.pi * freq * t)
            + 0.25 * math.sin(2 * math.pi * 3 * freq * t)
            + 0.12 * math.sin(2 * math.pi * 5 * freq * t)
        )
        noise = random.gauss(0, 0.005)
        harm_vals.append(v + noise)
    write_csv("harmonic_distortion_sample.csv", harm_vals)

    print(f"[✔] Successfully generated 4 sample CSV files in {output_dir}")


if __name__ == "__main__":
    generate_samples()
