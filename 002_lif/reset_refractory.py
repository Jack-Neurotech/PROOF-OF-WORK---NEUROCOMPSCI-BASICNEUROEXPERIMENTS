import sys
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.lif import LIFParameters, simulate_lif

# ============================================================
# EXPERIMENT 002C — RESET AND REFRACTORY PERIOD
# ============================================================

parameters = LIFParameters()

duration_ms = 200.0
dt_ms = 0.1
current_nA = 0.20

time_ms = np.arange(0.0, duration_ms, dt_ms)
current = np.full_like(time_ms, current_nA)

voltage, spike_times_ms = simulate_lif(
    current_nA=current,
    dt_ms=dt_ms,
    parameters=parameters,
)

print("=" * 70)
print("EXPERIMENT 002C — RESET AND REFRACTORY PERIOD")
print("=" * 70)

print()
print("MODEL")
print("-" * 70)
print(f"Injected current:       {current_nA:.3f} nA")
print(f"Threshold:               {parameters.threshold_mv:.3f} mV")
print(f"Reset potential:         {parameters.reset_potential_mv:.3f} mV")
print(f"Refractory period:       {parameters.refractory_period_ms:.3f} ms")
print(f"Time step:               {dt_ms:.3f} ms")

print()
print("SPIKE RESULTS")
print("-" * 70)
print(f"Spike count:              {len(spike_times_ms)}")

if len(spike_times_ms) > 0:
    print(f"First spike:              {spike_times_ms[0]:.3f} ms")

if len(spike_times_ms) > 1:
    isi_ms = np.diff(spike_times_ms)

    print(f"Minimum inter-spike interval: {np.min(isi_ms):.3f} ms")
    print(f"Mean inter-spike interval:    {np.mean(isi_ms):.3f} ms")
    print(f"Maximum inter-spike interval: {np.max(isi_ms):.3f} ms")
else:
    isi_ms = np.array([])

print()
print("RESET VALIDATION")
print("-" * 70)

reset_errors = []

for spike_time in spike_times_ms:
    spike_index = int(round(spike_time / dt_ms))

    if spike_index < len(voltage):
        reset_voltage = voltage[spike_index]
        error = abs(reset_voltage - parameters.reset_potential_mv)
        reset_errors.append(error)

        print(
            f"Spike at {spike_time:8.3f} ms | "
            f"reset voltage = {reset_voltage:8.3f} mV | "
            f"error = {error:.6f} mV"
        )

print()
print("VALIDATION")
print("-" * 70)

checks_passed = True

if len(spike_times_ms) == 0:
    print("FAIL: No spikes were generated.")
    checks_passed = False
else:
    print("PASS: Spikes were generated.")

if reset_errors and max(reset_errors) < 1e-9:
    print("PASS: Voltage resets exactly to reset potential.")
else:
    print("FAIL: Reset potential validation failed.")
    checks_passed = False

if len(spike_times_ms) > 1:
    min_isi = np.min(isi_ms)

    if min_isi >= parameters.refractory_period_ms:
        print(
            "PASS: Inter-spike interval respects the "
            "refractory-period constraint."
        )
    else:
        print("FAIL: Refractory-period constraint violated.")
        checks_passed = False

print()
if checks_passed:
    print("OVERALL RESULT: PASS")
else:
    print("OVERALL RESULT: FAIL")

# ============================================================
# SAVE DATA
# ============================================================

data_dir = Path("results/data")
figure_dir = Path("results/figures")

data_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)

output_csv = data_dir / "002C_lif_reset_refractory.csv"

with output_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "time_ms",
            "current_nA",
            "voltage_mv",
            "spike",
        ]
    )

    spike_set = set(np.round(spike_times_ms, 10))

    for t, current_value, voltage_value in zip(
        time_ms,
        current,
        voltage,
    ):
        spike = int(round(t, 10) in spike_set)

        writer.writerow(
            [
                f"{t:.6f}",
                f"{current_value:.6f}",
                f"{voltage_value:.6f}",
                spike,
            ]
        )

# ============================================================
# FIGURE
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time_ms,
    voltage,
    label="Membrane potential",
)

plt.axhline(
    parameters.threshold_mv,
    linestyle="--",
    label="Threshold",
)

plt.axhline(
    parameters.reset_potential_mv,
    linestyle=":",
    label="Reset potential",
)

for spike_time in spike_times_ms:
    plt.axvline(
        spike_time,
        linestyle="--",
        alpha=0.4,
    )

plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("002C — LIF Reset and Refractory Period")
plt.legend()
plt.tight_layout()

figure_path = figure_dir / "002C_lif_reset_refractory.png"
plt.savefig(figure_path, dpi=150)
plt.close()

print()
print("OUTPUTS")
print("-" * 70)
print(f"Data:   {output_csv}")
print(f"Figure: {figure_path}")
print("=" * 70)
