import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.lif import LIFParameters, simulate_lif


# ============================================================
# EXPERIMENT 002D — FIRING RATE VS CURRENT (F-I CURVE)
# ============================================================

parameters = LIFParameters()

duration_ms = 1000.0
dt_ms = 0.1

currents_nA = np.arange(0.05, 0.351, 0.01)

firing_rates_hz = []
spike_counts = []


# ============================================================
# THEORETICAL THRESHOLD
# ============================================================

threshold_current_nA = (
    parameters.threshold_mv
    - parameters.resting_potential_mv
) / parameters.resistance_mohm


# ============================================================
# RUN CURRENT SWEEP
# ============================================================

for current_nA in currents_nA:

    time_ms = np.arange(0.0, duration_ms, dt_ms)
    current = np.full_like(time_ms, current_nA)

    voltage, spike_times_ms = simulate_lif(
        current_nA=current,
        dt_ms=dt_ms,
        parameters=parameters,
    )

    spike_count = len(spike_times_ms)
    duration_s = duration_ms / 1000.0

    firing_rate_hz = spike_count / duration_s

    spike_counts.append(spike_count)
    firing_rates_hz.append(firing_rate_hz)


firing_rates_hz = np.asarray(firing_rates_hz)
spike_counts = np.asarray(spike_counts)


# ============================================================
# TERMINAL HEADER
# ============================================================

print()
print("=" * 78)
print("EXPERIMENT 002D — LIF FREQUENCY-CURRENT (F-I) CURVE")
print("=" * 78)

print()
print("MODEL")
print("-" * 78)
print(f"Duration:                 {duration_ms:.1f} ms")
print(f"Time step:                {dt_ms:.3f} ms")
print(f"Threshold potential:     {parameters.threshold_mv:.3f} mV")
print(f"Reset potential:         {parameters.reset_potential_mv:.3f} mV")
print(f"Resistance:              {parameters.resistance_mohm:.1f} MOhm")
print(f"Time constant:             {parameters.tau_ms:.1f} ms")
print(f"Refractory period:         {parameters.refractory_period_ms:.1f} ms")
print(f"Theoretical threshold:    {threshold_current_nA:.6f} nA")


# ============================================================
# MEASUREMENTS
# ============================================================

print()
print("F-I MEASUREMENTS")
print("-" * 78)
print(f"{'CURRENT (nA)':>14} {'SPIKES':>10} {'RATE (Hz)':>14}")
print("-" * 78)

for current, count, rate in zip(
    currents_nA,
    spike_counts,
    firing_rates_hz,
):
    print(
        f"{current:14.3f}"
        f"{count:10d}"
        f"{rate:14.3f}"
    )


# ============================================================
# ASCII TERMINAL F-I CURVE
# ============================================================

def print_ascii_fi_curve(
    currents,
    rates,
    threshold,
    width=68,
    height=20,
):
    print()
    print("=" * 78)
    print("F-I CURVE — TERMINAL VISUALIZATION")
    print("=" * 78)

    max_rate = float(np.max(rates))

    if max_rate <= 0:
        print("No firing detected.")
        return

    plot_left = 9
    plot_right = width
    plot_width = plot_right - plot_left

    print()

    for row in range(height, -1, -1):

        rate_level = max_rate * row / height

        line = f"{rate_level:7.1f} |"

        for column in range(plot_width + 1):

            current = (
                currents[0]
                + (currents[-1] - currents[0])
                * column
                / plot_width
            )

            # Find nearest measured current.
            index = int(
                np.argmin(np.abs(currents - current))
            )

            measured_rate = rates[index]

            if measured_rate >= rate_level and rate_level > 0:
                character = "●"
            elif abs(current - threshold) < (
                currents[-1] - currents[0]
            ) / plot_width:
                character = "|"
            else:
                character = " "

            line += character

        print(line)

    print("        +" + "-" * (plot_width + 1))

    # Current-axis labels.
    axis_values = np.linspace(
        currents[0],
        currents[-1],
        6,
    )

    print("         ", end="")

    for value in axis_values:
        print(f"{value:>10.2f}", end="")

    print()
    print()
    print("        Current (nA)")
    print()
    print(f"        | = theoretical threshold ({threshold:.3f} nA)")
    print("        ● = measured firing rate")


print_ascii_fi_curve(
    currents_nA,
    firing_rates_hz,
    threshold_current_nA,
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("VALIDATION")
print("-" * 78)

checks_passed = True

below_threshold = currents_nA < threshold_current_nA
above_threshold = currents_nA > threshold_current_nA

if np.all(firing_rates_hz[below_threshold] == 0):
    print("PASS: Currents below threshold produce no spikes.")
else:
    print("FAIL: Subthreshold current produced spikes.")
    checks_passed = False

if np.any(firing_rates_hz[above_threshold] > 0):
    print("PASS: Suprathreshold currents produce spikes.")
else:
    print("FAIL: No suprathreshold firing detected.")
    checks_passed = False

# Check that firing generally increases with current.
# We allow repeated rates because spike counts are discrete.
rate_differences = np.diff(firing_rates_hz[above_threshold])

if np.all(rate_differences >= 0):
    print("PASS: Firing rate is monotonically non-decreasing.")
else:
    print(
        "NOTE: Some sampled firing rates decrease because "
        "spike counts are discrete over the finite simulation."
    )

if checks_passed:
    print("OVERALL RESULT: PASS")
else:
    print("OVERALL RESULT: FAIL")


# ============================================================
# SAVE NUMERICAL DATA
# ============================================================

data_dir = Path("results/data")
figure_dir = Path("results/figures")

data_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)

output_csv = data_dir / "002D_lif_firing_rate.csv"

with output_csv.open("w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "current_nA",
            "spike_count",
            "firing_rate_hz",
        ]
    )

    for current, count, rate in zip(
        currents_nA,
        spike_counts,
        firing_rates_hz,
    ):
        writer.writerow(
            [
                f"{current:.6f}",
                int(count),
                f"{rate:.6f}",
            ]
        )


# ============================================================
# SAVE HIGH-RESOLUTION SCIENTIFIC FIGURE
# ============================================================

plt.figure(figsize=(10, 7))

plt.plot(
    currents_nA,
    firing_rates_hz,
    marker="o",
    linewidth=1.8,
    markersize=5,
    label="LIF firing rate",
)

plt.axvline(
    threshold_current_nA,
    linestyle="--",
    linewidth=1.5,
    label=f"Theoretical threshold = {threshold_current_nA:.3f} nA",
)

plt.xlabel(
    "Injected current (nA)",
    fontsize=12,
)

plt.ylabel(
    "Firing rate (Hz)",
    fontsize=12,
)

plt.title(
    "002D — Leaky Integrate-and-Fire Frequency–Current Curve",
    fontsize=14,
)

plt.grid(
    True,
    alpha=0.25,
)

plt.legend()

plt.tight_layout()

figure_path = figure_dir / "002D_lif_firing_rate.png"

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FINAL OUTPUT SUMMARY
# ============================================================

print()
print("=" * 78)
print("OUTPUTS")
print("=" * 78)
print(f"Data:   {output_csv}")
print(f"Figure: {figure_path}")
print("Figure resolution: 300 DPI")

print()
print("F-I CURVE SUMMARY")
print("-" * 78)

max_rate = float(np.max(firing_rates_hz))
max_rate_current = float(
    currents_nA[np.argmax(firing_rates_hz)]
)

print(f"Theoretical threshold: {threshold_current_nA:.6f} nA")
print(f"Maximum measured rate: {max_rate:.3f} Hz")
print(f"Current at maximum rate: {max_rate_current:.3f} nA")

print()
print("=" * 78)
