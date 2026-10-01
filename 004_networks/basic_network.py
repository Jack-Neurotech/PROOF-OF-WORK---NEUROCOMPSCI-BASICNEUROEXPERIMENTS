from __future__ import annotations

import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

# ============================================================
# IMPORTS
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

from models.lif import LIFParameters
from models.lif_network import (
    LIFNetworkParameters,
    simulate_lif_network,
)

def main():

    duration_ms = 500.0
    dt_ms = 0.1
    n_neurons = 5

    time_ms = np.arange(
        0.0,
        duration_ms + dt_ms,
        dt_ms,
    )

    current_nA = np.zeros((n_neurons, len(time_ms)))

    # External stimulus applied only to Neuron 1.
    current_nA[0, :] = 0.20

    # Rows = presynaptic neurons.
    # Columns = postsynaptic neurons.
    weights_nA = np.zeros((n_neurons, n_neurons))

    weights_nA[0, 1] = 0.08
    weights_nA[0, 2] = 0.08
    weights_nA[1, 3] = 0.08
    weights_nA[2, 3] = 0.08
    weights_nA[3, 4] = 0.08

    parameters = LIFNetworkParameters(
        neuron_parameters=LIFParameters(
            tau_ms=20.0,
            resistance_mohm=100.0,
            resting_potential_mv=-65.0,
            reset_potential_mv=-65.0,
            threshold_mv=-50.0,
            refractory_period_ms=2.0,
        ),
        synaptic_tau_ms=5.0,
    )

    voltage_mv, spike_times, synaptic_current_nA = (
        simulate_lif_network(
            current_nA=current_nA,
            weights_nA=weights_nA,
            dt_ms=dt_ms,
            parameters=parameters,
        )
    )

    print("=" * 78)
    print("EXPERIMENT 004A — MULTI-NEURON NETWORK")
    print("=" * 78)

    print()
    print("NETWORK")
    print("-" * 78)
    print(f"Neurons:                 {n_neurons}")
    print(f"Duration:                {duration_ms:.1f} ms")
    print(f"Time step:               {dt_ms:.3f} ms")
    print("External stimulus:       Neuron 1")
    print("Connectivity:            excitatory feed-forward network")

    print()
    print("CONNECTIONS")
    print("-" * 78)

    for source in range(n_neurons):
        for target in range(n_neurons):
            if weights_nA[source, target] != 0:
                print(
                    f"Neuron {source + 1} -> "
                    f"Neuron {target + 1}: "
                    f"{weights_nA[source, target]:.3f} nA"
                )

    print()
    print("SPIKE ACTIVITY")
    print("-" * 78)
    print(f"{'NEURON':<12}{'SPIKES':>10}{'RATE (Hz)':>15}")

    for neuron in range(n_neurons):
        count = len(spike_times[neuron])
        rate = count / (duration_ms / 1000.0)

        print(
            f"{neuron + 1:<12}"
            f"{count:>10}"
            f"{rate:>15.3f}"
        )

    print()
    print("SPIKE TIMES")
    print("-" * 78)

    for neuron in range(n_neurons):
        times = spike_times[neuron]

        if len(times) == 0:
            print(f"Neuron {neuron + 1}: no spikes")
        else:
            formatted = ", ".join(
                f"{t:.1f}" for t in times
            )
            print(
                f"Neuron {neuron + 1}: "
                f"{formatted} ms"
            )

    print()
    print("NETWORK PROPAGATION")
    print("-" * 78)

    active_neurons = [
        neuron + 1
        for neuron in range(n_neurons)
        if len(spike_times[neuron]) > 0
    ]

    print(
        "Neurons participating in activity: "
        + ", ".join(map(str, active_neurons))
    )

    print()
    print("VALIDATION")
    print("-" * 78)

    checks = []

    checks.append(
        len(spike_times[0]) > 0
    )

    checks.append(
        np.max(synaptic_current_nA[1:]) > 0
    )

    checks.append(
        any(len(spike_times[i]) > 0 for i in range(1, n_neurons))
    )

    if checks[0]:
        print("PASS: Stimulated Neuron 1 generated spikes.")
    else:
        print("FAIL: Stimulated Neuron 1 did not spike.")

    if checks[1]:
        print("PASS: Synaptic currents propagated through the network.")
    else:
        print("FAIL: No synaptic current propagated.")

    if checks[2]:
        print("PASS: Network activity reached downstream neurons.")
    else:
        print("FAIL: No downstream neuron generated spikes.")

    overall = all(checks)

    print(
        f"OVERALL RESULT: {'PASS' if overall else 'FAIL'}"
    )

    # ------------------------------------------------------------------
    # Save numerical data
    # ------------------------------------------------------------------

    data_dir = Path("results/data")
    figure_dir = Path("results/figures")

    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "time_ms": time_ms,
    }

    for neuron in range(n_neurons):
        data[f"neuron_{neuron + 1}_voltage_mv"] = (
            voltage_mv[neuron]
        )

        data[f"neuron_{neuron + 1}_synaptic_current_nA"] = (
            synaptic_current_nA[neuron]
        )

    output_data = data_dir / "004A_multi_neuron_network.csv"

    import pandas as pd

    pd.DataFrame(data).to_csv(
        output_data,
        index=False,
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
    )

    for neuron in range(n_neurons):
        axes[0].plot(
            time_ms,
            voltage_mv[neuron],
            label=f"Neuron {neuron + 1}",
        )

    axes[0].axhline(
        parameters.neuron_parameters.threshold_mv,
        linestyle="--",
        label="Threshold",
    )

    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title(
        "004A — Multi-Neuron Network Membrane Potentials"
    )
    axes[0].legend()

    for neuron in range(n_neurons):
        for spike_time in spike_times[neuron]:
            axes[1].vlines(
                spike_time,
                neuron + 0.6,
                neuron + 1.4,
            )

    axes[1].set_yticks(
        range(1, n_neurons + 1)
    )

    axes[1].set_yticklabels(
        [f"Neuron {i}" for i in range(1, n_neurons + 1)]
    )

    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Spikes")
    axes[1].set_title("Network Spike Raster")

    figure_path = figure_dir / "004A_multi_neuron_network.png"

    fig.tight_layout()
    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print()
    print("=" * 78)
    print("OUTPUTS")
    print("=" * 78)
    print(f"Data:   {output_data}")
    print(f"Figure: {figure_path}")
    print("Figure resolution: 300 DPI")
    print("=" * 78)


if __name__ == "__main__":
    main()
