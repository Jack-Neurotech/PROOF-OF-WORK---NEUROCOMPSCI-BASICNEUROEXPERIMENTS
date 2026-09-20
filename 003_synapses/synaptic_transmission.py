import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models.lif import LIFParameters, simulate_lif
from models.synapses.exponential import (
    ExponentialSynapseParameters,
    exponential_synaptic_current,
)


def main() -> None:
    duration_ms = 200.0
    dt_ms = 0.1

    time_ms = np.arange(
        0.0,
        duration_ms + dt_ms,
        dt_ms,
    )

    lif_parameters = LIFParameters()

    synapse_parameters = ExponentialSynapseParameters(
        peak_current_nA=0.05,
        time_constant_ms=5.0,
    )

    presynaptic_spike_time_ms = 50.0

    synaptic_current_nA = exponential_synaptic_current(
        time_ms=time_ms,
        spike_time_ms=presynaptic_spike_time_ms,
        parameters=synapse_parameters,
    )

    postsynaptic_voltage_mv, postsynaptic_spikes_ms = simulate_lif(
        current_nA=synaptic_current_nA,
        dt_ms=dt_ms,
        parameters=lif_parameters,
    )

    peak_current_nA = float(np.max(synaptic_current_nA))
    peak_voltage_mv = float(np.max(postsynaptic_voltage_mv))

    baseline_voltage_mv = float(postsynaptic_voltage_mv[0])

    voltage_change_mv = peak_voltage_mv - baseline_voltage_mv

    print("=" * 78)
    print("EXPERIMENT 003A — SYNAPTIC TRANSMISSION")
    print("=" * 78)

    print()
    print("SYSTEM")
    print("-" * 78)
    print("Presynaptic neuron:   Neuron A")
    print("Postsynaptic neuron:  Neuron B")
    print(f"Presynaptic spike:    {presynaptic_spike_time_ms:.1f} ms")
    print(f"Peak synaptic current:{peak_current_nA:.6f} nA")
    print(
        f"Synaptic time constant: "
        f"{synapse_parameters.time_constant_ms:.3f} ms"
    )

    print()
    print("POSTSYNAPTIC RESPONSE")
    print("-" * 78)
    print(f"Baseline voltage:     {baseline_voltage_mv:.6f} mV")
    print(f"Peak voltage:         {peak_voltage_mv:.6f} mV")
    print(f"Voltage change:       {voltage_change_mv:.6f} mV")
    print(
        "Postsynaptic spikes:  "
        f"{len(postsynaptic_spikes_ms)}"
    )

    print()
    print("VALIDATION")
    print("-" * 78)

    passed = True

    if peak_current_nA > 0:
        print("PASS: Presynaptic spike generated synaptic current.")
    else:
        print("FAIL: No synaptic current generated.")
        passed = False

    if peak_voltage_mv > baseline_voltage_mv:
        print("PASS: Synaptic input depolarized Neuron B.")
    else:
        print("FAIL: Neuron B did not depolarize.")
        passed = False

    if len(postsynaptic_spikes_ms) == 0:
        print(
            "PASS: Synaptic input remained subthreshold "
            "for this initial experiment."
        )
    else:
        print(
            "INFO: Synaptic input generated "
            "postsynaptic spike(s)."
        )

    print(
        f"OVERALL RESULT: {'PASS' if passed else 'FAIL'}"
    )

    data_dir = Path("results/data")
    figure_dir = Path("results/figures")

    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    output_csv = data_dir / "003A_synaptic_transmission.csv"

    with output_csv.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "time_ms",
                "synaptic_current_nA",
                "postsynaptic_voltage_mv",
            ]
        )

        for t, current, voltage in zip(
            time_ms,
            synaptic_current_nA,
            postsynaptic_voltage_mv,
        ):
            writer.writerow(
                [
                    f"{t:.6f}",
                    f"{current:.9f}",
                    f"{voltage:.9f}",
                ]
            )

    figure_path = figure_dir / "003A_synaptic_transmission.png"

    plt.figure(figsize=(10, 7))

    plt.plot(
        time_ms,
        postsynaptic_voltage_mv,
        label="Neuron B membrane voltage",
    )

    plt.axvline(
        presynaptic_spike_time_ms,
        linestyle="--",
        label="Neuron A spike",
    )

    plt.axhline(
        lif_parameters.threshold_mv,
        linestyle=":",
        label="Threshold",
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane voltage (mV)")
    plt.title("003A — Postsynaptic Response to Presynaptic Spike")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("=" * 78)
    print("OUTPUTS")
    print("=" * 78)
    print(f"Data:   {output_csv}")
    print(f"Figure: {figure_path}")
    print("Figure resolution: 300 DPI")
    print("=" * 78)


if __name__ == "__main__":
    main()
