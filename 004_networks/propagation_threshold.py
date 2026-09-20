from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.lif import LIFParameters
from models.networks.lif_network import (
    LIFNetworkParameters,
    simulate_lif_network,
)


def run_network(weight_nA: float):

    duration_ms = 500.0
    dt_ms = 0.1
    n_neurons = 5

    time_ms = np.arange(
        0.0,
        duration_ms + dt_ms,
        dt_ms,
    )

    current_nA = np.zeros((n_neurons, len(time_ms)))

    # External drive is applied only to Neuron 1.
    current_nA[0, :] = 0.20

    # Feed-forward network:
    #
    #        ┌──> N2 ──┐
    # N1 ────┤         ├──> N4 ──> N5
    #        └──> N3 ──┘

    weights_nA = np.zeros((n_neurons, n_neurons))

    weights_nA[0, 1] = weight_nA
    weights_nA[0, 2] = weight_nA
    weights_nA[1, 3] = weight_nA
    weights_nA[2, 3] = weight_nA
    weights_nA[3, 4] = weight_nA

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

    return simulate_lif_network(
        current_nA=current_nA,
        weights_nA=weights_nA,
        dt_ms=dt_ms,
        parameters=parameters,
    )


def main():

    # Previous experiment stopped at 0.20 nA.
    # That was insufficient to test the actual propagation threshold.
    #
    # We now sweep through 1.00 nA.

    weights = np.arange(0.01, 1.001, 0.01)

    duration_ms = 500.0

    results = []

    print("=" * 78)
    print("EXPERIMENT 004B — NETWORK PROPAGATION THRESHOLD")
    print("=" * 78)

    print()
    print("QUESTION")
    print("-" * 78)
    print(
        "What synaptic strength is required for activity "
        "to propagate through the network?"
    )

    print()
    print("NETWORK")
    print("-" * 78)
    print("Neuron 1 -> Neuron 2")
    print("Neuron 1 -> Neuron 3")
    print("Neuron 2 -> Neuron 4")
    print("Neuron 3 -> Neuron 4")
    print("Neuron 4 -> Neuron 5")

    print()
    print("SWEEP")
    print("-" * 78)
    print("Synaptic strength: 0.01 nA -> 1.00 nA")
    print("Step:              0.01 nA")

    print()
    print("PROPAGATION RESULTS")
    print("-" * 78)

    header = (
        f"{'WEIGHT':>10}"
        f"{'N1':>7}"
        f"{'N2':>7}"
        f"{'N3':>7}"
        f"{'N4':>7}"
        f"{'N5':>7}"
        f"{'TOTAL':>9}"
        f"{'DEPTH':>8}"
    )

    print(header)
    print("-" * 78)

    for weight in weights:

        (
            voltage_mv,
            spike_times,
            synaptic_current_nA,
        ) = run_network(weight)

        counts = [
            len(spike_times[i])
            for i in range(5)
        ]

        active_neurons = [
            i for i, count in enumerate(counts)
            if count > 0
        ]

        depth = (
            max(active_neurons) + 1
            if active_neurons
            else 0
        )

        total_spikes = sum(counts)

        results.append(
            {
                "weight_nA": weight,
                "neuron_1_spikes": counts[0],
                "neuron_2_spikes": counts[1],
                "neuron_3_spikes": counts[2],
                "neuron_4_spikes": counts[3],
                "neuron_5_spikes": counts[4],
                "total_spikes": total_spikes,
                "propagation_depth": depth,
            }
        )

        # Only print every 0.05 nA plus transition points.
        if (
            abs((weight * 100) % 5) < 1e-9
            or (
                len(results) > 1
                and depth != results[-2]["propagation_depth"]
            )
        ):
            print(
                f"{weight:>10.2f}"
                f"{counts[0]:>7}"
                f"{counts[1]:>7}"
                f"{counts[2]:>7}"
                f"{counts[3]:>7}"
                f"{counts[4]:>7}"
                f"{total_spikes:>9}"
                f"{depth:>8}"
            )

    df = pd.DataFrame(results)

    print()
    print("=" * 78)
    print("PROPAGATION THRESHOLDS")
    print("=" * 78)

    # First downstream activity.
    downstream = df[
        df[
            [
                "neuron_2_spikes",
                "neuron_3_spikes",
                "neuron_4_spikes",
                "neuron_5_spikes",
            ]
        ].sum(axis=1) > 0
    ]

    if len(downstream) > 0:
        first = downstream.iloc[0]

        print()
        print(
            f"First downstream activity: "
            f"{first['weight_nA']:.2f} nA"
        )

    else:
        print()
        print("No downstream activity detected up to 1.00 nA.")

    # First firing threshold for each neuron.
    print()

    for neuron in range(2, 6):

        column = f"neuron_{neuron}_spikes"

        found = df[df[column] > 0]

        if len(found) > 0:
            threshold = found.iloc[0]["weight_nA"]

            print(
                f"Neuron {neuron} first fires: "
                f"{threshold:.2f} nA"
            )
        else:
            print(
                f"Neuron {neuron}: "
                "does not fire within tested range"
            )

    # Maximum propagation depth.
    maximum_depth = int(
        df["propagation_depth"].max()
    )

    print()
    print(
        f"Maximum propagation depth: "
        f"{maximum_depth}/5 neurons"
    )

    print()
    print("VALIDATION")
    print("-" * 78)

    neuron_1_active = df["neuron_1_spikes"].gt(0).all()

    if neuron_1_active:
        print(
            "PASS: Stimulated Neuron 1 remains active "
            "throughout the sweep."
        )
    else:
        print(
            "FAIL: Neuron 1 unexpectedly became inactive."
        )

    if len(downstream) > 0:
        print(
            "PASS: Synaptic strength produces "
            "downstream neuronal firing."
        )
    else:
        print(
            "FAIL: No downstream firing detected."
        )

    if maximum_depth >= 5:
        print(
            "PASS: Activity propagated through all "
            "five neurons."
        )
    elif maximum_depth > 1:
        print(
            f"PARTIAL: Activity propagated through "
            f"{maximum_depth} neurons."
        )
    else:
        print(
            "FAIL: Activity remained confined "
            "to Neuron 1."
        )

    # ------------------------------------------------------------------
    # Save data
    # ------------------------------------------------------------------

    data_dir = Path("results/data")
    figure_dir = Path("results/figures")

    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data_path = (
        data_dir
        / "004B_network_propagation_threshold.csv"
    )

    df.to_csv(data_path, index=False)

    # ------------------------------------------------------------------
    # Figure 1 — total spikes
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["weight_nA"],
        df["total_spikes"],
        marker="o",
        markersize=3,
    )

    ax.set_xlabel("Synaptic strength (nA)")
    ax.set_ylabel("Total network spikes")
    ax.set_title(
        "004B — Network Activity vs Synaptic Strength"
    )

    fig.tight_layout()

    figure_path = (
        figure_dir
        / "004B_network_propagation_threshold.png"
    )

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
    print(f"Data:   {data_path}")
    print(f"Figure: {figure_path}")
    print("Figure resolution: 300 DPI")
    print("=" * 78)


if __name__ == "__main__":
    main()
