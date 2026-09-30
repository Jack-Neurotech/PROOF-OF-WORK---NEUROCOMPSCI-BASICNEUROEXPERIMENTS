
from __future__ import annotations

import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

# This file lives inside:
#
# NEURON VOLTAGE SIMULTION/
# └── experiments/
#     └── 004_networks/
#         └── propagation_threshold.py
#
# parents[2] moves back to:
#
# NEURON VOLTAGE SIMULTION/
#
# This allows Python to find the models/ package.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.lif import LIFParameters
from models.lif_network import (
    LIFNetworkParameters,
    simulate_lif_network,
)


# ============================================================
# EXPERIMENT
# ============================================================

def main():

    print("=" * 78)
    print("EXPERIMENT 004B — NETWORK PROPAGATION THRESHOLD")
    print("=" * 78)

    # ========================================================
    # QUESTION
    # ========================================================

    print("\nQUESTION")
    print("-" * 78)
    print(
        "What synaptic strength is required for activity "
        "to propagate through the network?"
    )

    # ========================================================
    # NETWORK CONFIGURATION
    # ========================================================

    n_neurons = 5

    duration_ms = 500.0
    dt_ms = 0.1

    external_current_nA = 0.20

    # Sweep synaptic strength from 0.01 nA to 1.00 nA.
    weight_start_nA = 0.01
    weight_end_nA = 1.00
    weight_step_nA = 0.01

    # --------------------------------------------------------
    # LIF neuron parameters
    # --------------------------------------------------------

    lif_parameters = LIFParameters(
        tau_ms=20.0,
        resistance_mohm=100.0,
        resting_potential_mv=-65.0,
        reset_potential_mv=-65.0,
        threshold_mv=-50.0,
        refractory_period_ms=2.0,
    )

    # --------------------------------------------------------
    # Network parameters
    # --------------------------------------------------------

    network_parameters = LIFNetworkParameters(
        neuron_parameters=lif_parameters,
        synaptic_tau_ms=5.0,
    )

    # --------------------------------------------------------
    # Simulation time
    # --------------------------------------------------------

    n_steps = int(duration_ms / dt_ms) + 1

    time_ms = np.arange(n_steps) * dt_ms

    print("\nNETWORK")
    print("-" * 78)
    print("Neuron 1 -> Neuron 2")
    print("Neuron 1 -> Neuron 3")
    print("Neuron 2 -> Neuron 4")
    print("Neuron 3 -> Neuron 4")
    print("Neuron 4 -> Neuron 5")

    # ========================================================
    # EXTERNAL STIMULUS
    # ========================================================

    current_nA = np.zeros(
        (n_neurons, n_steps),
        dtype=float,
    )

    # Only neuron 1 receives external current.
    current_nA[0, :] = external_current_nA

    # ========================================================
    # CONNECTIVITY
    # ========================================================

    # Create the fixed five-neuron feed-forward network.
    #
    # weights_nA[pre, post]
    #
    # means:
    #
    # when neuron 'pre' spikes,
    # neuron 'post' receives this synaptic current.

    connectivity = np.zeros(
        (n_neurons, n_neurons),
        dtype=float,
    )

    connectivity[0, 1] = 1.0
    connectivity[0, 2] = 1.0
    connectivity[1, 3] = 1.0
    connectivity[2, 3] = 1.0
    connectivity[3, 4] = 1.0

    # ========================================================
    # SYNAPTIC WEIGHT SWEEP
    # ========================================================

    weights_to_test = np.arange(
        weight_start_nA,
        weight_end_nA + weight_step_nA / 2,
        weight_step_nA,
    )

    print("\nSWEEP")
    print("-" * 78)
    print(
        f"Synaptic strength: "
        f"{weight_start_nA:.2f} nA -> "
        f"{weight_end_nA:.2f} nA"
    )
    print(
        f"Step:              "
        f"{weight_step_nA:.2f} nA"
    )

    # ========================================================
    # RESULTS STORAGE
    # ========================================================

    results = []

    print("\nPROPAGATION RESULTS")
    print("-" * 78)

    print(
        f"{'WEIGHT':>10} "
        f"{'N1':>6} "
        f"{'N2':>6} "
        f"{'N3':>6} "
        f"{'N4':>6} "
        f"{'N5':>6} "
        f"{'TOTAL':>8} "
        f"{'DEPTH':>8}"
    )

    print("-" * 78)

    # ========================================================
    # RUN THE WEIGHT SWEEP
    # ========================================================

    for synaptic_weight_nA in weights_to_test:

        # ----------------------------------------------------
        # Create weighted connection matrix
        # ----------------------------------------------------

        weights_nA = (
            connectivity
            * synaptic_weight_nA
        )

        # ----------------------------------------------------
        # Run network simulation
        # ----------------------------------------------------

        voltage_mv, spike_times, synaptic_current_nA = (
            simulate_lif_network(
                current_nA=current_nA,
                weights_nA=weights_nA,
                dt_ms=dt_ms,
                parameters=network_parameters,
            )
        )

        # ----------------------------------------------------
        # Count spikes
        # ----------------------------------------------------

        spike_counts = np.array(
            [len(times) for times in spike_times],
            dtype=int,
        )

        total_spikes = int(
            np.sum(spike_counts)
        )

        # ----------------------------------------------------
        # Determine propagation depth
        # ----------------------------------------------------

        active_neurons = spike_counts > 0

        # ----------------------------------------------------
        # Depth is based on the furthest neuron that actually
        # generated a spike.
        #
        # Neuron 1 = depth 1
        # Neuron 2/3 = depth 2
        # Neuron 4 = depth 3
        # Neuron 5 = depth 4
        # ----------------------------------------------------

        if not np.any(active_neurons):

            depth = 0

        else:

            active_indices = np.flatnonzero(
                active_neurons
            )

            depth = 0

            for index in active_indices:

                if index == 0:
                    neuron_depth = 1

                elif index in (1, 2):
                    neuron_depth = 2

                elif index == 3:
                    neuron_depth = 3

                elif index == 4:
                    neuron_depth = 4

                else:
                    neuron_depth = 0

                depth = max(
                    depth,
                    neuron_depth,
                )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "synaptic_weight_nA": float(
                    synaptic_weight_nA
                ),
                "neuron_1_spikes": int(
                    spike_counts[0]
                ),
                "neuron_2_spikes": int(
                    spike_counts[1]
                ),
                "neuron_3_spikes": int(
                    spike_counts[2]
                ),
                "neuron_4_spikes": int(
                    spike_counts[3]
                ),
                "neuron_5_spikes": int(
                    spike_counts[4]
                ),
                "total_spikes": total_spikes,
                "propagation_depth": depth,
            }
        )

        # ----------------------------------------------------
        # Print only meaningful propagation results.
        # ----------------------------------------------------

        if total_spikes > spike_counts[0]:

            print(
                f"{synaptic_weight_nA:10.2f} "
                f"{spike_counts[0]:6d} "
                f"{spike_counts[1]:6d} "
                f"{spike_counts[2]:6d} "
                f"{spike_counts[3]:6d} "
                f"{spike_counts[4]:6d} "
                f"{total_spikes:8d} "
                f"{depth:8d}"
            )

    # ========================================================
    # CONVERT RESULTS TO DATAFRAME
    # ========================================================

    df = pd.DataFrame(results)

    # ========================================================
    # FIND PROPAGATION THRESHOLDS
    # ========================================================

    # --------------------------------------------------------
    # First weight where any downstream neuron spikes.
    # --------------------------------------------------------

    downstream_activity = (
        (df["neuron_2_spikes"] > 0)
        | (df["neuron_3_spikes"] > 0)
        | (df["neuron_4_spikes"] > 0)
        | (df["neuron_5_spikes"] > 0)
    )

    if downstream_activity.any():

        first_propagation = df.loc[
            downstream_activity
        ].iloc[0]

        first_propagation_weight = float(
            first_propagation[
                "synaptic_weight_nA"
            ]
        )

    else:

        first_propagation_weight = None

    # --------------------------------------------------------
    # First weight where neuron 5 spikes.
    # --------------------------------------------------------

    neuron_5_activity = (
        df["neuron_5_spikes"] > 0
    )

    if neuron_5_activity.any():

        full_path = df.loc[
            neuron_5_activity
        ].iloc[0]

        full_propagation_weight = float(
            full_path[
                "synaptic_weight_nA"
            ]
        )

    else:

        full_propagation_weight = None

    # ========================================================
    # THRESHOLD SUMMARY
    # ========================================================

    print("\nTHRESHOLD SUMMARY")
    print("-" * 78)

    if first_propagation_weight is not None:

        print(
            "First downstream propagation: "
            f"{first_propagation_weight:.2f} nA"
        )

    else:

        print(
            "First downstream propagation: "
            "not observed"
        )

    if full_propagation_weight is not None:

        print(
            "Full N1 -> N5 propagation:     "
            f"{full_propagation_weight:.2f} nA"
        )

    else:

        print(
            "Full N1 -> N5 propagation:     "
            "not observed"
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\nVALIDATION")
    print("-" * 78)

    validation_pass = True

    # --------------------------------------------------------
    # Validate stimulated neuron.
    # --------------------------------------------------------

    if df["neuron_1_spikes"].max() > 0:

        print(
            "PASS: Stimulated Neuron 1 generated spikes."
        )

    else:

        print(
            "FAIL: Stimulated Neuron 1 generated no spikes."
        )

        validation_pass = False

    # --------------------------------------------------------
    # Validate downstream propagation.
    # --------------------------------------------------------

    if first_propagation_weight is not None:

        print(
            "PASS: Activity propagated beyond "
            "the stimulated neuron."
        )

    else:

        print(
            "FAIL: No downstream propagation detected."
        )

        validation_pass = False

    # --------------------------------------------------------
    # Validate full pathway.
    # --------------------------------------------------------

    if full_propagation_weight is not None:

        print(
            "PASS: Activity reached Neuron 5."
        )

    else:

        print(
            "WARNING: Activity did not reach Neuron 5 "
            "within the tested weight range."
        )

    print(
        "OVERALL RESULT: "
        f"{'PASS' if validation_pass else 'FAIL'}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    data_dir = Path("results/data")
    figure_dir = Path("results/figures")

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    data_path = (
        data_dir
        / "004B_propagation_threshold.csv"
    )

    df.to_csv(
        data_path,
        index=False,
    )

    # ========================================================
    # FIGURE 1 — TOTAL SPIKES
    # ========================================================

    total_spikes_path = (
        figure_dir
        / "004B_total_spikes_vs_weight.png"
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["synaptic_weight_nA"],
        df["total_spikes"],
        marker="o",
        markersize=3,
    )

    plt.xlabel(
        "Synaptic strength (nA)"
    )

    plt.ylabel(
        "Total network spikes"
    )

    plt.title(
        "004B — Network Spikes vs Synaptic Strength"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        total_spikes_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # FIGURE 2 — PROPAGATION DEPTH
    # ========================================================

    depth_path = (
        figure_dir
        / "004B_propagation_depth_vs_weight.png"
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["synaptic_weight_nA"],
        df["propagation_depth"],
        marker="o",
        markersize=3,
    )

    plt.xlabel(
        "Synaptic strength (nA)"
    )

    plt.ylabel(
        "Propagation depth"
    )

    plt.title(
        "004B — Propagation Depth vs Synaptic Strength"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        depth_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # OUTPUT SUMMARY
    # ========================================================

    print("\n" + "=" * 78)
    print("OUTPUTS")
    print("=" * 78)

    print(
        f"Data:        {data_path}"
    )

    print(
        f"Total spikes:{total_spikes_path}"
    )

    print(
        f"Depth:       {depth_path}"
    )

    print(
        "Figure resolution: 300 DPI"
    )

    print("=" * 78)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
