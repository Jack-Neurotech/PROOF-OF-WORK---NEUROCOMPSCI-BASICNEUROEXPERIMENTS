from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.lif import LIFParameters
from models.networks.lif_network import (
    LIFNetworkParameters,
    simulate_lif_network,
)


def main():
    print("=" * 78)
    print("EXPERIMENT 004C — 100-NEURON NETWORK")
    print("=" * 78)

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------
    n_neurons = 100
    duration_ms = 500.0
    dt_ms = 0.1

    external_current_nA = 0.20
    synaptic_weight_nA = 0.75

    connection_probability = 0.05
    random_seed = 337

    lif_parameters = LIFParameters(
        tau_ms=20.0,
        resistance_mohm=100.0,
        resting_potential_mv=-65.0,
        reset_potential_mv=-65.0,
        threshold_mv=-50.0,
        refractory_period_ms=2.0,
    )

    network_parameters = LIFNetworkParameters(
        neuron_parameters=lif_parameters,
        synaptic_tau_ms=5.0,
    )

    n_steps = int(duration_ms / dt_ms) + 1
    time_ms = np.arange(n_steps) * dt_ms

    print("\nNETWORK")
    print("-" * 78)
    print(f"Neurons:                  {n_neurons}")
    print(f"Duration:                 {duration_ms:.1f} ms")
    print(f"Time step:                {dt_ms:.3f} ms")
    print(f"External stimulus:        Neuron 1")
    print(f"Synaptic weight:          {synaptic_weight_nA:.3f} nA")
    print(f"Connection probability:   {connection_probability:.3f}")
    print(f"Random seed:              {random_seed}")

    # ------------------------------------------------------------------
    # CONNECTIVITY
    # ------------------------------------------------------------------
    rng = np.random.default_rng(random_seed)

    weights_nA = np.zeros((n_neurons, n_neurons), dtype=float)

    connections = rng.random((n_neurons, n_neurons)) < connection_probability

    # No self-connections in this first 100-neuron experiment.
    np.fill_diagonal(connections, False)

    weights_nA[connections] = synaptic_weight_nA

    n_connections = int(np.count_nonzero(weights_nA))

    # ------------------------------------------------------------------
    # EXTERNAL STIMULUS
    # ------------------------------------------------------------------
    current_nA = np.zeros((n_neurons, n_steps), dtype=float)

    # Stimulate only neuron 1.
    current_nA[0, :] = external_current_nA

    # ------------------------------------------------------------------
    # SIMULATION
    # ------------------------------------------------------------------
    print("\nSIMULATION")
    print("-" * 78)
    print("Running 100 coupled LIF neurons...")

    voltage_mv, spike_times, synaptic_current_nA = simulate_lif_network(
        current_nA=current_nA,
        weights_nA=weights_nA,
        dt_ms=dt_ms,
        parameters=network_parameters,
    )

    # ------------------------------------------------------------------
    # SPIKE MEASUREMENTS
    # ------------------------------------------------------------------
    spike_counts = np.array(
        [len(times) for times in spike_times],
        dtype=int,
    )

    firing_rates_hz = spike_counts / (duration_ms / 1000.0)

    active_neurons = spike_counts > 0
    n_active = int(np.count_nonzero(active_neurons))

    total_spikes = int(np.sum(spike_counts))

    population_rate_hz = total_spikes / (
        n_neurons * duration_ms / 1000.0
    )

    max_depth = 0

    # Approximate propagation depth using shortest directed path
    # from neuron 1 through the generated connectivity.
    reachable = {0}
    frontier = {0}

    while frontier:
        next_frontier = set()

        for source in frontier:
            targets = np.flatnonzero(weights_nA[source] > 0)

            for target in targets:
                if target not in reachable:
                    next_frontier.add(int(target))

        reachable.update(next_frontier)
        frontier = next_frontier

        if frontier:
            max_depth += 1

    # ------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------
    print("\nCONNECTIVITY")
    print("-" * 78)
    print(f"Possible directed connections: {n_neurons * (n_neurons - 1)}")
    print(f"Actual connections:            {n_connections}")
    print(
        f"Actual connection density:     "
        f"{n_connections / (n_neurons * (n_neurons - 1)):.4f}"
    )

    print("\nPOPULATION ACTIVITY")
    print("-" * 78)
    print(f"Active neurons:                {n_active}/{n_neurons}")
    print(
        f"Network recruitment:           "
        f"{100.0 * n_active / n_neurons:.2f}%"
    )
    print(f"Total spikes:                  {total_spikes}")
    print(f"Population firing rate:        {population_rate_hz:.3f} Hz")
    print(f"Maximum propagation depth:     {max_depth + 1}/{n_neurons}")

    print("\nNEURON ACTIVITY")
    print("-" * 78)
    print(f"{'NEURON':>8} {'SPIKES':>10} {'RATE (Hz)':>12}")
    print("-" * 78)

    for i in range(n_neurons):
        print(
            f"{i + 1:>8} "
            f"{spike_counts[i]:>10} "
            f"{firing_rates_hz[i]:>12.3f}"
        )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------
    print("\nVALIDATION")
    print("-" * 78)

    validation_pass = True

    if spike_counts[0] > 0:
        print("PASS: Stimulated Neuron 1 generated spikes.")
    else:
        print("FAIL: Stimulated Neuron 1 generated no spikes.")
        validation_pass = False

    if n_connections > 0:
        print("PASS: 100-neuron connectivity matrix contains synaptic connections.")
    else:
        print("FAIL: No synaptic connections were generated.")
        validation_pass = False

    if n_active > 1:
        print(
            "PASS: Activity propagated beyond the directly stimulated neuron."
        )
    else:
        print(
            "WARNING: Activity remained confined to the stimulated neuron."
        )

    if np.all(spike_counts >= 0):
        print("PASS: Spike counts are valid non-negative measurements.")
    else:
        print("FAIL: Invalid spike count detected.")
        validation_pass = False

    print(
        f"OVERALL RESULT: {'PASS' if validation_pass else 'FAIL'}"
    )

    # ------------------------------------------------------------------
    # SAVE DATA
    # ------------------------------------------------------------------
    data_dir = Path("results/data")
    figure_dir = Path("results/figures")

    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for neuron in range(n_neurons):
        rows.append(
            {
                "neuron": neuron + 1,
                "spike_count": spike_counts[neuron],
                "firing_rate_hz": firing_rates_hz[neuron],
                "mean_voltage_mv": float(np.mean(voltage_mv[neuron])),
                "max_voltage_mv": float(np.max(voltage_mv[neuron])),
            }
        )

    df = pd.DataFrame(rows)

    data_path = data_dir / "004C_100_neuron_network.csv"
    df.to_csv(data_path, index=False)

    # ------------------------------------------------------------------
    # FIGURE 1 — RASTER
    # ------------------------------------------------------------------
    raster_path = figure_dir / "004C_100_neuron_raster.png"

    plt.figure(figsize=(12, 8))

    for neuron, times in enumerate(spike_times, start=1):
        if len(times) > 0:
            plt.scatter(
                times,
                np.full(len(times), neuron),
                s=8,
            )

    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron")
    plt.title("004C — 100-Neuron Network Spike Raster")
    plt.ylim(0, n_neurons + 1)
    plt.xlim(0, duration_ms)
    plt.tight_layout()
    plt.savefig(raster_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # FIGURE 2 — FIRING RATE
    # ------------------------------------------------------------------
    rate_path = figure_dir / "004C_100_neuron_firing_rates.png"

    plt.figure(figsize=(12, 6))

    plt.bar(
        np.arange(1, n_neurons + 1),
        firing_rates_hz,
    )

    plt.xlabel("Neuron")
    plt.ylabel("Firing rate (Hz)")
    plt.title("004C — 100-Neuron Firing Rates")
    plt.xlim(0, n_neurons + 1)
    plt.tight_layout()
    plt.savefig(rate_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # OUTPUT SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("OUTPUTS")
    print("=" * 78)
    print(f"Data:       {data_path}")
    print(f"Raster:     {raster_path}")
    print(f"Rates:      {rate_path}")
    print("Figure resolution: 300 DPI")
    print("=" * 78)


if __name__ == "__main__":
    main()
