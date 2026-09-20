import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.lif import LIFParameters, simulate_lif


def analytical_threshold_current(parameters: LIFParameters) -> float:
    return (
        parameters.threshold_mv
        - parameters.resting_potential_mv
    ) / parameters.resistance_mohm


def analytical_threshold_crossing_time(
    current_nA: float,
    parameters: LIFParameters,
) -> float | None:
    steady_state_mv = (
        parameters.resting_potential_mv
        + parameters.resistance_mohm * current_nA
    )

    if steady_state_mv <= parameters.threshold_mv:
        return None

    numerator = (
        parameters.threshold_mv - steady_state_mv
    )
    denominator = (
        parameters.resting_potential_mv - steady_state_mv
    )

    return -parameters.tau_ms * np.log(numerator / denominator)


def main() -> None:
    parameters = LIFParameters()

    duration_ms = 200.0
    dt_ms = 0.1

    currents_nA = np.array(
        [0.10, 0.14, 0.15, 0.16, 0.20]
    )

    threshold_current = analytical_threshold_current(parameters)

    print("=" * 70)
    print("EXPERIMENT 002B — THRESHOLD AND SPIKE GENERATION")
    print("=" * 70)

    print()
    print("THEORETICAL THRESHOLD")
    print("-" * 70)
    print(
        f"Predicted threshold current: "
        f"{threshold_current:.6f} nA"
    )

    results = []

    for current_nA in currents_nA:
        time_ms = np.arange(
            0.0,
            duration_ms + dt_ms,
            dt_ms,
        )

        current = np.full_like(time_ms, current_nA)

        voltage_mv, spike_times_ms = simulate_lif(
            current_nA=current,
            dt_ms=dt_ms,
            parameters=parameters,
        )

        predicted_time = analytical_threshold_crossing_time(
            current_nA,
            parameters,
        )

        final_voltage = voltage_mv[-1]
        max_voltage = voltage_mv.max()
        spike_count = len(spike_times_ms)

        results.append(
            (
                current_nA,
                final_voltage,
                max_voltage,
                spike_count,
                spike_times_ms[0] if spike_count else np.nan,
                predicted_time
                if predicted_time is not None
                else np.nan,
            )
        )

    print()
    print(
        "CURRENT      FINAL V       MAX V       SPIKES      "
        "SIM SPIKE      ANALYTICAL"
    )
    print(
        "(nA)         (mV)          (mV)                   "
        "TIME (ms)      TIME (ms)"
    )
    print("-" * 70)

    for row in results:
        (
            current,
            final_v,
            max_v,
            spikes,
            sim_time,
            predicted_time,
        ) = row

        print(
            f"{current:6.2f}       "
            f"{final_v:10.4f}    "
            f"{max_v:10.4f}    "
            f"{spikes:5d}       "
            f"{sim_time:10.4f}      "
            f"{predicted_time:10.4f}"
        )

    print()
    print("VALIDATION")
    print("-" * 70)

    subthreshold_results = results[:2]
    threshold_result = results[2]
    suprathreshold_results = results[3:]

    subthreshold_pass = all(
        row[3] == 0
        and row[2] < parameters.threshold_mv
        for row in subthreshold_results
    )

    threshold_boundary_pass = (
        threshold_result[3] == 0
        and threshold_result[2] < parameters.threshold_mv
    )

    suprathreshold_pass = all(
        row[3] > 0
        for row in suprathreshold_results
    )

    print(
        "PASS: 0.10 and 0.14 nA remain subthreshold."
        if subthreshold_pass
        else "FAIL: Unexpected spike below threshold."
    )

    print(
        "PASS: 0.15 nA remains at the threshold boundary "
        "without numerical overshoot."
        if threshold_boundary_pass
        else "FAIL: Threshold boundary behaved unexpectedly."
    )

    print(
        "PASS: 0.16 and 0.20 nA generate spikes."
        if suprathreshold_pass
        else "FAIL: Suprathreshold currents did not spike."
    )

    # Detailed comparison for suprathreshold cases.
    for row in suprathreshold_results:
        current, _, _, spikes, sim_time, predicted_time = row

        if spikes > 0 and not np.isnan(predicted_time):
            error = abs(sim_time - predicted_time)

            print(
                f"Spike-time error at {current:.2f} nA: "
                f"{error:.6f} ms"
            )

    output_data = (
        PROJECT_ROOT
        / "results"
        / "data"
        / "002B_threshold_spiking.csv"
    )

    output_figure = (
        PROJECT_ROOT
        / "results"
        / "figures"
        / "002B_threshold_spiking.png"
    )

    output_data.parent.mkdir(parents=True, exist_ok=True)
    output_figure.parent.mkdir(parents=True, exist_ok=True)

    data = np.array(results)

    np.savetxt(
        output_data,
        data,
        delimiter=",",
        header=(
            "current_nA,final_voltage_mv,max_voltage_mv,"
            "spike_count,simulated_first_spike_ms,"
            "analytical_first_spike_ms"
        ),
        comments="",
    )

    # Plot the clearest suprathreshold example.
    plot_current = 0.20

    time_ms = np.arange(
        0.0,
        duration_ms + dt_ms,
        dt_ms,
    )

    current = np.full_like(time_ms, plot_current)

    voltage_mv, spike_times_ms = simulate_lif(
        current_nA=current,
        dt_ms=dt_ms,
        parameters=parameters,
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        time_ms,
        voltage_mv,
        label="LIF membrane potential",
    )

    plt.axhline(
        parameters.threshold_mv,
        linestyle="--",
        label="Spike threshold",
    )

    if len(spike_times_ms) > 0:
        for spike_time in spike_times_ms:
            plt.axvline(
                spike_time,
                linestyle=":",
                alpha=0.7,
            )

    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title(
        "Experiment 002B — LIF Threshold and Spike Generation "
        "(0.20 nA)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_figure, dpi=150)
    plt.close()

    print()
    print("OUTPUTS")
    print("-" * 70)
    print(output_data)
    print(output_figure)


if __name__ == "__main__":
    main()
