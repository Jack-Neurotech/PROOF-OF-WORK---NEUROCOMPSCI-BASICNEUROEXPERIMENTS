import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.lif import LIFParameters, simulate_lif


def analytical_solution(
    time_ms: np.ndarray,
    current_nA: float,
    parameters: LIFParameters,
) -> np.ndarray:
    steady_state_mv = (
        parameters.resting_potential_mv
        + parameters.resistance_mohm * current_nA
    )

    return steady_state_mv + (
        parameters.resting_potential_mv - steady_state_mv
    ) * np.exp(-time_ms / parameters.tau_ms)


def main() -> None:
    parameters = LIFParameters()

    duration_ms = 200.0
    dt_ms = 0.1
    current_nA = 0.10

    time_ms = np.arange(0.0, duration_ms + dt_ms, dt_ms)
    current = np.full_like(time_ms, current_nA)

    numerical_voltage, spike_times_ms = simulate_lif(
        current_nA=current,
        dt_ms=dt_ms,
        parameters=parameters,
    )

    analytical_voltage = analytical_solution(
        time_ms=time_ms,
        current_nA=current_nA,
        parameters=parameters,
    )

    error_mv = np.abs(numerical_voltage - analytical_voltage)

    steady_state_mv = (
        parameters.resting_potential_mv
        + parameters.resistance_mohm * current_nA
    )

    print("=" * 60)
    print("EXPERIMENT 002A — LIF SUBTHRESHOLD BEHAVIOR")
    print("=" * 60)
    print(f"Duration:                {duration_ms:.1f} ms")
    print(f"Time step:               {dt_ms:.3f} ms")
    print(f"Injected current:        {current_nA:.3f} nA")
    print(f"Resting potential:       {parameters.resting_potential_mv:.3f} mV")
    print(f"Threshold:               {parameters.threshold_mv:.3f} mV")
    print(f"Expected steady state:   {steady_state_mv:.3f} mV")
    print(f"Final numerical voltage: {numerical_voltage[-1]:.6f} mV")
    print(f"Maximum voltage:         {numerical_voltage.max():.6f} mV")
    print(f"Spike count:              {len(spike_times_ms)}")
    print(f"Maximum analytical error: {error_mv.max():.6f} mV")

    print()
    print("VALIDATION")
    print("-" * 60)

    if numerical_voltage.max() < parameters.threshold_mv:
        print("PASS: Voltage remained below threshold.")

    else:
        print("FAIL: Voltage reached threshold.")

    if len(spike_times_ms) == 0:
        print("PASS: No spikes were generated.")

    else:
        print("FAIL: Unexpected spike generation.")

    if error_mv.max() < 0.01:
        print("PASS: LIF subthreshold trajectory agrees with analytical solution.")

    else:
        print("FAIL: Numerical trajectory differs from analytical solution.")

    output_data = PROJECT_ROOT / "results" / "data" / "002A_lif_subthreshold.csv"
    output_figure = PROJECT_ROOT / "results" / "figures" / "002A_lif_subthreshold.png"

    output_data.parent.mkdir(parents=True, exist_ok=True)
    output_figure.parent.mkdir(parents=True, exist_ok=True)

    data = np.column_stack(
        (
            time_ms,
            current,
            numerical_voltage,
            analytical_voltage,
            error_mv,
        )
    )

    np.savetxt(
        output_data,
        data,
        delimiter=",",
        header="time_ms,current_nA,numerical_voltage_mv,analytical_voltage_mv,error_mv",
        comments="",
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        time_ms,
        numerical_voltage,
        label="LIF numerical",
    )
    plt.plot(
        time_ms,
        analytical_voltage,
        linestyle="--",
        label="Analytical passive solution",
    )
    plt.axhline(
        parameters.threshold_mv,
        linestyle=":",
        label="Spike threshold",
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title("Experiment 002A — LIF Subthreshold Behavior")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_figure, dpi=150)
    plt.close()

    print()
    print("OUTPUTS")
    print("-" * 60)
    print(output_data)
    print(output_figure)


if __name__ == "__main__":
    main()
