"""Experiment 001B: passive membrane steady-state response to current."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from models.passive_membrane import (
    PassiveMembraneParameters,
    analytical_constant_current_solution,
    simulate_passive_membrane,
)


DATA_DIR = ROOT / "results" / "data"
FIGURE_DIR = ROOT / "results" / "figures"


def main() -> None:
    parameters = PassiveMembraneParameters(
        tau_ms=20.0,
        resistance_mohm=100.0,
        resting_potential_mv=-65.0,
    )

    duration_ms = 200.0
    dt_ms = 0.1

    current_values_nA = np.arange(
        -0.2,
        0.401,
        0.05,
    )

    rows = []

    for current_nA in current_values_nA:
        time_ms = np.arange(
            0.0,
            duration_ms + dt_ms,
            dt_ms,
        )

        current = np.full_like(time_ms, current_nA)

        _, voltage_mv = simulate_passive_membrane(
            current_nA=current,
            dt_ms=dt_ms,
            parameters=parameters,
        )

        analytical_mv = analytical_constant_current_solution(
            time_ms=time_ms,
            current_nA=current_nA,
            parameters=parameters,
        )

        numerical_final_mv = voltage_mv[-1]
        analytical_final_mv = analytical_mv[-1]

        predicted_steady_state_mv = (
            parameters.resting_potential_mv
            + parameters.resistance_mohm * current_nA
        )

        rows.append(
            [
                current_nA,
                numerical_final_mv,
                analytical_final_mv,
                predicted_steady_state_mv,
                abs(numerical_final_mv - predicted_steady_state_mv),
            ]
        )

    results = np.asarray(rows)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data_path = DATA_DIR / "001B_current_response.csv"

    np.savetxt(
        data_path,
        results,
        delimiter=",",
        header=(
            "current_nA,numerical_final_mv,"
            "analytical_final_mv,predicted_steady_state_mv,"
            "absolute_error_mv"
        ),
        comments="",
    )

    # Fit the simulated steady-state response.
    slope, intercept = np.polyfit(
        results[:, 0],
        results[:, 1],
        1,
    )

    fitted_mv = slope * results[:, 0] + intercept

    figure_path = FIGURE_DIR / "001B_current_response.png"

    plt.figure(figsize=(9, 6))

    plt.plot(
        results[:, 0],
        results[:, 1],
        "o",
        label="Numerical simulation",
    )

    plt.plot(
        results[:, 0],
        results[:, 3],
        "--",
        label="Analytical prediction",
    )

    plt.plot(
        results[:, 0],
        fitted_mv,
        ":",
        label="Linear fit",
    )

    plt.xlabel("Injected current (nA)")
    plt.ylabel("Final membrane potential (mV)")
    plt.title("Experiment 001B — Current–Voltage Relationship")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.close()

    print("=" * 60)
    print("EXPERIMENT 001B — CURRENT RESPONSE")
    print("=" * 60)
    print(f"Duration:         {duration_ms:.1f} ms")
    print(f"Time step:        {dt_ms:.3f} ms")
    print(f"Resistance:       {parameters.resistance_mohm:.1f} MOhm")
    print(f"Resting potential:{parameters.resting_potential_mv:.1f} mV")
    print()
    print("Current      Simulated V      Predicted V      Error")
    print("(nA)         (mV)             (mV)             (mV)")
    print("-" * 60)

    for row in results:
        print(
            f"{row[0]:7.2f}      "
            f"{row[1]:12.4f}      "
            f"{row[3]:12.4f}      "
            f"{row[4]:10.6f}"
        )

    print()
    print("LINEAR RESPONSE")
    print("-" * 60)
    print(f"Fitted slope:     {slope:.6f} mV/nA")
    print(f"Expected slope:   {parameters.resistance_mohm:.6f} mV/nA")
    print(f"Fitted intercept: {intercept:.6f} mV")
    print(
        f"Expected intercept:"
        f" {parameters.resting_potential_mv:.6f} mV"
    )

    slope_error = abs(slope - parameters.resistance_mohm)
    intercept_error = abs(
        intercept - parameters.resting_potential_mv
    )

    print(f"Slope error:      {slope_error:.6f} mV/nA")
    print(f"Intercept error:  {intercept_error:.6f} mV")

    print()
    print(f"Data:   {data_path}")
    print(f"Figure: {figure_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
