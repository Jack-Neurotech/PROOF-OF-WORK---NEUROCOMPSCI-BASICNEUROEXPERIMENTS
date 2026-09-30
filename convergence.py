"""Experiment 001A: numerical convergence of the passive membrane model."""

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
    current_nA = 0.2

    dt_values_ms = np.array([
        1.0,
        0.5,
        0.2,
        0.1,
        0.05,
        0.02,
        0.01,
    ])

    rows = []

    for dt_ms in dt_values_ms:
        time_ms = np.arange(
            0.0,
            duration_ms + dt_ms,
            dt_ms,
        )

        current = np.full_like(time_ms, current_nA)

        _, numerical_mv = simulate_passive_membrane(
            current_nA=current,
            dt_ms=dt_ms,
            parameters=parameters,
        )

        analytical_mv = analytical_constant_current_solution(
            time_ms=time_ms,
            current_nA=current_nA,
            parameters=parameters,
        )

        error_mv = numerical_mv - analytical_mv
        max_error_mv = np.max(np.abs(error_mv))
        final_error_mv = abs(error_mv[-1])

        rows.append(
            [
                dt_ms,
                max_error_mv,
                final_error_mv,
            ]
        )

    results = np.asarray(rows)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data_path = DATA_DIR / "001A_convergence.csv"

    np.savetxt(
        data_path,
        results,
        delimiter=",",
        header="dt_ms,max_absolute_error_mv,final_absolute_error_mv",
        comments="",
    )

    figure_path = FIGURE_DIR / "001A_convergence.png"

    plt.figure(figsize=(9, 6))

    plt.loglog(
        results[:, 0],
        results[:, 1],
        "o-",
        label="Maximum absolute error",
    )

    plt.xlabel("Time step (ms)")
    plt.ylabel("Maximum absolute error (mV)")
    plt.title("Experiment 001A — Numerical Convergence")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.close()

    print("=" * 60)
    print("EXPERIMENT 001A — NUMERICAL CONVERGENCE")
    print("=" * 60)
    print(f"Duration:         {duration_ms:.1f} ms")
    print(f"Injected current: {current_nA:.3f} nA")
    print()
    print("dt (ms)    max error (mV)    final error (mV)")
    print("-" * 48)

    for dt_ms, max_error, final_error in results:
        print(
            f"{dt_ms:7.3f}    "
            f"{max_error:14.8f}    "
            f"{final_error:15.8f}"
        )

    print()
    print(f"Data:   {data_path}")
    print(f"Figure: {figure_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
