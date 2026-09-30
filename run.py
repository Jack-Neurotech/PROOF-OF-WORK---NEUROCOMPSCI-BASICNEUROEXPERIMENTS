"""Experiment 001: passive membrane response to constant current."""

from pathlib import Path
import sys

# Ensure the repository root is importable when this file is executed directly.
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
    current_nA = 0.2

    time_ms = np.arange(
        0.0,
        duration_ms + dt_ms,
        dt_ms,
    )

    current = np.full_like(time_ms, current_nA)

    time_ms, voltage_mv = simulate_passive_membrane(
        current_nA=current,
        dt_ms=dt_ms,
        parameters=parameters,
    )

    analytical_mv = analytical_constant_current_solution(
        time_ms=time_ms,
        current_nA=current_nA,
        parameters=parameters,
    )

    error_mv = voltage_mv - analytical_mv

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    output = np.column_stack(
        [
            time_ms,
            current,
            voltage_mv,
            analytical_mv,
            error_mv,
        ]
    )

    np.savetxt(
        DATA_DIR / "001_passive_membrane.csv",
        output,
        delimiter=",",
        header=(
            "time_ms,current_nA,numerical_voltage_mv,"
            "analytical_voltage_mv,error_mv"
        ),
        comments="",
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        time_ms,
        voltage_mv,
        label="Numerical Euler solution",
    )

    plt.plot(
        time_ms,
        analytical_mv,
        "--",
        label="Analytical solution",
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title("Experiment 001 — Passive Membrane Response")
    plt.legend()
    plt.tight_layout()

    figure_path = FIGURE_DIR / "001_passive_membrane.png"
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print("=" * 60)
    print("EXPERIMENT 001 — PASSIVE MEMBRANE")
    print("=" * 60)
    print(f"Duration:               {duration_ms:.1f} ms")
    print(f"Time step:              {dt_ms:.3f} ms")
    print(f"Injected current:       {current_nA:.3f} nA")
    print(f"Resting potential:      {parameters.resting_potential_mv:.1f} mV")
    print(f"Membrane resistance:    {parameters.resistance_mohm:.1f} MOhm")
    print(f"Membrane time constant: {parameters.tau_ms:.1f} ms")
    print()
    print(f"Final numerical V:      {voltage_mv[-1]:.4f} mV")
    print(f"Final analytical V:     {analytical_mv[-1]:.4f} mV")
    print(
        f"Maximum absolute error: "
        f"{np.max(np.abs(error_mv)):.6f} mV"
    )
    print()
    print(f"Data:   {DATA_DIR / '001_passive_membrane.csv'}")
    print(f"Figure: {figure_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
