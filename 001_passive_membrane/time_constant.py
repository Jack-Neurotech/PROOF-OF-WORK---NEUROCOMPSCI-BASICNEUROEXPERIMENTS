"""Experiment 001C: characterize the passive membrane time constant."""

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
    current_nA = 0.2

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

    resting_mv = parameters.resting_potential_mv

    steady_state_mv = (
        resting_mv
        + parameters.resistance_mohm * current_nA
    )

    total_response_mv = steady_state_mv - resting_mv

    # The membrane reaches 63.212% of its total response
    # after exactly one time constant.
    target_fraction = 1.0 - np.exp(-1.0)

    target_voltage_mv = (
        resting_mv
        + target_fraction * total_response_mv
    )

    # Find the first simulated point at or above the target.
    crossing_index = np.where(
        numerical_mv >= target_voltage_mv
    )[0][0]

    # Linear interpolation between neighboring points
    # gives a more precise estimate of the crossing time.
    if crossing_index == 0:
        tau_63_ms = time_ms[0]
    else:
        t0 = time_ms[crossing_index - 1]
        t1 = time_ms[crossing_index]

        v0 = numerical_mv[crossing_index - 1]
        v1 = numerical_mv[crossing_index]

        tau_63_ms = t0 + (
            (target_voltage_mv - v0)
            / (v1 - v0)
        ) * (t1 - t0)

    # Independent exponential estimate:
    #
    # V(t) = Vss - (Vss - V0) exp(-t/tau)
    #
    # Therefore:
    #
    # ln((Vss - V(t)) / (Vss - V0)) = -t/tau
    #
    # Fit the transformed relationship to estimate tau.
    valid = (
        (steady_state_mv - numerical_mv) > 0
    )

    fit_time_ms = time_ms[valid]
    fit_voltage_mv = numerical_mv[valid]

    normalized_remaining = (
        (steady_state_mv - fit_voltage_mv)
        / (steady_state_mv - resting_mv)
    )

    log_remaining = np.log(normalized_remaining)

    slope, intercept = np.polyfit(
        fit_time_ms,
        log_remaining,
        1,
    )

    tau_fit_ms = -1.0 / slope

    fitted_log_remaining = (
        intercept + slope * fit_time_ms
    )

    fitted_voltage_mv = (
        steady_state_mv
        - (steady_state_mv - resting_mv)
        * np.exp(fitted_log_remaining)
    )

    max_error_mv = np.max(
        np.abs(numerical_mv - analytical_mv)
    )

    tau_63_error_ms = abs(
        tau_63_ms - parameters.tau_ms
    )

    tau_fit_error_ms = abs(
        tau_fit_ms - parameters.tau_ms
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    results = np.column_stack(
        [
            time_ms,
            current,
            numerical_mv,
            analytical_mv,
            numerical_mv - analytical_mv,
        ]
    )

    data_path = DATA_DIR / "001C_time_constant.csv"

    np.savetxt(
        data_path,
        results,
        delimiter=",",
        header=(
            "time_ms,current_nA,numerical_voltage_mv,"
            "analytical_voltage_mv,error_mv"
        ),
        comments="",
    )

    figure_path = FIGURE_DIR / "001C_time_constant.png"

    plt.figure(figsize=(9, 6))

    plt.plot(
        time_ms,
        numerical_mv,
        label="Numerical simulation",
    )

    plt.plot(
        time_ms,
        analytical_mv,
        "--",
        label="Analytical solution",
    )

    plt.plot(
        fit_time_ms,
        fitted_voltage_mv,
        ":",
        label="Exponential fit",
    )

    plt.axhline(
        target_voltage_mv,
        linestyle="--",
        label="63.2% response",
    )

    plt.axvline(
        tau_63_ms,
        linestyle="--",
        label=f"τ estimate = {tau_63_ms:.2f} ms",
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title("Experiment 001C — Passive Membrane Time Constant")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.close()

    print("=" * 60)
    print("EXPERIMENT 001C — TIME CONSTANT")
    print("=" * 60)
    print(f"Duration:              {duration_ms:.1f} ms")
    print(f"Time step:             {dt_ms:.3f} ms")
    print(f"Injected current:      {current_nA:.3f} nA")
    print(f"Specified tau:         {parameters.tau_ms:.6f} ms")
    print(f"Steady-state voltage:  {steady_state_mv:.6f} mV")
    print()

    print("63.2% METHOD")
    print("-" * 60)
    print(f"Target voltage:        {target_voltage_mv:.6f} mV")
    print(f"Estimated tau:         {tau_63_ms:.6f} ms")
    print(f"Absolute error:        {tau_63_error_ms:.6f} ms")
    print()

    print("EXPONENTIAL FIT")
    print("-" * 60)
    print(f"Fitted tau:            {tau_fit_ms:.6f} ms")
    print(f"Expected tau:          {parameters.tau_ms:.6f} ms")
    print(f"Absolute error:        {tau_fit_error_ms:.6f} ms")
    print()

    print("NUMERICAL VALIDATION")
    print("-" * 60)
    print(f"Max trajectory error:  {max_error_mv:.6f} mV")

    print()
    print(f"Data:   {data_path}")
    print(f"Figure: {figure_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
