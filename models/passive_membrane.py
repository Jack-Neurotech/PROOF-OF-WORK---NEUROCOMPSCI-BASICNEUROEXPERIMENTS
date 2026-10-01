"""Passive membrane model.

Implements the standard first-order passive membrane equation:

    tau * dV/dt = -(V - V_rest) + R * I

where:

    V       = membrane potential
    V_rest  = resting membrane potential
    R       = membrane resistance
    I       = injected current
    tau     = membrane time constant
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PassiveMembraneParameters:
    """Parameters defining a passive membrane."""

    tau_ms: float
    resistance_mohm: float
    resting_potential_mv: float


def simulate_passive_membrane(
    current_nA: np.ndarray,
    dt_ms: float,
    parameters: PassiveMembraneParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate passive membrane voltage using forward Euler."""

    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive.")

    if parameters.tau_ms <= 0:
        raise ValueError("tau_ms must be positive.")

    if parameters.resistance_mohm < 0:
        raise ValueError("resistance_mohm must not be negative.")

    current_nA = np.asarray(current_nA, dtype=float)

    if current_nA.ndim != 1:
        raise ValueError("current_nA must be a one-dimensional array.")

    if current_nA.size == 0:
        raise ValueError("current_nA must contain at least one value.")

    time_ms = np.arange(
        current_nA.size,
        dtype=float,
    ) * dt_ms

    voltage_mv = np.empty(
        current_nA.size,
        dtype=float,
    )

    voltage_mv[0] = parameters.resting_potential_mv

    for index in range(current_nA.size - 1):
        voltage_mv[index + 1] = voltage_mv[index] + (
            dt_ms / parameters.tau_ms
        ) * (
            -(
                voltage_mv[index]
                - parameters.resting_potential_mv
            )
            + parameters.resistance_mohm * current_nA[index]
        )

    return time_ms, voltage_mv


def analytical_constant_current_solution(
    time_ms: np.ndarray,
    current_nA: float,
    parameters: PassiveMembraneParameters,
) -> np.ndarray:
    """Return the analytical solution for constant injected current."""

    if parameters.tau_ms <= 0:
        raise ValueError("tau_ms must be positive.")

    if parameters.resistance_mohm < 0:
        raise ValueError("resistance_mohm must not be negative.")

    time_ms = np.asarray(
        time_ms,
        dtype=float,
    )

    steady_state_change_mv = (
        parameters.resistance_mohm
        * current_nA
    )

    return (
        parameters.resting_potential_mv
        + steady_state_change_mv
        * (
            1.0
            - np.exp(
                -time_ms / parameters.tau_ms
            )
        )
    )
