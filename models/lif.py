"""Leaky Integrate-and-Fire neuron model.

This module implements a mathematical Leaky Integrate-and-Fire
(LIF) neuron.

The membrane equation is:

    tau * dV/dt = -(V - V_rest) + R * I

where:

    V       = membrane potential
    V_rest  = resting membrane potential
    R       = membrane resistance
    I       = injected current
    tau     = membrane time constant

The differential equation is solved numerically using
forward Euler integration.

When the membrane potential exceeds the spike threshold,
a spike is recorded and the membrane potential is reset.
A refractory period prevents immediate additional spikes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================
# LIF PARAMETERS
# ============================================================

@dataclass(frozen=True)
class LIFParameters:
    """Parameters defining a Leaky Integrate-and-Fire neuron."""

    # --------------------------------------------------------
    # MEMBRANE DYNAMICS
    # --------------------------------------------------------

    tau_ms: float = 20.0
    resistance_mohm: float = 100.0

    # --------------------------------------------------------
    # MEMBRANE VOLTAGE
    # --------------------------------------------------------

    resting_potential_mv: float = -65.0
    threshold_mv: float = -55.0
    reset_potential_mv: float = -65.0

    # --------------------------------------------------------
    # REFRACTORY PERIOD
    # --------------------------------------------------------

    refractory_period_ms: float = 5.0


# ============================================================
# LIF SIMULATION
# ============================================================

def simulate_lif(
    current_nA: np.ndarray,
    dt_ms: float,
    parameters: LIFParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a Leaky Integrate-and-Fire neuron.

    Parameters
    ----------
    current_nA:
        One-dimensional array containing the injected current
        at each simulation time step.

    dt_ms:
        Simulation time step in milliseconds.

    parameters:
        LIF model parameters.

    Returns
    -------
    voltage_mv:
        Membrane potential at every simulation time step.

    spike_times_ms:
        One-dimensional array containing the time of every
        detected spike.
    """

    # ========================================================
    # VALIDATE TIME STEP
    # ========================================================

    if dt_ms <= 0:
        raise ValueError(
            "dt_ms must be positive."
        )

    # ========================================================
    # VALIDATE MEMBRANE PARAMETERS
    # ========================================================

    if parameters.tau_ms <= 0:
        raise ValueError(
            "tau_ms must be positive."
        )

    if parameters.resistance_mohm < 0:
        raise ValueError(
            "resistance_mohm must not be negative."
        )

    if parameters.refractory_period_ms < 0:
        raise ValueError(
            "refractory_period_ms must not be negative."
        )

    # ========================================================
    # CONVERT CURRENT TO NUMPY ARRAY
    # ========================================================

    current_nA = np.asarray(
        current_nA,
        dtype=float,
    )

    # ========================================================
    # VALIDATE CURRENT ARRAY
    # ========================================================

    if current_nA.ndim != 1:
        raise ValueError(
            "current_nA must be a one-dimensional array."
        )

    if current_nA.size == 0:
        raise ValueError(
            "current_nA must contain at least one value."
        )

    # ========================================================
    # CREATE TIME ARRAY
    # ========================================================

    time_ms = (
        np.arange(
            current_nA.size,
            dtype=float,
        )
        * dt_ms
    )

    # ========================================================
    # CREATE VOLTAGE ARRAY
    # ========================================================

    voltage_mv = np.empty(
        current_nA.size,
        dtype=float,
    )

    # Initial membrane potential is the resting potential.
    voltage_mv[0] = (
        parameters.resting_potential_mv
    )

    # ========================================================
    # CREATE SPIKE STORAGE
    # ========================================================

    spike_times_ms: list[float] = []

    # This stores the time until which the neuron remains
    # refractory.
    refractory_until_ms = -np.inf

    # ========================================================
    # MAIN SIMULATION LOOP
    # ========================================================

    for index in range(
        current_nA.size - 1
    ):

        current_time_ms = time_ms[index]

        # ====================================================
        # REFRACTORY PERIOD
        # ====================================================

        if current_time_ms < refractory_until_ms:

            # During the refractory period the neuron remains
            # at the reset potential.

            voltage_mv[index + 1] = (
                parameters.reset_potential_mv
            )

            continue

        # ====================================================
        # LIF DIFFERENTIAL EQUATION
        # ====================================================

        derivative_mv_per_ms = (
            -(
                voltage_mv[index]
                - parameters.resting_potential_mv
            )
            + (
                parameters.resistance_mohm
                * current_nA[index]
            )
        ) / parameters.tau_ms

        # ====================================================
        # FORWARD EULER INTEGRATION
        # ====================================================

        next_voltage_mv = (
            voltage_mv[index]
            + dt_ms
            * derivative_mv_per_ms
        )

        # ====================================================
        # SPIKE DETECTION
        # ====================================================

        # A spike occurs only when the membrane potential
        # becomes strictly greater than threshold.
        #
        # This is important for the 002B threshold experiment:
        # an exact threshold boundary does not produce a spike.

        if next_voltage_mv > parameters.threshold_mv:

            # ------------------------------------------------
            # RECORD SPIKE TIME
            # ------------------------------------------------

            spike_time_ms = time_ms[index + 1]

            spike_times_ms.append(
                spike_time_ms
            )

            # ------------------------------------------------
            # RESET MEMBRANE POTENTIAL
            # ------------------------------------------------

            # The reset occurs at the exact spike index.
            # This allows the reset-refractory experiment
            # to directly verify the reset voltage.

            voltage_mv[index + 1] = (
                parameters.reset_potential_mv
            )

            # ------------------------------------------------
            # START REFRACTORY PERIOD
            # ------------------------------------------------

            refractory_until_ms = (
                spike_time_ms
                + parameters.refractory_period_ms
            )

        else:

            # =================================================
            # NO SPIKE
            # =================================================

            voltage_mv[index + 1] = (
                next_voltage_mv
            )

    # ========================================================
    # RETURN SIMULATION RESULTS
    # ========================================================

    return (
        voltage_mv,
        np.asarray(
            spike_times_ms,
            dtype=float,
        ),
    )