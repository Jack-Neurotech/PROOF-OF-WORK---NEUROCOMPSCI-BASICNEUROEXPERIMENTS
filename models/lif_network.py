
"""
LIF Network Model
=================

A simple feed-forward network of leaky integrate-and-fire (LIF)
neurons connected through exponentially decaying synaptic currents.

The network:
    1. Receives external current.
    2. Simulates each LIF neuron.
    3. Detects spikes.
    4. Converts presynaptic spikes into synaptic currents.
    5. Propagates those currents to downstream neurons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.lif import LIFParameters


# ============================================================
# NETWORK PARAMETERS
# ============================================================

@dataclass
class LIFNetworkParameters:
    """
    Parameters controlling the LIF network.

    neuron_parameters:
        LIF parameters shared by every neuron.

    synaptic_tau_ms:
        Time constant controlling how quickly synaptic current
        decays after a presynaptic spike.
    """

    neuron_parameters: LIFParameters
    synaptic_tau_ms: float = 5.0


# ============================================================
# NETWORK SIMULATION
# ============================================================

def simulate_lif_network(
    current_nA: np.ndarray,
    weights_nA: np.ndarray,
    dt_ms: float,
    parameters: LIFNetworkParameters,
):
    """
    Simulate a network of LIF neurons.

    Parameters
    ----------
    current_nA:
        External current for every neuron.

        Shape:
            (n_neurons, n_timepoints)

    weights_nA:
        Synaptic connection matrix.

        weights_nA[pre, post]

        means:
            when neuron 'pre' spikes, it sends this amount
            of synaptic current to neuron 'post'.

    dt_ms:
        Simulation timestep in milliseconds.

    parameters:
        LIF network parameters.

    Returns
    -------
    voltage_mv:
        Voltage of every neuron over time.

    spike_times:
        List containing spike times for each neuron.

    synaptic_current_nA:
        Synaptic current delivered to every neuron over time.
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    current_nA = np.asarray(current_nA, dtype=float)
    weights_nA = np.asarray(weights_nA, dtype=float)

    if current_nA.ndim != 2:
        raise ValueError(
            "current_nA must have shape (n_neurons, n_timepoints)."
        )

    if weights_nA.ndim != 2:
        raise ValueError(
            "weights_nA must have shape (n_neurons, n_neurons)."
        )

    n_neurons, n_timepoints = current_nA.shape

    if weights_nA.shape != (n_neurons, n_neurons):
        raise ValueError(
            "weights_nA must have shape "
            "(n_neurons, n_neurons)."
        )

    if dt_ms <= 0:
        raise ValueError("dt_ms must be greater than zero.")

    if parameters.synaptic_tau_ms <= 0:
        raise ValueError(
            "synaptic_tau_ms must be greater than zero."
        )

    # --------------------------------------------------------
    # Create simulation arrays
    # --------------------------------------------------------

    voltage_mv = np.full(
        (n_neurons, n_timepoints),
        parameters.neuron_parameters.resting_potential_mv,
        dtype=float,
    )

    synaptic_current_nA = np.zeros(
        (n_neurons, n_timepoints),
        dtype=float,
    )

    spike_times = [[] for _ in range(n_neurons)]

    # --------------------------------------------------------
    # Track refractory state
    # --------------------------------------------------------

    refractory_until_ms = np.full(
        n_neurons,
        -np.inf,
        dtype=float,
    )

    # --------------------------------------------------------
    # Exponential synaptic decay
    # --------------------------------------------------------

    synaptic_decay = np.exp(
        -dt_ms / parameters.synaptic_tau_ms
    )

    # --------------------------------------------------------
    # Main simulation loop
    # --------------------------------------------------------

    for t in range(1, n_timepoints):

        time_ms = t * dt_ms

        # ----------------------------------------------------
        # Decay synaptic currents from previous spikes
        # ----------------------------------------------------

        synaptic_current_nA[:, t] = (
            synaptic_current_nA[:, t - 1]
            * synaptic_decay
        )

        # ----------------------------------------------------
        # Simulate each neuron
        # ----------------------------------------------------

        for neuron in range(n_neurons):

            # -----------------------------------------------
            # Check refractory period
            # -----------------------------------------------

            if time_ms < refractory_until_ms[neuron]:

                voltage_mv[neuron, t] = (
                    parameters.neuron_parameters.reset_potential_mv
                )

                continue

            # -----------------------------------------------
            # Total input current
            # -----------------------------------------------

            total_current_nA = (
                current_nA[neuron, t]
                + synaptic_current_nA[neuron, t]
            )

            # -----------------------------------------------
            # LIF differential equation
            #
            # dV/dt =
            #     (-(V - V_rest) + R*I) / tau
            # -----------------------------------------------

            voltage_previous = voltage_mv[neuron, t - 1]

            dv_dt = (
                -(
                    voltage_previous
                    - parameters.neuron_parameters.resting_potential_mv
                )
                + parameters.neuron_parameters.resistance_mohm
                * total_current_nA
            ) / parameters.neuron_parameters.tau_ms

            voltage_new = (
                voltage_previous
                + dv_dt * dt_ms
            )

            # -----------------------------------------------
            # Spike detection
            # -----------------------------------------------

            if (
                voltage_new
                > parameters.neuron_parameters.threshold_mv
            ):

                # Record spike time.
                spike_times[neuron].append(time_ms)

                # Reset neuron.
                voltage_mv[neuron, t] = (
                    parameters.neuron_parameters.reset_potential_mv
                )

                # Enter refractory period.
                refractory_until_ms[neuron] = (
                    time_ms
                    + parameters.neuron_parameters.refractory_period_ms
                )

                # ------------------------------------------------
                # Propagate spike to downstream neurons.
                #
                # weights_nA[pre, post]
                # ------------------------------------------------

                synaptic_current_nA[:, t] += (
                    weights_nA[neuron, :]
                )

            else:

                voltage_mv[neuron, t] = voltage_new

    return (
        voltage_mv,
        spike_times,
        synaptic_current_nA,
    )

