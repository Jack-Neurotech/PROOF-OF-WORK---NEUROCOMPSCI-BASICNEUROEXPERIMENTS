
"""Experiment 003 — Synaptic Transmission.

This experiment demonstrates communication between two
Leaky Integrate-and-Fire neurons.

Neuron 1:
    Presynaptic neuron
    Produces an action potential.

Synapse:
    Detects the presynaptic spike.
    Converts the spike into a postsynaptic current.

Neuron 2:
    Postsynaptic neuron
    Receives the synaptic current and responds.

The experiment demonstrates the basic computational
sequence:

    Neuron 1
        ↓
    Spike
        ↓
    Synapse
        ↓
    Synaptic current
        ↓
    Neuron 2
"""

from __future__ import annotations

# ============================================================
# STANDARD LIBRARY IMPORTS
# ============================================================

import sys
from pathlib import Path

# ============================================================
# SCIENTIFIC PYTHON IMPORTS
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# PROJECT ROOT
# ============================================================

# This file is located at:
#
# NEURON VOLTAGE SIMULTION/
# └── experiments/
#     └── 003_synapses/
#         └── synaptic_transmission.py
#
# parents[2] moves from:
#
# synaptic_transmission.py
#        ↓
# 003_synapses/
#        ↓
# experiments/
#        ↓
# NEURON VOLTAGE SIMULTION/
#
# This allows Python to locate the models package.

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

# ============================================================
# LIF MODEL IMPORT
# ============================================================

from models.lif import (
    LIFParameters,
    simulate_lif,
)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main() -> None:
    """Run the synaptic transmission experiment."""

    # ========================================================
    # SIMULATION PARAMETERS
    # ========================================================

    duration_ms = 200.0

    dt_ms = 0.1

    # ========================================================
    # CREATE TIME ARRAY
    # ========================================================

    time_ms = np.arange(
        0.0,
        duration_ms,
        dt_ms,
    )

    # ========================================================
    # CREATE LIF PARAMETERS
    # ========================================================

    parameters = LIFParameters()

    # ========================================================
    # PRESYNAPTIC NEURON
    # ========================================================

    # Neuron 1 receives a constant current strong enough
    # to generate action potentials.

    presynaptic_current_nA = np.full(
        time_ms.size,
        0.20,
        dtype=float,
    )

    # ========================================================
    # SIMULATE PRESYNAPTIC NEURON
    # ========================================================

    presynaptic_voltage_mv, presynaptic_spikes_ms = (
        simulate_lif(
            presynaptic_current_nA,
            dt_ms,
            parameters,
        )
    )

    # ========================================================
    # SYNAPTIC CURRENT
    # ========================================================

    # The synapse converts every presynaptic spike into
    # a brief postsynaptic current.

    synaptic_current_nA = np.zeros(
        time_ms.size,
        dtype=float,
    )

    # Amount of current delivered by the synapse.

    synaptic_amplitude_nA = 0.20

    # Duration of the synaptic current.

    synaptic_duration_ms = 2.0

    # ========================================================
    # CONVERT PRESYNAPTIC SPIKES INTO SYNAPTIC CURRENT
    # ========================================================

    for spike_time_ms in presynaptic_spikes_ms:

        # Find the array index corresponding to the
        # presynaptic spike.

        spike_index = int(
            round(
                spike_time_ms
                / dt_ms
            )
        )

        # Number of simulation samples corresponding
        # to the synaptic current duration.

        synaptic_steps = int(
            round(
                synaptic_duration_ms
                / dt_ms
            )
        )

        # Determine where the synaptic current ends.

        end_index = min(
            spike_index
            + synaptic_steps,
            time_ms.size,
        )

        # Add the synaptic current.

        synaptic_current_nA[
            spike_index:end_index
        ] += synaptic_amplitude_nA

    # ========================================================
    # POSTSYNAPTIC NEURON
    # ========================================================

    # The postsynaptic neuron receives the synaptic current.

    postsynaptic_voltage_mv, postsynaptic_spikes_ms = (
        simulate_lif(
            synaptic_current_nA,
            dt_ms,
            parameters,
        )
    )

    # ========================================================
    # EXPERIMENT SUMMARY
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "EXPERIMENT 003 — SYNAPTIC TRANSMISSION"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "MODEL"
    )

    print(
        "-" * 70
    )

    print(
        f"Duration:                 "
        f"{duration_ms:.1f} ms"
    )

    print(
        f"Time step:                "
        f"{dt_ms:.3f} ms"
    )

    print(
        f"Synaptic amplitude:       "
        f"{synaptic_amplitude_nA:.3f} nA"
    )

    print(
        f"Synaptic duration:        "
        f"{synaptic_duration_ms:.1f} ms"
    )

    print()

    # ========================================================
    # PRESYNAPTIC RESULTS
    # ========================================================

    print(
        "PRESYNAPTIC NEURON"
    )

    print(
        "-" * 70
    )

    print(
        f"Spike count:              "
        f"{len(presynaptic_spikes_ms)}"
    )

    if presynaptic_spikes_ms.size > 0:

        print(
            f"First spike:              "
            f"{presynaptic_spikes_ms[0]:.3f} ms"
        )

    print()

    # ========================================================
    # SYNAPTIC RESULTS
    # ========================================================

    print(
        "SYNAPSE"
    )

    print(
        "-" * 70
    )

    print(
        f"Maximum synaptic current: "
        f"{np.max(synaptic_current_nA):.3f} nA"
    )

    print(
        f"Active synaptic samples:  "
        f"{np.count_nonzero(synaptic_current_nA)}"
    )

    print()

    # ========================================================
    # POSTSYNAPTIC RESULTS
    # ========================================================

    print(
        "POSTSYNAPTIC NEURON"
    )

    print(
        "-" * 70
    )

    print(
        f"Spike count:              "
        f"{len(postsynaptic_spikes_ms)}"
    )

    if postsynaptic_spikes_ms.size > 0:

        print(
            f"First spike:              "
            f"{postsynaptic_spikes_ms[0]:.3f} ms"
        )

    print()

    # ========================================================
    # TRANSMISSION VALIDATION
    # ========================================================

    print(
        "VALIDATION"
    )

    print(
        "-" * 70
    )

    # --------------------------------------------------------
    # VALIDATE PRESYNAPTIC SPIKING
    # --------------------------------------------------------

    if presynaptic_spikes_ms.size > 0:

        print(
            "PASS: Presynaptic neuron generated spikes."
        )

    else:

        print(
            "FAIL: Presynaptic neuron generated no spikes."
        )

    # --------------------------------------------------------
    # VALIDATE SYNAPTIC TRANSMISSION
    # --------------------------------------------------------

    if np.max(synaptic_current_nA) > 0:

        print(
            "PASS: Presynaptic spikes produced "
            "synaptic current."
        )

    else:

        print(
            "FAIL: No synaptic current was generated."
        )

    # --------------------------------------------------------
    # VALIDATE POSTSYNAPTIC RESPONSE
    # --------------------------------------------------------

    if postsynaptic_voltage_mv.max() > (
        parameters.resting_potential_mv
    ):

        print(
            "PASS: Postsynaptic neuron responded "
            "to synaptic input."
        )

    else:

        print(
            "FAIL: Postsynaptic neuron showed "
            "no voltage response."
        )

    # --------------------------------------------------------
    # OVERALL RESULT
    # --------------------------------------------------------

    if (
        presynaptic_spikes_ms.size > 0
        and np.max(synaptic_current_nA) > 0
        and postsynaptic_voltage_mv.max()
        > parameters.resting_potential_mv
    ):

        print(
            "OVERALL RESULT: PASS"
        )

    else:

        print(
            "OVERALL RESULT: FAIL"
        )

    # ========================================================
    # PLOT
    # ========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True,
    )

    # ========================================================
    # PRESYNAPTIC VOLTAGE
    # ========================================================

    axes[0].plot(
        time_ms,
        presynaptic_voltage_mv,
    )

    axes[0].set_ylabel(
        "Voltage (mV)"
    )

    axes[0].set_title(
        "Presynaptic Neuron"
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    # ========================================================
    # SYNAPTIC CURRENT
    # ========================================================

    axes[1].plot(
        time_ms,
        synaptic_current_nA,
    )

    axes[1].set_ylabel(
        "Current (nA)"
    )

    axes[1].set_title(
        "Synaptic Current"
    )

    axes[1].grid(
        True,
        alpha=0.3,
    )

    # ========================================================
    # POSTSYNAPTIC VOLTAGE
    # ========================================================

    axes[2].plot(
        time_ms,
        postsynaptic_voltage_mv,
    )

    axes[2].set_xlabel(
        "Time (ms)"
    )

    axes[2].set_ylabel(
        "Voltage (mV)"
    )

    axes[2].set_title(
        "Postsynaptic Neuron"
    )

    axes[2].grid(
        True,
        alpha=0.3,
    )

    # ========================================================
    # FINALIZE FIGURE
    # ========================================================

    fig.suptitle(
        "Experiment 003 — Synaptic Transmission",
        fontsize=14,
    )

    fig.tight_layout()

    # ========================================================
    # SAVE FIGURE
    # ========================================================

    results_directory = (
        PROJECT_ROOT
        / "results"
        / "003_synapses"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (
        results_directory
        / "synaptic_transmission.png"
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    print()

    print(
        f"Figure saved to:"
    )

    print(
        figure_path
    )

    # ========================================================
    # DISPLAY FIGURE
    # ========================================================

    plt.show()


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("Starting Experiment 003 — Synaptic Transmission...")
    main()

