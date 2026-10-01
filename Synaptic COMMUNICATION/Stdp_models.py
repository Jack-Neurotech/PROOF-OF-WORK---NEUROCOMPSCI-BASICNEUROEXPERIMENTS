"""Mathematical spike-timing-dependent plasticity (STDP) model."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class STDPParameters:
    """Parameters defining an exponential STDP learning window."""

    tau_plus_ms: float = 20.0
    tau_minus_ms: float = 20.0

    a_plus: float = 0.01
    a_minus: float = -0.0105


def weight_change(delta_t_ms: float, parameters: STDPParameters) -> float:
    """Calculate synaptic weight change for a spike-time difference.

    delta_t_ms = t_post - t_pre.

    Positive delta_t:
        presynaptic spike occurs first -> potentiation.

    Negative delta_t:
        postsynaptic spike occurs first -> depression.
    """

    if parameters.tau_plus_ms <= 0:
        raise ValueError("tau_plus_ms must be positive.")

    if parameters.tau_minus_ms <= 0:
        raise ValueError("tau_minus_ms must be positive.")

    if delta_t_ms > 0:
        return parameters.a_plus * math.exp(
            -delta_t_ms / parameters.tau_plus_ms
        )

    if delta_t_ms < 0:
        return parameters.a_minus * math.exp(
            delta_t_ms / parameters.tau_minus_ms
        )

    return 0.0
