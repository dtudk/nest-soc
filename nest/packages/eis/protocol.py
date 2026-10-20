"""Protocol definitions for fast transient simulations, 
i.e., voltage ramps and current interruptions."""

from dataclasses import dataclass
from collections.abc import Callable
from math import expm1

__all__ = ["voltage_ramp", "current_interrupt"]


@dataclass(frozen=True)
class _Protocol:
    """Voltage or current command with an optional transition time [s]."""

    value_at_time: Callable[[float], float]
    control_mode: str = "voltage"
    change_time: float | None = None

    def __call__(self, t: float) -> float:
        return self.value_at_time(t)


def voltage_ramp(
    initial: float,
    final: float,
    tau: float,
    start: float = 0.0,
) -> _Protocol:
    """Create an exponential voltage ramp.

    Parameters
    ----------
    initial, final : float
        Initial and target voltages [V].
    tau : float
        Exponential time constant [s].
    start : float, optional
        Time at which the ramp begins [s].

    Returns
    -------
    _Protocol
        Voltage command and transition time.
    """
    return _Protocol(
        lambda t: initial - (final - initial) * expm1(-max(t - start, 0.0) / tau),
        change_time=start if initial != final else None,
    )


def current_interrupt(
    current_before: float,
    at: float = 0.0,
) -> _Protocol:
    """Interrupt the mean current density.

    Parameters
    ----------
    current_before : float
        Current density before interruption [A/m²].
    at : float, optional
        Time at which the interruption occurs [s].

    Returns
    -------
    _Protocol
        Current command and interruption time.
    """
    return _Protocol(
        lambda t: current_before if t < at else 0.0,
        control_mode="current",
        change_time=at if current_before != 0 else None,
    )
