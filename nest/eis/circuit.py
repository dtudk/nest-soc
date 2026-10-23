"""Model the effects of external inductance
and other electrical circuit components.
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = ["ExternalCircuit"]


@dataclass(frozen=True)
class ExternalCircuit:
    r"""External circuit definition for current-interrupt simulations.

    Parameters
    ----------
    inductance : float
        Effective inductance of the external circuit [H].
    wiring_resistance : float
        Resistance of the external wiring [ohm].
    shunt_resistance : float
        Resistance of the current shunt [ohm].
    capacitance : float
        Effective parasitic capacitance of the current path [F].

    Notes
    -----
    After current interrupt, the external circuit is modeled as
    a series RLC circuit coupled to the electrochemical cell:

    .. math::
        L \frac{dI}{dt} &= V_{cell} - V_{source} - (R_{wire} + R_{shunt})I - V_{capacitor} \\
        C \frac{dV_{capacitor}}{dt} &= I

    The initial conditions are:

    .. math::
        I(0) = I_{before}
        V_{capacitor}(0) = 0

    """

    inductance: float
    wiring_resistance: float
    shunt_resistance: float
    capacitance: float

    @property
    def resistance(self) -> float:
        """Total external series resistance [ohm]."""
        return self.wiring_resistance + self.shunt_resistance

    def derivatives(
        self,
        current: float,
        capacitor_voltage: float,
        cell_voltage: float,
        source_voltage: float,
    ) -> tuple[float, float]:
        """
        Paramters
        ---------

        Returns
        -------
            dI/dt [A/s] and dVc/dt [V/s].
        """
        dcurrent = (
            cell_voltage
            - source_voltage
            - self.resistance * current
            - capacitor_voltage
        ) / self.inductance

        dcapacitor_voltage = current / self.capacitance

        return dcurrent, dcapacitor_voltage
