"""Electrode, channel, and fuel-support inputs for nest-soc-eis. SI units."""

from __future__ import annotations

from dataclasses import dataclass

from nest.layers import PorousTransport

__all__ = ["ElectrodeEIS", "ChannelGeometry", "CellEIS"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ElectrodeEIS:
    """Reaction and capacitance inputs for one electrode.

    Parameters
    ----------
    tpb_density : float
        Triple-phase-boundary length per electrode volume [m^-2], constant.
    exchange_current_prefactor : float
        Exchange-current prefactor per TPB length [A/(m K^theta)].
        Uses the electrode's kinetic.theta; not the area-based kinetic.gamma.
    capacitance_volumetric : float
        Double-layer capacitance per electrode volume [F/m^3].
    """

    tpb_density: float
    exchange_current_prefactor: float
    capacitance_volumetric: float

@dataclass(frozen=True, slots=True, kw_only=True)
class ChannelGeometry:
    """Flow-channel dimensions for one gas side.

    Parameters
    ----------
    length : float
        Active channel length [m], equal on the fuel and air sides.
    width : float
        Gas-channel width [m]; may differ from the active cell width.
    thickness : float
        Gas-channel height [m], perpendicular to the electrode surface.
        Fuel and air channels may have different heights.
    """

    length: float
    width: float
    thickness: float

@dataclass(frozen=True, slots=True, kw_only=True)
class CellEIS:
    """Extra inputs used alongside a NEST-SOC Cell.

    Parameters
    ----------
    fuel, air : ElectrodeEIS
        TPB and capacitance inputs. Other electrode properties come from NEST-SOC Cell.
    fuel_channel, air_channel : ChannelGeometry
        Fuel and air channel dimensions, with equal active lengths.
    support_thickness : float
        Fuel-support thickness [m]; zero means no support.
        The support transports and stores gas but has no reactions.
    support_transport : PorousTransport or None, optional
        Support transport properties; defaults to those of the fuel electrode.
    fuel_pressure, air_pressure : float or None, optional
        Inlet absolute pressures [Pa]; each defaults to boundary.P.
    """

    fuel: ElectrodeEIS
    air: ElectrodeEIS
    fuel_channel: ChannelGeometry
    air_channel: ChannelGeometry
    support_thickness: float
    support_transport: PorousTransport | None = None
    fuel_pressure: float | None = None
    air_pressure: float | None = None

