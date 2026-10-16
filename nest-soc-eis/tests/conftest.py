"""Shared model for tests."""

import pytest

from nest.boundary import BoundaryData
from nest.cell import Cell
from nest.layers import ButlerVolmer, Conductivity, Layer, PorousTransport
from nest.properties import BasicSpecies, Mixture
from nesteis import CellEIS, ChannelGeometry, ElectrodeEIS, Model


@pytest.fixture
def model():
    transport = PorousTransport(dp=600e-9, epsilon=0.35, tau=2.0)
    fuel = Layer(
        delta=20e-6, transport=transport,
        kinetic=ButlerVolmer(
            gas=Mixture((BasicSpecies.H2, BasicSpecies.H2O)), nu=[-1, 1],
            alpha=0.5, beta=0.5, theta=0,
        ),
        conductivity=Conductivity(sigma0=3.0, theta=0),
    )
    air = Layer(
        delta=20e-6, transport=transport,
        kinetic=ButlerVolmer(
            gas=Mixture((BasicSpecies.O2, BasicSpecies.N2)), nu=[-1, 0], n_e=4,
            alpha=0.5, beta=0.5, theta=0,
        ),
        conductivity=Conductivity(sigma0=2.0, theta=0),
    )
    cell = Cell(
        area=0.04 * 0.03, electrode_fuel=fuel, electrode_air=air,
        electrolyte=(Layer(delta=10e-6, conductivity=Conductivity(sigma0=2.0, theta=0)),),
    )
    electrode = ElectrodeEIS(
        tpb_density=1e8, exchange_current_prefactor=1e-5, capacitance_volumetric=1e6,
    )
    eis = CellEIS(
        fuel=electrode, air=electrode,
        fuel_channel=ChannelGeometry(length=0.04, width=0.02, thickness=1e-3),
        air_channel=ChannelGeometry(length=0.04, width=0.025, thickness=2e-3),
        support_thickness=300e-6,
    )
    boundary = BoundaryData(n_fuel=[2e-5, 3e-5], n_air=[1e-5, 4e-5], T=1000)
    return Model(cell, eis, boundary, n_x=8, n_zf=4, n_za=3, n_zs=2)
