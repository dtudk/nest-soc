"""
Gradient-check cases for ``nest.grad.layers.Layer`` and ``nest.grad.cell`` (merged by ``test_gradients.py``).

:func:`build_cell` builds the same durability-like cell from either the base
``nest`` classes or the ``nest.grad`` ones, so ``test_grad_cell.py`` can
compare the two.
"""

from types import SimpleNamespace

import numpy as np

from nest import cell as base_cell
from nest import degradation as base_degradation
from nest import layers as base_layers
from nest import properties as base_properties
from nest.boundary import BoundaryData
from nest.grad import cell, degradation, layers, properties

GRAD = SimpleNamespace(cell=cell, degradation=degradation, layers=layers, properties=properties)
BASE = SimpleNamespace(
    cell=base_cell, degradation=base_degradation, layers=base_layers, properties=base_properties
)

T = 700 + 273.15  # K
EV = 8.314510 / 8.617333262e-5  # J/mol per eV
# Degradation parameters from nest/examples/durability.py.
NI_AGGLOMERATION = dict(
    psi_ed=0.4, psi_el=0.6, r0_ed=250e-9, r0_el=350e-9, epsilon=0.3,
    alpha=3.04e8, r_ed_max=500e-9,
)
CHROMIUM = dict(x_H2O=0.01, j0=4.27254e7, E_act=1.45 * EV)
OXIDATION = dict(k_mg=0.02361158401133492, E_ox=260_000)


def fuel_electrode(ns, fuel_species=("H2", "H2O"), transport="BinaryFick", degradation=True):
    """Ni-YSZ fuel electrode of the durability example.

    Returns
    -------
    Layer
        fuel electrode built from the classes of ``ns``
    """
    species = ns.properties.BasicSpecies
    n_extra = len(fuel_species) - 2
    return ns.layers.Layer(
        delta=1000e-6 + 10e-6,
        kinetic=ns.layers.ButlerVolmer(
            ns.properties.Mixture(tuple(getattr(species, s) for s in fuel_species)),
            nu=np.array([-1, 1] + [0] * n_extra), n_e=2, alpha=0.59, beta=1 - 0.59,
            # per TPB length with Ni agglomeration (pol_deg is the TPB density), else per area
            gamma=1.82527e6 / 4.19e12 if degradation else 0.56 * 1.82527e6, p=np.array([-0.1, 0.33] + [0.0] * n_extra),
            theta=1, E_act=1.09 * EV,
        ),
        transport=getattr(ns.layers, transport)(dp=600e-9, epsilon=0.3, tau=2),
        degradation=ns.degradation.NickelAgglomeration(**NI_AGGLOMERATION) if degradation else ns.degradation.NoDegradation(),
    )


def air_electrode(ns, degradation=True):
    """LSCF-CGO air electrode with O2-only kinetics, as in the durability example.

    Returns
    -------
    Layer
        air electrode built from the classes of ``ns``
    """
    return ns.layers.Layer(
        delta=32e-6,
        kinetic=ns.layers.ButlerVolmer(
            ns.properties.Mixture((ns.properties.BasicSpecies.O2,)), alpha=0.65,
            beta=1 - 0.65, gamma=1.51556e8, nu=np.array([-0.5]), E_act=1.45 * EV,
            p=np.array([0.22]),
        ),
        transport=ns.layers.BinaryFick(dp=6e-7, epsilon=0.3, tau=2.8),
        degradation=ns.degradation.ChromiumPoison(**CHROMIUM) if degradation else ns.degradation.NoDegradation(),
    )


def electrolytes(ns):
    """YSZ electrolyte and oxidising chromium scale of the durability example.

    Returns
    -------
    tuple of Layer
        electrolyte-like layers built from the classes of ``ns``
    """
    return (
        ns.layers.Layer(delta=8e-6 - 1.5e-6, conductivity=ns.layers.Conductivity(sigma0=3.6e7, E_act=8e4)),
        ns.layers.Layer(
            delta=0,
            conductivity=ns.layers.Conductivity(sigma0=45.8, E_act=33.3e3, theta=0),
            degradation=ns.degradation.InterconnectOxidation(**OXIDATION),
        ),
    )



def build_cell(ns=GRAD, elements=3, energy_balance=False):
    """Durability-like cell: Ni agglomeration, Cr poisoning and interconnect oxidation.

    Returns
    -------
    Cell
        cell built from the classes of ``ns``
    """
    fuel, air = fuel_electrode(ns), air_electrode(ns)
    return ns.cell.Cell(
        16e-4, fuel, electrolytes(ns), air, elements=elements, energy_balance=energy_balance
    )


def boundary(V=0.8, j=3000.0):
    """Inlet conditions of the durability example.

    Returns
    -------
    BoundaryData
        boundary conditions
    """
    n_fuel, n_air = 8.23e-05, 0.000345476  # mol/s
    return BoundaryData(
        V=V, j=j, n_fuel=np.array([n_fuel * 0.94, n_fuel * 0.06]),
        n_air=np.array([n_air * 0.21, n_air * 0.79]), T=T, P=1e5,
    )


WGS = dict(k=0.0171 * 3e-4, E=103191, nu=np.array([1.0, -1.0, 1.0, -1.0]), index=np.array([0, 1, 2, 3]))

INSTANCES = {
    layers.Layer: [
        fuel_electrode(GRAD),
        air_electrode(GRAD),  # O2-only kinetics, evaluated with (O2, N2) pressures
        electrolytes(GRAD)[1],  # Conductivity + InterconnectOxidation
        layers.Layer(delta=1e-5),  # default sigma0 = inf: no resistance
    ],
    cell.NoReaction: [cell.NoReaction()],
    cell.WaterGasShift: [cell.WaterGasShift(**WGS)],
    cell.Cell: [build_cell(), build_cell(energy_balance=True)],
}

CLASS_INPUTS = {
    # pol_deg: Ni radius [μm] or Cr ratio; ohm_deg: oxide thickness squared [m^2]
    layers.Layer: [{"T": T, "Ps": np.array([6.0e4, 4.0e4]), "pol_deg": 0.3, "ohm_deg": 1e-11}],
    # (H2, H2O, CO2, CO) with no CO
    cell.NoReaction: [{"T": 1027.15, "Ps": np.array([1.0e4, 4.0e4, 5.0e4, 0.0])}],
    cell.WaterGasShift: [{"T": 1027.15, "Ps": np.array([1.0e4, 4.0e4, 5.0e4, 0.0])}],
    cell.Cell: [
        {
            "T": T, "Ps_fuel": np.array([8.5e4, 1.5e4]), "Ps_air": np.array([2.0e4, 8.0e4]),
            # indexed by layer: fuel, air, YSZ, Cr scale
            "pol_deg": np.array([0.3, 0.9, 0.0, 0.0]), "ohm_deg": np.array([0.0, 0.0, 0.0, 1e-11]),
            "boundary": boundary(),
        }
    ],
}

# Degradation ODE state of the 3-element cell: (Ni radius [μm], Cr ratio) per
# element, oxide thickness squared [m^2] per element (> 0: sqrt), then power.
Y = np.array([0.3, 0.9, 0.32, 0.95, 0.34, 0.97, 1e-11, 2e-11, 3e-11, 0.0])
# Operation with moderate heating in the energy-balance cell.
OPERATION = dict(V=0.95, j=2000.0)
_POL_DEG, _OHM_DEG = build_cell()._deg_layout(Y)  # (elements, layers)

# Steady solves and degradation rates; the Newton guess (j in voltage mode, V in
# current mode) has an exactly zero gradient. Current mode nests the voltage solve
# in a Newton loop, so its finite differences are slow.
METHOD_INPUTS = {
    "Cell.solve_for_voltage": [
        {"boundary": boundary(**OPERATION), "pol_deg": _POL_DEG, "ohm_deg": _OHM_DEG, "_zero": ("j",)},
    ],
    "Cell.solve_for_current": [
        {
            "boundary": boundary(**OPERATION), "pol_deg": _POL_DEG, "ohm_deg": _OHM_DEG, "_zero": ("V",),
            "_slow": True,
        },
    ],
    "Cell.advance_time_step": [
        {"t": 10.0, "y": Y, "boundary": boundary(**OPERATION), "mode": "voltage", "_zero": ("j",)},
        {"t": 10.0, "y": Y, "boundary": boundary(**OPERATION), "mode": "current", "_zero": ("V",), "_slow": True},
    ],
}

# The adjoint gradients of the degradation simulation are checked in test_cell_gradients.py.
EXCLUDE = {"Cell.solve_for_time"}

# Flows [mol/s], the oxide thickness squared [m^2] and y are small; use relative steps.
STEP_FLOOR = {"n_fuel": 1e-20, "n_air": 1e-20, "ohm_deg": 1e-20, "y": 1e-20}
