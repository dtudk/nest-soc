"""
Gradient-check cases for ``nest.grad.degradation`` (merged by ``test_gradients.py``).
"""

import numpy as np

from nest.boundary import BoundaryData
from nest.grad import degradation


def migration_rate(T, j, eta):
    """Smooth Ni migration rate used for testing.

    Returns
    -------
    float
        growth rate [m/h]
    """
    return 1e-9 * np.exp(-1e4 / T) * (1 + 1e-4 * j) * (1 + eta)


def migration_rate_grad(T, j, eta):
    """Analytical gradient of :func:`migration_rate`.

    Returns
    -------
    tuple of float
        ``(d/dT, d/dj, d/deta)``
    """
    rate = migration_rate(T, j, eta)
    return (
        rate * 1e4 / T**2,
        1e-9 * np.exp(-1e4 / T) * 1e-4 * (1 + eta),
        1e-9 * np.exp(-1e4 / T) * (1 + 1e-4 * j),
    )


# Degradation parameters from nest/examples/durability.py.
NI_AGGLOMERATION = dict(
    psi_ed=0.4, psi_el=0.6, r0_ed=250e-9, r0_el=350e-9, epsilon=0.3,
    alpha=3.04e8, r_ed_max=500e-9,
)
CHROMIUM = dict(x_H2O=0.01, j0=4.27254e7, E_act=1.45 / 8.617333262e-5 * 8.314510)
OXIDATION = dict(k_mg=0.02361158401133492, E_ox=260_000)


def state(V=0.05):
    """Layer state as built by the cell.

    Returns
    -------
    BoundaryData
        state whose n_fuel holds the TPB partial pressures [Pa]
    """
    return BoundaryData(
        V=V, j=7000.0, n_fuel=np.array([9.0e4, 1.0e4]), n_air=np.array([2.1e4, 7.9e4]),
        T=973.15, P=1e5,
    )


INSTANCES = {
    degradation.NoDegradation: [degradation.NoDegradation()],
    degradation.NickelAgglomeration: [degradation.NickelAgglomeration(**NI_AGGLOMERATION)],
    degradation.ChromiumPoison: [degradation.ChromiumPoison(**CHROMIUM)],
    degradation.InterconnectOxidation: [degradation.InterconnectOxidation(**OXIDATION)],
    degradation.NickelMigration: [
        degradation.NickelMigration(migration_rate, rate_grad=migration_rate_grad),
        degradation.NickelMigration(migration_rate),  # central differences, warns
    ],
}


def _ni_case(m):
    """Ni agglomeration case at radius m [μm].

    Returns
    -------
    dict
        method arguments; radius arguments in m
    """
    r = m * 1e-6
    return {
        "m": m, "state": state(), "time": 10.0, "r_ed": r,
        "psi_i": 0.4, "psi_j": 0.6, "r_i": r, "r_j": 350e-9,
    }


CLASS_INPUTS = {
    degradation.NoDegradation: [{"m": 0.5, "state": state(), "time": 10.0}],
    # Radii [μm]: percolation clipped at 0.01 (0.08, 0.7), both p in the power-law
    # branch (0.25, 0.4), above r_ed_max = 0.5 μm (0.52, 0.7).
    degradation.NickelAgglomeration: [_ni_case(m) for m in (0.08, 0.25, 0.4, 0.52, 0.7)]
    + [
        # z_ii above 4.236 (p_i = 1) and below 1.764 (p_i = 0) for i = electrolyte
        {**_ni_case(0.25), "psi_i": 0.6, "psi_j": 0.4, "r_i": 350e-9, "r_j": 0.7e-6},
        {**_ni_case(0.25), "psi_i": 0.6, "psi_j": 0.4, "r_i": 350e-9, "r_j": 0.08e-6},
    ],
    degradation.ChromiumPoison: [
        {"m": 0.9, "state": state(0.05), "time": 10.0},
        {"m": 1.0, "state": state(-0.02), "time": 10.0},
        {"m": 1.0, "state": state(1e-4), "time": 10.0},  # near sinh(0)
    ],
    degradation.InterconnectOxidation: [{"m": 1e-11, "state": state(), "time": 10.0}],
    degradation.NickelMigration: [
        {"m": 1e-6, "state": state(), "time": 10.0, "eta_fuel": 0.08},
    ],
}

# Fallback of the NickelMigration instance without rate_grad (checked in test_grad_degradation.py).
EXPECTED_WARNINGS = {"NickelMigration: no rate gradient given"}

# Material variables and particle radii are tiny; use relative steps.
STEP_FLOOR = {"m": 1e-20, "r_i": 1e-20, "r_j": 1e-20, "r_ed": 1e-20}
