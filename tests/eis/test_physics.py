"""Physical consistency tests."""

from copy import copy

import numpy as np

from nest.constants import F, R, P_atm
from nest.layers import Layer, PorousTransport
from nest.properties import BasicSpecies, Mixture
from nest.eis import Model


def test_equilibrium(model):
    state = model.initial_state()

    # Initial voltage must equal the Nernst voltage.
    expected = model.cell.V_nernst(
        model.T,
        model.c_in_f * R * model.T,
        model.c_in_a * R * model.T,
    )
    np.testing.assert_allclose(model.voltage_for_current(state), expected)

    # Both electrode overpotentials must be zero at equilibrium.
    for eta in model.overpotentials(state):
        np.testing.assert_allclose(eta, 0, atol=1e-14)


def test_species_conservation(model):
    # fuel support with different transport properties.
    support = PorousTransport(dp=200e-9, epsilon=0.42, tau=3.5)
    cell = copy(model.cell)
    cell.fuel_support = Layer(delta=model.cell.fuel_support.delta, transport=support)
    model = Model(
        cell,
        model.boundary,
        model.n_x,
        model.n_zf,
        model.n_za,
        n_zs=2,
    )

    state = model.initial_state()
    psi_f, psi_a, c_pf, c_pa, c_cf, c_ca, c_ps, *_ = model.unpack_state(state)

    # gas concentration gradients.
    c_ps[0, -1] *= 1.05
    c_pf[0, 0] *= 1.05
    c_pa[0, -1] *= 1.05

    # No faradaic sources to test gas transport alone.
    psi_f[:] = model.cell.electrode_fuel.kinetic.V_nernst_half(
        model.T, R * model.T * c_pf
    )
    psi_a[:] = -model.cell.electrode_air.kinetic.V_nernst_half(
        model.T, R * model.T * c_pa
    )

    voltage = model.voltage_for_current(state)
    _, _, dc_pf, dc_pa, dc_cf, dc_ca, dc_ps, *_ = model.unpack_state(
        model.time_derivatives(0, state, voltage)
    )

    sides = (
        (
            model.cell.electrode_fuel,
            c_cf,
            dc_cf,
            dc_pf,
            dc_ps,
            model.c_in_f,
            model.u_in_f,
            model.fuel_pressure,
            model.fuel_channel,
        ),
        (
            model.cell.electrode_air,
            c_ca,
            dc_ca,
            dc_pa,
            None,
            model.c_in_a,
            model.u_in_a,
            model.air_pressure,
            model.air_channel,
        ),
    )

    dx = model.grid.dx
    width = model.width_active

    for electrode, c_ch, dc_ch, dc_p, dc_s, inlet, u_in, pressure, channel in sides:
        area = channel.width * channel.height
        dz = electrode.delta / dc_p.shape[1]

        # Gas accumulation in porous electrode and support.
        porous_rate = electrode.transport.epsilon * dz * dc_p.sum(axis=1)
        if dc_s is not None:
            porous_rate += support.epsilon * model.grid.dz_support * dc_s.sum(axis=1)

        u_out = u_in - dx * width * R * model.T / (area * pressure) * porous_rate.sum()
        stored = dx * (area * dc_ch.sum(axis=1) + width * porous_rate.sum(axis=1))

        # conservation of each gas species.
        np.testing.assert_allclose(
            stored,
            area * (u_in * inlet - u_out * c_ch[:, -1]),
            rtol=1e-9,
            atol=1e-12,
        )

        # constant total channel concentration.
        np.testing.assert_allclose(dc_ch.sum(axis=0), 0, atol=1e-9)


def test_stoichiometry(model):
    state = model.initial_state()
    psi_f, psi_a, *_ = model.unpack_state(state)

    # overpotential perturbation to activate Faradaic reactions.
    psi_f += 0.01
    psi_a -= 0.01

    voltage = model.voltage_for_current(state)
    dpsi_f, dpsi_a, dc_pf, dc_pa, *_ = model.unpack_state(
        model.time_derivatives(0, state, voltage)
    )

    # Fuel electrode: H2 and H2O have opposite production rates.
    expected_h2 = (
        model.cell.electrode_fuel.capacitance_volumetric
        * dpsi_f
        / (2 * F * model.cell.electrode_fuel.transport.epsilon)
    )
    np.testing.assert_allclose(dc_pf[0], expected_h2)
    np.testing.assert_allclose(dc_pf[1], -dc_pf[0])

    # Air electrode: O2 follows four-electron stoichiometry; N2 is inert.
    expected_o2 = (
        -model.cell.electrode_air.capacitance_volumetric
        * dpsi_a
        / (4 * F * model.cell.electrode_air.transport.epsilon)
    )
    np.testing.assert_allclose(dc_pa[0], expected_o2)
    np.testing.assert_allclose(dc_pa[1], 0, atol=1e-12)


def test_dusty_gas():
    T = 1073.15
    gas = Mixture((BasicSpecies.H2, BasicSpecies.H2O))
    transport = PorousTransport(dp=600e-9, epsilon=0.3, tau=2.0)

    c = P_atm / (R * T) * np.array([0.3, 0.7])
    gradient = np.array([100.0, 400.0])

    flux = transport.dusty_gas_flux(T, gas, c, gradient)

    # DGM equations
    pressure = c.sum() * R * T
    x = c / c.sum()
    factor = transport.epsilon / transport.tau

    d12 = factor * gas.D_ij(0, 1, T, pressure)
    dk = factor * np.array([transport.D_knudsen(T, sp) for sp in gas.species])

    matrix = np.array(
        [
            [1 / dk[0] + x[1] / d12, -x[0] / d12],
            [-x[1] / d12, 1 / dk[1] + x[0] / d12],
        ]
    )

    permeability = factor * transport.dp**2 / 32
    viscosity = gas.viscosity(T, x)

    driving = -gradient - (permeability / viscosity) * (c / dk) * (
        R * T * gradient.sum()
    )

    # Verify that calculated species fluxes satisfy the DGM equations.
    np.testing.assert_allclose(matrix @ flux, driving, rtol=1e-12, atol=1e-12)
