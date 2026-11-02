"""
Behaviour of the gradient-aware layer models in ``nest.grad.layers``.

Gradient values are checked against finite differences in ``test_gradients.py``
(cases in ``cases_layers.py``); this file covers values and special inputs.
"""

import numpy as np
import pytest

from nest import layers as base_layers
from nest import properties as base_properties
from nest.grad import layers, properties

species = properties.BasicSpecies
AIR = dict(
    gas=properties.Mixture((species.O2, species.N2)), nu=[-0.5, 0], n_e=2,
    alpha=0.65, beta=0.35, theta=1, E_act=1.496e5, p=[0.22, 0], gamma=10.0,
)


def _numerical(f, x):
    # Central differences per component of x, Jacobian of shape f.shape + x.shape.
    x = np.asarray(x, dtype=float)
    columns = []
    for k in range(x.size):
        h = 1e-6 * max(abs(x.flat[k]), 1.0)
        dx = np.zeros_like(x)
        dx.flat[k] = h
        columns.append((np.asarray(f(x + dx)) - np.asarray(f(x - dx))) / (2 * h))
    return np.stack(columns, axis=-1).reshape(np.shape(f(x)) + x.shape)


@pytest.mark.parametrize("cls", [layers.Kinetic, layers.ButlerVolmer])
def test_zero_pressure_of_inactive_species(cls):
    # N2 (p = 0, nu = 0) at zero pressure must not give 0/0 = NaN gradients.
    kinetic = cls(**AIR)
    T = 950.0
    Ps_1d = np.array([2.1e4, 0.0])
    grad = kinetic.j_0.grad_Ps(T, Ps_1d)
    assert np.all(np.isfinite(grad))
    np.testing.assert_allclose(grad, _numerical(lambda Ps: kinetic.j_0(T, Ps), Ps_1d), rtol=1e-6)

    Ps_2d = np.array([[2.1e4, 3.0e4], [0.0, 7.0e4]])  # 2 locations
    for method in (kinetic.j_0, kinetic.V_nernst_half):
        grad = method.grad_Ps(T, Ps_2d)
        assert np.all(np.isfinite(grad))
        full = _numerical(lambda Ps: method(T, Ps), Ps_2d)  # (2,) + (2, 2)
        np.testing.assert_allclose(grad, np.einsum("lsl->sl", full), rtol=1e-6, atol=1e-30)
    np.testing.assert_array_equal(kinetic.V_nernst_half.grad_Ps(T, Ps_2d)[1], 0.0)


def test_oxygen_only_gas_ignores_nitrogen():
    # O2-only kinetics with (O2, N2) pressures: V_nernst_half does not depend on N2.
    kinetic = layers.Kinetic(**{**AIR, "gas": properties.Mixture((species.O2,)), "nu": [-0.5], "p": [0.22]})
    grad_1d = kinetic.V_nernst_half.grad_Ps(950.0, np.array([2.1e4, 7.9e4]))
    assert grad_1d[1] == 0.0
    grad_2d = kinetic.V_nernst_half.grad_Ps(950.0, np.array([[2.1e4, 3e4], [7.9e4, 7e4]]))
    np.testing.assert_array_equal(grad_2d[1], 0.0)


@pytest.mark.parametrize(
    "kwargs", [dict(sigma0=3.34e4, E_act=8.6e4, theta=-1), dict()]
)
def test_conductivity_matches_base(kwargs):
    T = np.array([700.0, 1000.0])
    conductivity = layers.Conductivity(**kwargs)
    np.testing.assert_array_equal(conductivity.sigma(T), base_layers.Conductivity(**kwargs).sigma(T))
    np.testing.assert_array_equal(conductivity.resistivity(T), 1 / conductivity.sigma(T))
    assert np.all(np.isfinite(conductivity.resistivity.grad_T(T)))


def test_infinite_conductivity_is_constant():
    # sigma0 = inf (no resistance): finite differences of sigma are NaN, so the
    # sigma gradient is checked here instead of in test_gradients.py.
    T = np.array([700.0, 1000.0])
    conductivity = layers.Conductivity()
    np.testing.assert_array_equal(conductivity.sigma.grad_T(T), 0.0)
    np.testing.assert_array_equal(conductivity.resistivity(T), 0.0)
    np.testing.assert_array_equal(conductivity.resistivity.grad_T(T), 0.0)


@pytest.mark.parametrize("gas_species", [("H2", "H2O"), ("O2",)])
def test_binary_fick_matches_base(gas_species):
    gas = base_properties.Mixture(tuple(getattr(base_properties.BasicSpecies, s) for s in gas_species))
    P = np.array([4.0e4, 6.0e4])
    J = np.array([-0.02, 0.02])[: len(gas_species)]
    args = (J, 950.0, P, gas, 1e-3)
    np.testing.assert_array_equal(
        layers.BinaryFick(1e-6, 0.3, 3.0).dP_dl(*args), base_layers.BinaryFick(1e-6, 0.3, 3.0).dP_dl(*args)
    )


def test_stefan_maxwell_gradients_not_implemented():
    transport = layers.StefanMaxwell(1e-6, 0.3, 3.0)
    with pytest.raises(NotImplementedError):
        transport.dP_dl.grad_P(np.array([-0.02, 0.02]), 950.0, np.array([4e4, 6e4]),
                               properties.Mixture((species.H2, species.H2O)), 1e-3)
