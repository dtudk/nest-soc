import warnings

import numpy as np
import pytest

from nest import properties
from nest.grad.properties import BasicSpecies

# Species that have a CoolProp viscosity fit (others raise NotImplementedError).
FITTED = ["H2", "H2O", "O2", "N2"]


# Run for every fitted species, at three scalar temperatures and one array of temperatures.
@pytest.mark.parametrize("name", FITTED)
@pytest.mark.parametrize("T", [600.0, 1000.0, 1400.0, np.linspace(550.0, 1450.0, 7)])
def test_viscosity_grad_T_matches_finite_difference(name, T):
    specie = getattr(BasicSpecies, name)
    # Central finite difference: dmu/dT ~ (mu(T + h) - mu(T - h)) / 2h, step h in K.
    h = 1e-3
    numerical = (np.asarray(specie.viscosity(T + h)) - specie.viscosity(T - h)) / (2 * h)
    analytical = specie.viscosity.grad_T(T)
    # Largest relative difference between analytical and numerical gradient.
    rel_diff = np.max(np.abs(analytical - numerical) / np.abs(numerical))
    # Fail above 1e-3; warn (test still passes) above 1e-6.
    if rel_diff > 1e-3:
        pytest.fail(f"{name}: grad_T differs from finite difference by {rel_diff:.1e} (> 1e-3)")
    if rel_diff > 1e-6:
        warnings.warn(
            f"{name}: grad_T differs from finite difference by {rel_diff:.1e} (> 1e-6)",
            stacklevel=1,
        )


@pytest.mark.parametrize("name", FITTED)
def test_viscosity_matches_base(name):
    # Wrapping a specie in nest.grad must not change its viscosity values.
    T = np.linspace(500.0, 1500.0, 11)
    np.testing.assert_array_equal(
        getattr(BasicSpecies, name).viscosity(T),  # nest.grad version
        getattr(properties.BasicSpecies, name).viscosity(T),  # original nest version
    )


def test_scalar_in_float_out():
    # A scalar temperature returns a plain float, not a 0-d numpy array.
    assert isinstance(BasicSpecies.H2.viscosity(1000.0), float)
    assert isinstance(BasicSpecies.H2.viscosity.grad_T(1000.0), float)


def test_pressure_broadcasts():
    # Pressure does not change the value, it only sets the output shape.
    P = np.array([1e5, 2e5, 3e5])
    grad = BasicSpecies.N2.viscosity.grad_T(1000.0, P)
    assert grad.shape == P.shape  # one gradient per pressure
    np.testing.assert_allclose(grad, BasicSpecies.N2.viscosity.grad_T(1000.0))  # all equal


def test_species_without_fit_raises():
    # CO2 has no viscosity fit, so both the value and its gradient must fail clearly.
    with pytest.raises(NotImplementedError):
        BasicSpecies.CO2.viscosity(1000.0)
    with pytest.raises(NotImplementedError):
        BasicSpecies.CO2.viscosity.grad_T(1000.0)


def test_unregistered_gradient_raises():
    # Only grad_T is registered; asking for grad_P fails and the error lists grad_T.
    with pytest.raises(AttributeError, match="grad_T"):
        BasicSpecies.H2.viscosity.grad_P(1000.0)


def test_grad_species_in_mixture():
    # nest.grad species are drop-in replacements inside a nest.properties.Mixture.
    gas = properties.Mixture((BasicSpecies.H2, BasicSpecies.H2O))
    base = properties.Mixture((properties.BasicSpecies.H2, properties.BasicSpecies.H2O))
    xs = np.array([0.5, 0.5])  # mole fractions
    assert gas.viscosity(1000.0, xs) == pytest.approx(base.viscosity(1000.0, xs))
