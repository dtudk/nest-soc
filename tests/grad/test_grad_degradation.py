"""
Behaviour of the gradient-aware degradation models in ``nest.grad.degradation``.

Gradient values are checked against finite differences in ``test_gradients.py``.
"""

import warnings

import cases_degradation as cases  # example parameters of the gradient-check cases (see conftest.py)
import numpy as np
import pytest

from nest import degradation as base
from nest.grad import degradation

PAIRS = [
    (base.NoDegradation(), degradation.NoDegradation(), 0.5),
    *[
        (base.NickelAgglomeration(**cases.NI_AGGLOMERATION), degradation.NickelAgglomeration(**cases.NI_AGGLOMERATION), m)
        for m in (0.08, 0.25, 0.4, 0.52, 0.7)
    ],
    (base.ChromiumPoison(**cases.CHROMIUM), degradation.ChromiumPoison(**cases.CHROMIUM), 0.9),
    (base.InterconnectOxidation(**cases.OXIDATION), degradation.InterconnectOxidation(**cases.OXIDATION), 1e-11),
    (
        base.NickelMigration(cases.migration_rate),
        degradation.NickelMigration(cases.migration_rate, rate_grad=cases.migration_rate_grad),
        1e-6,
    ),
]
IDS = [f"{type(new).__name__}-{m}" for _, new, m in PAIRS]


@pytest.mark.parametrize("old, new, m", PAIRS, ids=IDS)
def test_values_match_base(old, new, m):
    # The nest.grad models must reproduce the original rates and ratios exactly.
    state = cases.state()
    assert new.material_dt(m, state, 10.0, eta_fuel=0.08) == old.material_dt(m, state, 10.0, eta_fuel=0.08)
    assert new.pol_deg(m) == old.pol_deg(m)
    assert new.ohm_deg(m) == old.ohm_deg(m)
    assert new.m0 == old.m0


@pytest.mark.parametrize("old, new, m", PAIRS, ids=IDS)
def test_scalar_in_float_out(old, new, m):
    state = cases.state()
    assert isinstance(new.material_dt(m, state, 10.0, eta_fuel=0.08), float)
    assert isinstance(new.pol_deg(m), float)
    assert isinstance(new.ohm_deg(m), float)


def test_call_site_keywords():
    # cell.py calls material_dt(m, state, t) and, for ohmic layers, with eta_fuel=...
    model = degradation.InterconnectOxidation(**cases.OXIDATION)
    state = cases.state()
    assert model.material_dt(1e-11, state, 10.0) == model.material_dt(1e-11, state, 10.0, eta_fuel=0.1)


def test_migration_warns_once_without_rate_grad():
    model = degradation.NickelMigration(cases.migration_rate)
    state = cases.state()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.material_dt.grad_T(1e-6, state, 10.0, eta_fuel=0.08)
        model.material_dt.grad_j(1e-6, state, 10.0, eta_fuel=0.08)
        model.material_dt_partials(1e-6, state, 10.0, eta_fuel=0.08)
    assert len([w for w in caught if "central differences" in str(w.message)]) == 1


def test_migration_no_warning_with_rate_grad():
    model = degradation.NickelMigration(cases.migration_rate, rate_grad=cases.migration_rate_grad)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.material_dt_partials(1e-6, cases.state(), 10.0, eta_fuel=0.08)


def test_migration_fallback_matches_rate_grad():
    # Central differences agree with the analytical rate gradient.
    exact = degradation.NickelMigration(cases.migration_rate, rate_grad=cases.migration_rate_grad)
    numeric = degradation.NickelMigration(cases.migration_rate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.testing.assert_allclose(
            numeric.rate_partials(973.15, 7000.0, 0.08), exact.rate_partials(973.15, 7000.0, 0.08), rtol=1e-6
        )


def test_migration_threshold_switches_rate_grad():
    # Above the threshold the gradient of rate_after_threshold is used.
    model = degradation.NickelMigration(
        cases.migration_rate,
        threshold=lambda T, j, eta: j > 5000,
        rate_after_threshold=lambda T, j, eta: 2 * cases.migration_rate(T, j, eta),
        rate_grad=cases.migration_rate_grad,
        rate_after_threshold_grad=lambda T, j, eta: tuple(2 * g for g in cases.migration_rate_grad(T, j, eta)),
    )
    state = cases.state()  # j = 7000
    np.testing.assert_allclose(
        model.material_dt.grad_T(0.0, state, 10.0, eta_fuel=0.08),
        2 * cases.migration_rate_grad(state.T, state.j, 0.08)[0],
    )


@pytest.mark.parametrize("old, new, m", PAIRS, ids=IDS)
def test_partials_match_registered_gradients(old, new, m):
    state = cases.state()
    partials = new.material_dt_partials(m, state, 10.0, eta_fuel=0.08)
    assert set(partials) == set(degradation.PARTIALS)
    assert np.shape(partials["n_fuel"]) == (len(state.n_fuel),)
    for name, value in partials.items():
        if name in type(new).material_dt.gradients:
            expected = getattr(new.material_dt, f"grad_{name}")(m, state, 10.0, eta_fuel=0.08)
            np.testing.assert_array_equal(value, expected)
        else:
            assert np.all(np.asarray(value) == 0)


def test_m_scale():
    assert degradation.NickelAgglomeration.m_scale == 0.25
    assert degradation.ChromiumPoison.m_scale == 1.0
    assert degradation.InterconnectOxidation.m_scale == 1e-12
    assert degradation.NickelMigration.m_scale == 1e-6
    assert degradation.NoDegradation.m_scale == 1.0
