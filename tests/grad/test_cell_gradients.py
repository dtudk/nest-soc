"""
Adjoint gradients of the degradation simulation, ``nest.grad.cell.Cell.solve_for_time``.

The adjoint (:meth:`~nest.grad.cell.Cell.solve_for_time_gradients`) is checked
against finite differences of re-solves and against the forward sensitivities
(:meth:`~nest.grad.cell.Cell._forward_sensitivity`) on a small durability-like
cell with all four degradation models. A loose ``rtol`` keeps the solves fast;
the agreement is still about 1e-6.
"""

import copy
from types import SimpleNamespace

import cases_cell as cases  # cell builders of the gradient-check cases (see conftest.py)
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from nest.grad import cell, degradation, layers
from nest.grad.boundary import THETA, flatten, split

# Adjoint solves and finite differences over re-solves of the degradation simulation.
slow = pytest.mark.slow

T_MAX = 100.0  # h
OPTIONS = {"rtol": 1e-6}
RTOL = 1e-4  # adjoint vs finite differences / forward sensitivities
GUESS = {"voltage": "j", "current": "V"}  # Newton guess: zero gradient
OPERATED = {"voltage": "V", "current": "j"}


def cell_migration_rate(T, j, eta):
    """Smooth Ni migration rate [m/h] (sub-micron growth over 100 h).

    Returns
    -------
    float
        growth rate [m/h]
    """
    return 2e-4 * np.exp(-1e4 / T) * (1 + 1e-4 * j) * (1 + eta)


def cell_migration_rate_grad(T, j, eta):
    """Analytical gradient of :func:`cell_migration_rate`.

    Returns
    -------
    tuple of float
        ``(d/dT, d/dj, d/deta)``
    """
    k = 2e-4 * np.exp(-1e4 / T)
    return (cell_migration_rate(T, j, eta) * 1e4 / T**2, k * 1e-4 * (1 + eta), k * (1 + 1e-4 * j))


def build_cell(energy_balance=False, r_ed_max=None, migration_grad=True, transport="BinaryFick", **migration):
    """Durability-like 3-element cell: Ni agglomeration, Cr poisoning, oxidation and Ni migration.

    Returns
    -------
    Cell
        gradient-aware cell
    """
    fuel = cases.fuel_electrode(cases.GRAD, transport=transport)
    if r_ed_max is not None:
        fuel.degradation.r_ed_max = r_ed_max
    depleted = layers.Layer(
        delta=0.0,
        conductivity=layers.Conductivity(sigma0=3.6e7 * 0.3 / 2, E_act=8e4, theta=-1),
        degradation=degradation.NickelMigration(
            cell_migration_rate, rate_grad=cell_migration_rate_grad if migration_grad else None, **migration
        ),
    )
    return cell.Cell(
        16e-4, fuel, cases.electrolytes(cases.GRAD) + (depleted,), cases.air_electrode(cases.GRAD),
        elements=3, energy_balance=energy_balance,
    )


def boundary():
    """Inlet conditions with moderate heating in the energy-balance cell.

    Returns
    -------
    BoundaryData
        boundary conditions
    """
    return cases.boundary(**cases.OPERATION)


def _perturbed(b, name, index, h):
    # Copy of b with one attribute (or one flow entry) moved by h.
    b = copy.deepcopy(b)
    if index is None:
        setattr(b, name, getattr(b, name) + h)
    else:
        value = np.array(getattr(b, name), dtype=float)
        value[index] += h
        setattr(b, name, value)
    return b


def _finite_differences(c, b, mode, variables, events=None, objective=lambda y: y[-1], t_max=T_MAX, base=None):
    # Forward differences of objective(y(t_f)) over re-solves with the gradient's solver options.
    options = c._ivp_settings(b, t_max, OPTIONS)[0]

    def G(bb):
        return objective(c._forward(bb, mode, t_max, events, options).y)

    base = G(b) if base is None else base
    result = {}
    for name, index in variables:
        x = getattr(b, name) if index is None else getattr(b, name)[index]
        h = 1e-6 * abs(x)
        result[(name, index)] = (G(_perturbed(b, name, index, h)) - base) / h
    return result


def _entry(gradient, name, index):
    return gradient[name] if index is None else gradient[name][index]


@pytest.fixture(scope="module", params=["voltage", "current"])
def adjoint(request):
    """Adjoint gradient of the integrated power density in each mode.

    Returns
    -------
    SimpleNamespace
        cell, boundary (and a copy from before the call), mode and gradient
    """
    c, b = build_cell(), boundary()
    before = copy.deepcopy(b)
    gradient = c.solve_for_time_gradients(b, request.param, T_MAX, **OPTIONS)
    return SimpleNamespace(cell=c, boundary=b, before=before, mode=request.param, gradient=gradient)


@pytest.fixture(scope="module")
def sensitivity(adjoint):
    """Forward sensitivities ``dy(t_max)/dtheta`` of the same solve.

    Returns
    -------
    numpy.ndarray
        sensitivities, shape ``(len(y), n_theta)``
    """
    return adjoint.cell._forward_sensitivity(adjoint.boundary, adjoint.mode, T_MAX, **OPTIONS)


@slow
def test_adjoint_matches_finite_differences(adjoint):
    variables = [(OPERATED[adjoint.mode], None), ("T", None), ("P", None), ("n_fuel", 0), ("n_air", 0)]
    numerical = _finite_differences(adjoint.cell, adjoint.boundary, adjoint.mode, variables)
    for (name, index), value in numerical.items():
        assert _entry(adjoint.gradient, name, index) == pytest.approx(value, rel=RTOL), name


@slow
def test_adjoint_matches_forward_sensitivity(adjoint, sensitivity):
    forward = split(sensitivity[-1], adjoint.boundary)
    for name in THETA:
        np.testing.assert_allclose(adjoint.gradient[name], forward[name], rtol=RTOL, atol=0, err_msg=name)


@slow
def test_structural_zeros(adjoint):
    for name in (GUESS[adjoint.mode], "fuel_pressure", "air_pressure"):
        assert adjoint.gradient[name] == 0.0
    assert set(adjoint.gradient) == set(THETA)
    assert np.shape(adjoint.gradient["n_fuel"]) == np.shape(adjoint.boundary.n_fuel)
    assert isinstance(adjoint.gradient["T"], float)


@slow
def test_boundary_not_mutated(adjoint):
    np.testing.assert_array_equal(flatten(adjoint.boundary), flatten(adjoint.before))


@slow
def test_grad_methods_share_cached_result(adjoint, monkeypatch):
    c, b, mode = adjoint.cell, adjoint.boundary, adjoint.mode

    def fail(*args, **kwargs):
        raise AssertionError("recomputed")

    monkeypatch.setattr(c, "_adjoint", fail)
    for name in THETA:
        value = getattr(c.solve_for_time, f"grad_{name}")(b, mode, T_MAX, **OPTIONS)
        np.testing.assert_array_equal(value, adjoint.gradient[name])
    # Returned arrays are copies of the cached result
    c.solve_for_time.grad_n_fuel(b, mode, T_MAX, **OPTIONS)[0] = np.nan
    assert np.isfinite(c.solve_for_time.grad_n_fuel(b, mode, T_MAX, **OPTIONS)).all()


@slow
@pytest.mark.parametrize("adjoint", ["voltage"], indirect=True)
def test_custom_objective(adjoint, sensitivity):
    def objective_grad(y):
        grad = np.zeros_like(y)
        grad[0] = 1.0
        return grad

    gradient = adjoint.cell.solve_for_time_gradients(
        adjoint.boundary, adjoint.mode, T_MAX, objective=lambda y: y[0], objective_grad=objective_grad, **OPTIONS
    )
    forward = split(sensitivity[0], adjoint.boundary)
    for name in THETA:
        np.testing.assert_allclose(gradient[name], forward[name], rtol=RTOL, atol=0, err_msg=name)


@slow
def test_objective_without_gradient_warns():
    c, b = build_cell(), boundary()
    exact = c.solve_for_time_gradients(
        b, "voltage", 10.0, objective=lambda y: y[0] ** 2, objective_grad=lambda y: 2 * y[0] * np.eye(len(y))[0],
        **OPTIONS,
    )
    with pytest.warns(RuntimeWarning, match="objective"):
        numerical = c.solve_for_time_gradients(b, "voltage", 10.0, objective=lambda y: y[0] ** 2, **OPTIONS)
    for name in ("V", "T", "n_fuel"):
        np.testing.assert_allclose(numerical[name], exact[name], rtol=1e-6)


@slow
def test_terminal_event():
    c, b, mode = build_cell(), boundary(), "voltage"

    def power_reached(t, y, *args):
        return y[-1] - 1e5  # W h/m^2, about half way

    power_reached.terminal = True

    def radius(y):
        return y[0]  # Ni radius of the first element at the event

    with pytest.warns(RuntimeWarning, match="terminal event"):
        gradient = c.solve_for_time_gradients(
            b, mode, T_MAX, events=power_reached, objective=radius,
            objective_grad=lambda y: np.eye(len(y))[0], **OPTIONS,
        )
    options = c._ivp_settings(b, T_MAX, OPTIONS)[0]
    assert c._forward(b, mode, T_MAX, [power_reached], options).t < 0.9 * T_MAX
    variables = [("V", None), ("T", None), ("n_fuel", 0)]
    numerical = _finite_differences(c, b, mode, variables, events=[power_reached], objective=radius)
    for (name, index), value in numerical.items():
        assert _entry(gradient, name, index) == pytest.approx(value, rel=RTOL), name
    assert gradient["j"] == 0.0


@slow
def test_r_max_crossing():
    # The Ni radius starts at 0.25 μm and reaches r_ed_max within t_max in every element
    c, b, mode = build_cell(r_ed_max=0.2505e-6), boundary(), "voltage"
    options = c._ivp_settings(b, T_MAX, OPTIONS)[0]
    forward = c._forward(b, mode, T_MAX, None, options)
    assert len(forward.segments) == 1 + c.elements
    gradient = c.solve_for_time_gradients(b, mode, T_MAX, **OPTIONS)
    numerical = _finite_differences(c, b, mode, [("V", None)], base=forward.y[-1])
    for (name, index), value in numerical.items():
        assert _entry(gradient, name, index) == pytest.approx(value, rel=RTOL), name
    sensitivity = split(c._forward_sensitivity(b, mode, T_MAX, **OPTIONS)[-1], b)
    for name in THETA:
        np.testing.assert_allclose(gradient[name], sensitivity[name], rtol=RTOL, atol=0, err_msg=name)


@slow
def test_energy_balance():
    c, b, mode, t_max = build_cell(energy_balance=True), boundary(), "voltage", 20.0
    gradient = c.solve_for_time_gradients(b, mode, t_max, **OPTIONS)
    numerical = _finite_differences(c, b, mode, [("V", None), ("T", None)], t_max=t_max)
    for (name, index), value in numerical.items():
        assert _entry(gradient, name, index) == pytest.approx(value, rel=RTOL), name


@slow
def test_migration_fallback_warns():
    b = boundary()
    exact = build_cell().solve_for_time_gradients(b, "voltage", 10.0, **OPTIONS)
    c = build_cell(migration_grad=False)
    with pytest.warns(RuntimeWarning, match="central differences"):
        numerical = c.solve_for_time_gradients(b, "voltage", 10.0, **OPTIONS)
    for name in ("V", "T", "n_fuel"):
        np.testing.assert_allclose(numerical[name], exact[name], rtol=1e-5)


@slow
def test_migration_threshold_switch_warns():
    b, mode = boundary(), "voltage"
    c = build_cell()
    # Cut just below the lowest initial element current: degradation lowers it past the cut
    S = c.solve_time_step(c._initial_state(), copy.deepcopy(b), mode)
    j_cut = np.min(S[1]) - 1.0
    c = build_cell(
        threshold=lambda T, j, eta: j < j_cut, rate_after_threshold=cell_migration_rate,
        rate_after_threshold_grad=cell_migration_rate_grad,
    )
    with pytest.warns(RuntimeWarning, match="threshold switched"):
        c.solve_for_time_gradients(b, mode, T_MAX, **OPTIONS)


def test_stefan_maxwell_not_implemented():
    c = build_cell(transport="StefanMaxwell")
    with pytest.raises(NotImplementedError):
        c.solve_for_time.grad_T(boundary(), "voltage", T_MAX)


@slow
def test_solve_for_time_values():
    # Without options: the base solve (RK23, default tolerances)
    base, grad = cases.build_cell(cases.BASE), cases.build_cell(cases.GRAD)
    b = boundary()
    reference = base.solve_for_time(copy.deepcopy(b), "voltage", T_MAX)
    solution = grad.solve_for_time(copy.deepcopy(b), "voltage", T_MAX)
    # The grad cell solves the steady states to tighter tolerances
    np.testing.assert_allclose(solution.y[:, -1], reference.y[:, -1], rtol=5e-3, atol=1e-20)
    # Options are passed to solve_ivp on top of method="RK23"
    tight = grad.solve_for_time(copy.deepcopy(b), "voltage", T_MAX, rtol=1e-8)
    direct = solve_ivp(
        grad.advance_time_step, (0, T_MAX), grad._initial_state(), args=(copy.deepcopy(b), "voltage"),
        method="RK23", rtol=1e-8,
    )
    assert len(tight.t) > len(solution.t)
    np.testing.assert_array_equal(tight.t, direct.t)
    np.testing.assert_array_equal(tight.y, direct.y)
