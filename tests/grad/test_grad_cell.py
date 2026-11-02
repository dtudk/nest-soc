"""
Behaviour of the gradient-aware cell model in ``nest.grad.cell`` and ``nest.grad.layers.Layer``.

Leaf gradients are checked against finite differences in ``test_gradients.py``
(cases in ``cases_cell.py``); this file compares values and solves with the
base ``nest`` model, checks the layout of the steady tangents and degradation
rate Jacobians, and checks a cell with a water gas shift reaction.
"""

import copy

import cases_cell as cases  # cell builders of the gradient-check cases (see conftest.py)
import numpy as np
import pytest

from nest import layers as base_layers
from nest.grad import cell, layers
from nest.grad.boundary import flatten, theta_slices

J = 3000.0
OPERATION = dict(V=0.95, j=2000.0)  # moderate heating in the energy-balance cell
PS_FUEL = np.array([8.5e4, 1.5e4])
PS_AIR = np.array([2.0e4, 8.0e4])
POL_DEG = np.array([0.3, 0.9, 0.0, 0.0])  # fuel, air, YSZ, Cr scale
OHM_DEG = np.array([0.0, 0.0, 0.0, 1e-11])
# Degradation ODE state of a 3-element cell: (Ni radius, Cr ratio) per element,
# oxide thickness squared per element, then power.
Y = np.array([0.3, 0.9, 0.32, 0.95, 0.34, 0.97, 1e-11, 2e-11, 3e-11, 0.0])


def _numerical(f, x):
    # Central differences per component of x, Jacobian of shape f.shape + x.shape.
    x = np.asarray(x, dtype=float)
    columns = []
    for k in range(x.size):
        h = 1e-6 * abs(x.flat[k]) if x.flat[k] != 0 else 1e-6
        dx = np.zeros_like(x)
        dx.flat[k] = h
        columns.append((np.asarray(f(x + dx)) - np.asarray(f(x - dx))) / (2 * h))
    return np.stack(columns, axis=-1).reshape(np.shape(f(x)) + x.shape)


def _boundary_function(call, boundary, name):
    # call(boundary) as a function of one boundary attribute, evaluated on copies.
    def f(x):
        b = copy.deepcopy(boundary)
        setattr(b, name, x if np.ndim(x) else float(x))
        return call(b)

    return f


@pytest.fixture(params=[False, True], ids=["isothermal", "energy_balance"])
def cells(request):
    """Same cell from the base and the gradient-aware classes.

    Returns
    -------
    tuple of Cell
        ``(base, grad)`` cells
    """
    return (
        cases.build_cell(cases.BASE, energy_balance=request.param),
        cases.build_cell(cases.GRAD, energy_balance=request.param),
    )


def test_leaf_values_match_base(cells):
    base, grad = cells
    b = cases.boundary(**OPERATION)
    T = cases.T
    assert grad.V_nernst(T, PS_FUEL, PS_AIR) == base.V_nernst(T, PS_FUEL, PS_AIR)
    assert grad.V_thermoneutral(T) == base.V_thermoneutral(T)
    assert grad.V(J, T, PS_FUEL, PS_AIR, POL_DEG, OHM_DEG) == base.V(
        J, T, PS_FUEL, PS_AIR, pol_deg=POL_DEG, ohm_deg=OHM_DEG
    )
    np.testing.assert_array_equal(grad.dn_fuel(J, b), base.dn_fuel(J, b))
    np.testing.assert_array_equal(grad.dn_air(J), base.dn_air(J))
    np.testing.assert_array_equal(grad.Ps_fuel(b, J), base.Ps_fuel(b, J))
    np.testing.assert_array_equal(grad.Ps_air(b, J), base.Ps_air(b, J))


def test_layer_values_match_base(cells):
    base, grad = cells
    base_layers_ = (base.electrode_fuel, base.electrode_air, *base.electrolyte)
    for k, (base_layer, layer) in enumerate(zip(base_layers_, grad._layers())):
        Ps = PS_FUEL if k == 0 else PS_AIR
        kwargs = {"pol_deg": POL_DEG[k], "ohm_deg": OHM_DEG[k]}
        assert layer.V(J, cases.T, Ps, **kwargs) == base_layer.V(J, cases.T, Ps, **kwargs)
        assert layer.V_ohm(J, cases.T, **kwargs) == base_layer.V_ohm(J, cases.T, **kwargs)
        assert layer.V_act(J, cases.T, Ps, **kwargs) == base_layer.V_act(J, cases.T, Ps, **kwargs)
        np.testing.assert_array_equal(layer.Ps_star(J, cases.T, Ps), base_layer.Ps_star(J, cases.T, Ps))


def test_none_means_initial_material_value():
    grad = cases.build_cell()
    m0 = np.array([layer.degradation.m0 for layer in grad._layers()], dtype=float)
    assert grad.V(J, cases.T, PS_FUEL, PS_AIR) == grad.V(J, cases.T, PS_FUEL, PS_AIR, m0, m0)
    fuel = grad.electrode_fuel
    assert fuel.V(J, cases.T, PS_FUEL) == fuel.V(J, cases.T, PS_FUEL, pol_deg=fuel.degradation.m0)


def test_layer_requires_grad_components():
    with pytest.raises(TypeError, match="kinetic"):
        layers.Layer(kinetic=base_layers.Kinetic())
    with pytest.raises(TypeError, match="conductivity"):
        layers.Layer(conductivity=base_layers.Conductivity())
    with pytest.raises(TypeError, match="degradation"):
        layers.Layer(degradation=cases.BASE.degradation.NoDegradation())
    with pytest.raises(TypeError, match="electrode_fuel"):
        cell.Cell(1e-4, base_layers.Layer(), (), layers.Layer())
    with pytest.raises(TypeError, match="reactions"):
        cell.Cell(1e-4, layers.Layer(), (), layers.Layer(), reactions=(cases.BASE.cell.NoReaction(),))


def test_layer_defaults_are_not_shared():
    first, second = layers.Layer(), layers.Layer()
    for name in ("kinetic", "conductivity", "transport", "degradation"):
        assert getattr(first, name) is not getattr(second, name)
    assert isinstance(cell.Cell(1e-4, first, (), second).reactions[0], cell.NoReaction)


def test_advance_step_area_matches_base(cells):
    base, grad = cells
    b = cases.boundary(**OPERATION)
    out_base = base.advance_step_area(copy.deepcopy(b), pol_deg=POL_DEG, ohm_deg=OHM_DEG)
    out = grad.advance_step_area(copy.deepcopy(b), pol_deg=POL_DEG, ohm_deg=OHM_DEG)
    assert abs(out.j - out_base.j) < 10  # base Newton tolerance [A/m^2]
    # tight solve: the residual vanishes
    residual = b.V - grad.V(out.j, b.T, grad.Ps_fuel(b, out.j), grad.Ps_air(b, out.j), POL_DEG, OHM_DEG)
    assert abs(residual) < 1e-12
    np.testing.assert_allclose(out.n_fuel, out_base.n_fuel, rtol=1e-3)
    np.testing.assert_allclose(out.n_air, out_base.n_air, rtol=1e-3)
    np.testing.assert_allclose(out.T, out_base.T, rtol=1e-4)  # heat scales with j
    assert out.P == b.P and out.V == b.V


def test_solve_for_voltage_matches_base(cells):
    base, grad = cells
    pol_deg, ohm_deg = grad._deg_layout(Y)
    b = cases.boundary(**OPERATION)
    solution_base = base.solve_for_voltage(copy.deepcopy(b), pol_deg=pol_deg, ohm_deg=ohm_deg)
    solution = grad.solve_for_voltage(copy.deepcopy(b), pol_deg=pol_deg, ohm_deg=ohm_deg)
    assert solution.shape == solution_base.shape
    np.testing.assert_allclose(solution[1], solution_base[1], atol=10)
    np.testing.assert_allclose(solution, solution_base, rtol=5e-3, atol=1e-12)


def test_solve_for_current_matches_base(cells):
    base, grad = cells
    pol_deg, ohm_deg = grad._deg_layout(Y)
    b_base, b = cases.boundary(**OPERATION), cases.boundary(**OPERATION)
    solution_base = base.solve_for_current(b_base, pol_deg=pol_deg, ohm_deg=ohm_deg)
    solution = grad.solve_for_current(b, pol_deg=pol_deg, ohm_deg=ohm_deg)
    assert abs(b.V - b_base.V) < 5e-3  # base tolerance 5e-4 V on the Newton step
    assert abs(np.mean(solution[1]) - b.j) < 1e-6  # tight: the mean current is met
    np.testing.assert_allclose(solution, solution_base, rtol=5e-3, atol=1e-12)


def test_deg_layout_mirrors_solve_time_step():
    grad = cases.build_cell()
    pol_deg, ohm_deg = grad._deg_layout(Y)
    expected_pol = np.zeros((3, 4))
    expected_pol[:, 0], expected_pol[:, 1] = Y[0:6:2], Y[1:6:2]
    expected_ohm = np.zeros((3, 4))
    expected_ohm[:, 3] = Y[6:9]
    np.testing.assert_array_equal(pol_deg, expected_pol)
    np.testing.assert_array_equal(ohm_deg, expected_ohm)
    slots = grad._deg_slots()
    assert sorted(index for index, *_ in slots) == list(range(len(Y) - 1))
    for index, name, element, layer in slots:
        assert {"pol_deg": pol_deg, "ohm_deg": ohm_deg}[name][element, layer] == Y[index]


@pytest.mark.parametrize("mode", ["voltage", "current"])
def test_solve_time_step_matches_base(cells, mode):
    base, grad = cells
    b_base, b = cases.boundary(**OPERATION), cases.boundary(**OPERATION)
    solution_base = base.solve_time_step(Y, b_base, mode)
    solution = grad.solve_time_step(Y, b, mode)
    np.testing.assert_allclose(solution, solution_base, rtol=5e-3, atol=1e-12)
    # the boundary is updated like in the base model
    if mode == "voltage":
        assert b.j == np.sum(solution[1]) / grad.elements
    else:
        assert b.V == np.sum(solution[0]) / grad.elements


@pytest.mark.parametrize("mode", ["voltage", "current"])
def test_advance_time_step_matches_base(cells, mode):
    base, grad = cells
    b_base, b = cases.boundary(**OPERATION), cases.boundary(**OPERATION)
    rates_base = base.advance_time_step(10.0, Y, b_base, mode)
    rates = grad.advance_time_step(10.0, Y, b, mode)
    np.testing.assert_allclose(rates, rates_base, rtol=5e-3, atol=1e-30)
    # the value method updates the boundary like the base model
    assert abs(b.j - b_base.j) < 10 and abs(b.V - b_base.V) < 5e-3


@pytest.mark.parametrize("mode", ["voltage", "current"])
def test_rhs_jacobians_layout(cells, mode):
    _, grad = cells
    b = cases.boundary(**OPERATION)
    theta = flatten(b)
    f, J_y, J_theta = grad._rhs_jacobians(10.0, Y, b, mode)
    np.testing.assert_array_equal(flatten(b), theta)  # boundary not modified
    np.testing.assert_allclose(f, grad.advance_time_step(10.0, Y, copy.deepcopy(b), mode), rtol=1e-12)
    slices = theta_slices(b)
    assert J_y.shape == (len(Y), len(Y)) and J_theta.shape == (len(Y), theta.size)
    assert np.all(J_y[:, -1] == 0)  # power does not feed back
    guess = "j" if mode == "voltage" else "V"
    for name in (guess, "fuel_pressure", "air_pressure"):
        assert np.all(J_theta[:, slices[name]] == 0)
    assert np.all(np.isfinite(J_y)) and np.all(np.isfinite(J_theta))

    # y columns of the steady tangents are the pol_deg / ohm_deg gradients of the solve
    S, dS = grad._steady_tangent(Y, b, mode)
    n_theta = theta.size
    assert dS.shape == S.shape + (n_theta + len(Y),)
    solve = grad.solve_for_voltage if mode == "voltage" else grad.solve_for_current
    pol_deg, ohm_deg = grad._deg_layout(Y)
    by_name = {
        "pol_deg": solve.grad_pol_deg(b, pol_deg, ohm_deg),
        "ohm_deg": solve.grad_ohm_deg(b, pol_deg, ohm_deg),
    }
    for index, name, element, layer in grad._deg_slots():
        np.testing.assert_allclose(dS[..., n_theta + index], by_name[name][..., element, layer], rtol=1e-10)
    np.testing.assert_array_equal(dS[..., -1], 0)
    np.testing.assert_array_equal(flatten(b), theta)


@pytest.mark.parametrize("mode", ["voltage", "current"])
def test_gradients_do_not_modify_boundary(mode):
    grad = cases.build_cell()
    b = cases.boundary(**OPERATION)
    theta = flatten(b)
    pol_deg, ohm_deg = grad._deg_layout(Y)
    solve = grad.solve_for_voltage if mode == "voltage" else grad.solve_for_current
    solve.grad_T(b, pol_deg, ohm_deg)
    grad.advance_time_step.grad_y(10.0, Y, b, mode)
    grad.advance_time_step.grad_n_fuel(10.0, Y, b, mode)
    np.testing.assert_array_equal(flatten(b), theta)


# Cell with Stefan-Maxwell fuel transport: values only (secant solves), no gradients.
def _stefan_maxwell_cell(ns):
    fuel = cases.fuel_electrode(ns, transport="StefanMaxwell", degradation=False)
    return ns.cell.Cell(
        16e-4, fuel, cases.electrolytes(ns)[:1], cases.air_electrode(ns, degradation=False), elements=3
    )


def test_stefan_maxwell_cell_values_match_base():
    base, grad = _stefan_maxwell_cell(cases.BASE), _stefan_maxwell_cell(cases.GRAD)
    b_base, b = cases.boundary(**OPERATION), cases.boundary(**OPERATION)
    solution_base = base.solve_for_voltage(copy.deepcopy(b_base))
    solution = grad.solve_for_voltage(copy.deepcopy(b))
    np.testing.assert_allclose(solution[1], solution_base[1], atol=10)  # base Newton tolerance [A/m^2]
    np.testing.assert_allclose(solution, solution_base, rtol=5e-3, atol=1e-12)
    solution_base = base.solve_for_current(b_base)
    solution = grad.solve_for_current(b)
    assert abs(b.V - b_base.V) < 5e-3  # base tolerance 5e-4 V on the Newton step
    assert abs(np.mean(solution[1]) - b.j) < 1e-6
    np.testing.assert_allclose(solution, solution_base, rtol=5e-3, atol=1e-12)
    # only the gradient methods raise
    with pytest.raises(NotImplementedError):
        grad.solve_for_voltage.grad_T(cases.boundary(**OPERATION))


# Cell with a water gas shift reaction: (H2, H2O, CO2, CO) fuel without gas transport.
def _wgs_cell(ns):
    fuel = cases.fuel_electrode(
        ns, fuel_species=("H2", "H2O", "CO2", "CO"), transport="PorousTransport", degradation=False
    )
    return ns.cell.Cell(
        16e-4, fuel, cases.electrolytes(ns)[:1], cases.air_electrode(ns, degradation=False),
        elements=3, reactions=(ns.cell.WaterGasShift(**cases.WGS),),
    )


def _wgs_boundary():
    b = cases.boundary(j=-3000.0, V=1.2)
    b.n_fuel = 8e-5 * np.array([0.1, 0.4, 0.45, 0.05])
    b.T = 1027.15
    return b


def test_water_gas_shift_cell_values_match_base():
    base, grad = _wgs_cell(cases.BASE), _wgs_cell(cases.GRAD)
    b = _wgs_boundary()
    np.testing.assert_array_equal(grad.dn_fuel(b.j, b), base.dn_fuel(b.j, b))
    np.testing.assert_array_equal(grad.Ps_fuel(b, b.j), base.Ps_fuel(b, b.j))
    Ps_fuel = grad.Ps_fuel(b, b.j)
    assert grad.V(b.j, b.T, Ps_fuel, PS_AIR) == base.V(b.j, b.T, Ps_fuel, PS_AIR)


@pytest.mark.parametrize("method_name", ["dn_fuel", "Ps_fuel"])
@pytest.mark.parametrize("variable", ["n_fuel", "T", "P"])
def test_water_gas_shift_cell_boundary_gradients(method_name, variable):
    grad = _wgs_cell(cases.GRAD)
    b = _wgs_boundary()
    method = getattr(grad, method_name)
    if method_name == "Ps_fuel":
        def call(boundary):
            return method(boundary, b.j)
        analytical = getattr(method, f"grad_{variable}")(b, b.j)
    else:
        def call(boundary):
            return method(b.j, boundary)
        analytical = getattr(method, f"grad_{variable}")(b.j, b)
    numerical = _numerical(_boundary_function(call, b, variable), getattr(b, variable))
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-6 * np.max(np.abs(numerical)))


def test_water_gas_shift_cell_current_gradients():
    grad = _wgs_cell(cases.GRAD)
    b = _wgs_boundary()
    for analytical, f in [
        (grad.dn_fuel.grad_j(b.j, b), lambda j: grad.dn_fuel(j, b)),
        (grad.Ps_fuel.grad_j(b, b.j), lambda j: grad.Ps_fuel(b, j)),
    ]:
        numerical = _numerical(f, b.j)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-6 * np.max(np.abs(numerical)))
    Ps_fuel = grad.Ps_fuel(b, b.j)
    numerical = _numerical(lambda Ps: grad.V(b.j, b.T, Ps, PS_AIR), Ps_fuel)
    np.testing.assert_allclose(grad.V.grad_Ps_fuel(b.j, b.T, Ps_fuel, PS_AIR), numerical, rtol=1e-5)


def test_water_gas_shift_cell_advance_step_area():
    base, grad = _wgs_cell(cases.BASE), _wgs_cell(cases.GRAD)
    b = _wgs_boundary()
    out_base = base.advance_step_area(copy.deepcopy(b))
    out = grad.advance_step_area(copy.deepcopy(b))
    assert abs(out.j - out_base.j) < 10
    residual = b.V - grad.V(out.j, b.T, grad.Ps_fuel(b, out.j), grad.Ps_air(b, out.j))
    assert abs(residual) < 1e-12
