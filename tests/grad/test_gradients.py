"""
General gradient check for every ``@differentiable`` method in ``nest.grad``.

Each method's registered gradients (``grad_<variable>``) are compared with a
central finite-difference Jacobian. New differentiable methods are picked up
automatically; new classes only need an entry in ``INSTANCES``, and new
argument names an entry in ``INPUTS`` (or ``CLASS_INPUTS``).
"""

import importlib
import inspect
import pkgutil
import warnings

import numpy as np
import pytest

import nest.grad
from nest.grad import layers, properties
from nest.grad.differentiable import differentiable

species = properties.BasicSpecies
# Kinetic parameters from nest/examples/eis_grid_independence.py.
FUEL = dict(
    gas=properties.Mixture((species.H2, species.H2O)), nu=[-1, 1], n_e=2,
    alpha=0.59, beta=0.41, theta=1, E_act=1.6e5, p=[-0.1, 0.33], gamma=60.0,
)
AIR = dict(
    gas=properties.Mixture((species.O2, species.N2)), nu=[-0.5, 0], n_e=2,
    alpha=0.65, beta=0.35, theta=1, E_act=1.496e5, p=[0.22, 0], gamma=10.0,
)

# Example objects to test each class on.
INSTANCES = {
    properties.Specie: [species.H2, species.H2O, species.O2, species.N2],
    properties.Mixture: [
        properties.Mixture((species.H2, species.H2O)),
        properties.Mixture((species.O2, species.N2)),
    ],
    layers.Kinetic: [layers.Kinetic(**FUEL), layers.Kinetic(**AIR)],
    layers.ButlerVolmer: [layers.ButlerVolmer(**FUEL), layers.ButlerVolmer(**AIR)],
}

# Sample values per argument name (vectors, so the full Jacobian is checked).
INPUTS = {
    "T": np.array([600.0, 950.0, 1400.0]),  # temperature [K]; avoids the 1000 K NASA range switch
    "P": np.array([1.0e5, 2.0e5, 3.0e5]),  # pressure [Pa]
    "j": 3000.0,  # current density [A/m^2]
    "j0": np.array([500.0, 1000.0, 2000.0]),  # exchange current density [A/m^2]
    "V": np.array([-0.05, 0.02, 0.1]),  # activation overpotential [V]
}

# Per-class overrides of INPUTS; each dict gives one set of test cases.
CLASS_INPUTS = {
    properties.Mixture: [{"xs": np.array([0.3, 0.7]), "i": 0, "j": 1}],  # j: species index
    layers.Kinetic: [
        {"Ps": np.array([4.0e4, 6.0e4])},  # partial pressures [Pa], one location
        {"Ps": np.array([[4.0e4, 5.0e4, 6.0e4], [6.0e4, 5.0e4, 4.0e4]])},  # 3 locations
    ],
}

# Relative difference between analytical and numerical Jacobian: fail / warn above.
FAIL_TOL = 1e-3
WARN_TOL = 1e-6


def _differentiable_methods():
    """Find every (class, method name, gradient variable) in nest.grad.

    Returns
    -------
    list of tuple
        ``(cls, method_name, variable)`` for each registered gradient
    """
    found = []
    for module_info in pkgutil.walk_packages(nest.grad.__path__, "nest.grad."):
        module = importlib.import_module(module_info.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue  # only classes defined in this module
            for method_name, attribute in vars(cls).items():
                if isinstance(attribute, differentiable):
                    found.extend((cls, method_name, v) for v in attribute.gradients)
    return found


def numerical_jacobian(f, x):
    """Central-difference Jacobian J[i, j] = d f_i / d x_j of f: R^n -> R^m.

    Perturbs one component x_j at a time (step relative to its size), so it
    makes no assumption about how the outputs depend on the inputs.

    Returns
    -------
    numpy.ndarray
        Jacobian of shape (m, n)
    """
    x = np.asarray(x, dtype=float)
    columns = []
    for j in range(x.size):
        h = 1e-6 * max(abs(x.flat[j]), 1.0)
        step = np.zeros_like(x)
        step.flat[j] = h  # perturb only x_j
        df = (np.asarray(f(x + step)) - np.asarray(f(x - step))) / (2 * h)
        columns.append(np.ravel(df))  # column j: change of every output f_i
    return np.stack(columns, axis=1)


def _cases():
    """Expand each gradient into one test case per example object.

    Returns
    -------
    list
        ``pytest.param`` objects with readable ids
    """
    cases = []
    for cls, method_name, variable in _differentiable_methods():
        for i, obj in enumerate(INSTANCES.get(cls, [None])):
            for k, inputs in enumerate(CLASS_INPUTS.get(cls, [{}])):
                case_id = f"{cls.__name__}.{method_name}.grad_{variable}[{i}-{k}]"
                cases.append(
                    pytest.param(cls, obj, method_name, variable, inputs, id=case_id)
                )
    return cases


def _full_jacobian(analytical, f_shape, x_shape, numerical_shape):
    """Bring an analytical gradient to the layout of the numerical Jacobian.

    Accepts the full Jacobian (any shape of the same size) or the location-wise
    form of shape ``x_shape``, where output location ``l`` only depends on
    ``x[..., l]`` (element-wise when ``x_shape == f_shape``).

    Returns
    -------
    numpy.ndarray or None
        Jacobian of shape ``numerical_shape``; None if the shape is not supported
    """
    if analytical.size == np.prod(numerical_shape, dtype=int):
        return analytical.reshape(numerical_shape)
    lead = len(x_shape) - len(f_shape)
    if lead < 0 or x_shape[lead:] != f_shape or analytical.shape != x_shape:
        return None
    full = np.zeros(f_shape + x_shape)
    leading = (slice(None),) * lead
    for location in np.ndindex(f_shape):
        full[location + leading + location] = analytical[leading + location]
    return full.reshape(numerical_shape)


def test_differentiable_methods_found():
    # Guard: the discovery must find something, otherwise the test below is empty.
    assert _differentiable_methods(), "no @differentiable methods found in nest.grad"


@pytest.mark.parametrize("cls, obj, method_name, variable, class_inputs", _cases())
def test_gradient_matches_finite_difference(cls, obj, method_name, variable, class_inputs):
    assert obj is not None, f"add example objects for {cls.__name__} to INSTANCES"
    method = getattr(obj, method_name)
    gradient = getattr(method, f"grad_{variable}")
    inputs = {**INPUTS, **class_inputs}

    # Fill the method's arguments: required ones and the differentiated one from
    # the inputs, optional ones keep their defaults.
    parameters = list(inspect.signature(vars(cls)[method_name].func).parameters.items())
    kwargs = {}
    for name, parameter in parameters[1:]:  # skip self
        if name == variable or parameter.default is inspect.Parameter.empty:
            assert name in inputs, f"add a sample value for argument {name!r} to INPUTS"
            kwargs[name] = inputs[name]
    x = kwargs[variable]

    def f(x_new):
        # The method as a function of the differentiated variable only.
        return method(**{**kwargs, variable: x_new})

    numerical = numerical_jacobian(f, x)
    analytical = np.asarray(gradient(**kwargs), dtype=float)
    f_shape = np.shape(f(x))
    full = _full_jacobian(analytical, f_shape, np.shape(x), numerical.shape)
    if full is None:
        pytest.fail(
            f"grad_{variable} has shape {analytical.shape}; expected the Jacobian "
            f"{f_shape + np.shape(x)} or location-wise {np.shape(x)}"
        )

    # Relative difference, scaled by the largest entry so zero entries are fine;
    # absolute difference if the true Jacobian is all zeros.
    scale = np.max(np.abs(numerical))
    rel_diff = np.max(np.abs(full - numerical)) / (scale if scale > 0 else 1.0)
    label = f"{cls.__name__}.{method_name}.grad_{variable}"
    if rel_diff > FAIL_TOL:
        pytest.fail(f"{label}: differs from finite difference by {rel_diff:.1e} (> {FAIL_TOL})")
    if rel_diff > WARN_TOL:
        warnings.warn(
            f"{label}: differs from finite difference by {rel_diff:.1e} (> {WARN_TOL})",
            stacklevel=1,
        )
