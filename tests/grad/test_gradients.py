"""
General gradient check for every ``@differentiable`` method in ``nest.grad``.

Each method's registered gradients (``grad_<variable>``) are compared with a
central finite-difference Jacobian. New differentiable methods are picked up
automatically; new classes only need an entry in ``INSTANCES``, and new
argument names an entry in ``INPUTS``.
"""

import importlib
import inspect
import pkgutil
import warnings

import numpy as np
import pytest

import nest.grad
from nest.grad import properties
from nest.grad.differentiable import differentiable

# Example objects to test each class on.
INSTANCES = {
    properties.Specie: [
        properties.BasicSpecies.H2,
        properties.BasicSpecies.H2O,
        properties.BasicSpecies.O2,
        properties.BasicSpecies.N2,
    ],
}

# Sample values per argument name (vectors, so the full Jacobian is checked).
INPUTS = {
    "T": np.array([600.0, 1000.0, 1400.0]),  # temperature [K]
    "P": np.array([1.0e5, 2.0e5, 3.0e5]),  # pressure [Pa]
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
            case_id = f"{cls.__name__}.{method_name}.grad_{variable}[{i}]"
            cases.append(pytest.param(cls, obj, method_name, variable, id=case_id))
    return cases


def test_differentiable_methods_found():
    # Guard: the discovery must find something, otherwise the test below is empty.
    assert _differentiable_methods(), "no @differentiable methods found in nest.grad"


@pytest.mark.parametrize("cls, obj, method_name, variable", _cases())
def test_gradient_matches_finite_difference(cls, obj, method_name, variable):
    assert obj is not None, f"add example objects for {cls.__name__} to INSTANCES"
    method = getattr(obj, method_name)
    gradient = getattr(method, f"grad_{variable}")

    # Fill the method's arguments: required ones and the differentiated one from
    # INPUTS, optional ones keep their defaults.
    parameters = list(inspect.signature(vars(cls)[method_name].func).parameters.items())
    kwargs = {}
    for name, parameter in parameters[1:]:  # skip self
        if name == variable or parameter.default is inspect.Parameter.empty:
            assert name in INPUTS, f"add a sample value for argument {name!r} to INPUTS"
            kwargs[name] = INPUTS[name]
    x = kwargs[variable]

    def f(x_new):
        # The method as a function of the differentiated variable only.
        return method(**{**kwargs, variable: x_new})

    numerical = numerical_jacobian(f, x)
    analytical = np.asarray(gradient(**kwargs), dtype=float)
    if analytical.shape == numerical.shape:
        pass  # gradient returns the full Jacobian
    elif analytical.shape == np.shape(x) and numerical.shape[0] == np.size(x):
        analytical = np.diag(np.ravel(analytical))  # element-wise: Jacobian diagonal
    else:
        pytest.fail(
            f"grad_{variable} has shape {analytical.shape}; expected the Jacobian "
            f"{numerical.shape} or element-wise {np.shape(x)}"
        )

    # Relative difference, scaled by the largest entry so zero entries are fine.
    rel_diff = np.max(np.abs(analytical - numerical)) / np.max(np.abs(numerical))
    label = f"{cls.__name__}.{method_name}.grad_{variable}"
    if rel_diff > FAIL_TOL:
        pytest.fail(f"{label}: differs from finite difference by {rel_diff:.1e} (> {FAIL_TOL})")
    if rel_diff > WARN_TOL:
        warnings.warn(
            f"{label}: differs from finite difference by {rel_diff:.1e} (> {WARN_TOL})",
            stacklevel=1,
        )
