"""
General gradient check for every ``@differentiable`` method in ``nest.grad``.

Each method's registered gradients (``grad_<variable>``) are compared with a
central finite-difference Jacobian. New differentiable methods are picked up
automatically; new classes only need an entry in ``INSTANCES``, and new
argument names an entry in ``INPUTS`` (or ``CLASS_INPUTS`` / ``METHOD_INPUTS``).
Entries may also live in ``cases_*.py`` files next to this one, which define
the same dicts (their ``INSTANCES``, ``CLASS_INPUTS`` and ``METHOD_INPUTS``
lists extend the ones here).

A gradient variable that is not an argument of the method is looked up as an
attribute of its ``BoundaryData`` argument and perturbed on a copy.
"""

import copy
import importlib
import inspect
import pkgutil
import warnings
from pathlib import Path

import numpy as np
import pytest

import nest.grad
from nest.boundary import BoundaryData
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

# Per-method overrides of CLASS_INPUTS, keyed by "Class.method"; each dict gives one
# set of test cases. An optional "_zero" entry lists variables whose gradient must
# be exactly zero (e.g. Newton guesses); those are not compared with finite differences.
# "_slow": True marks the cases with ``pytest.mark.slow``.
METHOD_INPUTS = {}

# Smallest step scale per argument name (default 1); the step is 1e-6 * max(|x|, floor).
STEP_FLOOR = {}

# Cases checked elsewhere (e.g. too expensive for this harness): "Class.method" for
# every instance, "Class.method[i]" for instance i only.
EXCLUDE = set()

# Messages (regex) of expected warnings, ignored while computing the gradients.
EXPECTED_WARNINGS = set()

# Merge the dicts of every cases_*.py next to this file (importable, see conftest.py).
for _path in sorted(Path(__file__).parent.glob("cases_*.py")):
    _module = importlib.import_module(_path.stem)
    for _name in ("INPUTS", "STEP_FLOOR"):
        globals()[_name].update(getattr(_module, _name, {}))
    for _name in ("INSTANCES", "CLASS_INPUTS", "METHOD_INPUTS"):  # lists extend the existing ones
        for _cls, _items in getattr(_module, _name, {}).items():
            globals()[_name][_cls] = globals()[_name].get(_cls, []) + list(_items)
    EXCLUDE |= set(getattr(_module, "EXCLUDE", ()))
    EXPECTED_WARNINGS |= set(getattr(_module, "EXPECTED_WARNINGS", ()))

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


def numerical_jacobian(f, x, floor=1.0):
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
        h = 1e-6 * max(abs(x.flat[j]), floor)
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
        name = f"{cls.__name__}.{method_name}"
        if name in EXCLUDE:
            continue
        for i, obj in enumerate(INSTANCES.get(cls, [None])):
            if f"{name}[{i}]" in EXCLUDE:
                continue
            for k, inputs in enumerate(METHOD_INPUTS.get(name, CLASS_INPUTS.get(cls, [{}]))):
                case_id = f"{cls.__name__}.{method_name}.grad_{variable}[{i}-{k}]"
                marks = [pytest.mark.slow] if inputs.get("_slow") else []
                cases.append(
                    pytest.param(cls, obj, method_name, variable, inputs, id=case_id, marks=marks)
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
    # the inputs, optional ones keep their defaults (unless given in class_inputs).
    parameters = list(inspect.signature(vars(cls)[method_name].func).parameters.items())
    kwargs = {}
    for name, parameter in parameters[1:]:  # skip self
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        required = parameter.default is inspect.Parameter.empty
        if name == variable or required or name in class_inputs:
            assert name in inputs, f"add a sample value for argument {name!r} to INPUTS"
            kwargs[name] = inputs[name]

    # Variable that is a BoundaryData attribute rather than an argument.
    holder = None
    if variable not in kwargs:
        holder = next(
            (k for k, v in kwargs.items() if isinstance(v, BoundaryData)), None
        )
        assert holder is not None, f"{variable!r} is neither an argument nor a boundary attribute"
        x = getattr(kwargs[holder], variable)
    else:
        x = kwargs[variable]

    def call(func, x_new):
        # Fresh copies every call: some methods mutate their BoundaryData.
        args = {k: copy.deepcopy(v) if isinstance(v, BoundaryData) else v for k, v in kwargs.items()}
        if holder is None:
            args[variable] = x_new
        else:
            setattr(args[holder], variable, x_new)
        return func(**args)

    def f(x_new):
        # The method as a function of the differentiated variable only.
        return call(method, x_new)

    if variable in class_inputs.get("_zero", ()):
        # Structural zero (e.g. a Newton guess): finite differences only see solver noise.
        analytical = np.asarray(call(gradient, x), dtype=float)
        assert analytical.size == np.size(f(x)) * np.size(x)
        assert np.all(analytical == 0), f"grad_{variable} must be exactly zero"
        return

    try:
        numerical = numerical_jacobian(f, x, STEP_FLOOR.get(variable, 1.0))
        with warnings.catch_warnings():
            for message in EXPECTED_WARNINGS:
                warnings.filterwarnings("ignore", message)
            analytical = np.asarray(call(gradient, x), dtype=float)
    except NotImplementedError as error:
        pytest.skip(f"not implemented: {error}")
    f_shape = np.shape(f(x))
    full = _full_jacobian(analytical, f_shape, np.shape(x), numerical.shape)
    if full is None:
        pytest.fail(
            f"grad_{variable} has shape {analytical.shape}; expected the Jacobian "
            f"{f_shape + np.shape(x)} or location-wise {np.shape(x)}"
        )

    label = f"{cls.__name__}.{method_name}.grad_{variable}"
    # NaN would pass the tolerance checks below
    if not (np.all(np.isfinite(full)) and np.all(np.isfinite(numerical))):
        pytest.fail(f"{label}: non-finite analytical or finite-difference Jacobian")

    # Relative difference, scaled by the largest entry so zero entries are fine;
    # absolute difference if the true Jacobian is all zeros.
    scale = np.max(np.abs(numerical))
    rel_diff = np.max(np.abs(full - numerical)) / (scale if scale > 0 else 1.0)
    if rel_diff > FAIL_TOL:
        pytest.fail(f"{label}: differs from finite difference by {rel_diff:.1e} (> {FAIL_TOL})")
    if rel_diff > WARN_TOL:
        warnings.warn(
            f"{label}: differs from finite difference by {rel_diff:.1e} (> {WARN_TOL})",
            stacklevel=1,
        )
