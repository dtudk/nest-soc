"""
Array helpers shared by the ``nest.grad`` modules.
"""

import warnings

import numpy as np


def as_output(value):
    """
    Return a plain float for 0-d results, otherwise an array.

    Parameters
    ----------
    value : float or numpy.ndarray
        result to convert

    Returns
    -------
    float or numpy.ndarray
        ``float(value)`` if 0-d, else ``value`` as an array
    """
    value = np.asarray(value, dtype=float)
    return float(value) if value.ndim == 0 else value


def elementwise(func, *args):
    """
    Apply a scalar-only function element by element, with broadcasting.

    Parameters
    ----------
    func : callable
        function of scalars returning a scalar
    *args : float or numpy.ndarray
        arguments, broadcast to a common shape

    Returns
    -------
    float or numpy.ndarray
        ``func`` applied to each element; float for scalar arguments
    """
    return as_output(np.vectorize(func, otypes=[float])(*args))


def stack_species(terms, xs, shape):
    """
    Stack per-species derivative terms into a gradient w.r.t. mole fractions.

    Parameters
    ----------
    terms : list of float or numpy.ndarray
        derivative w.r.t. each species' mole fraction
    xs : numpy.ndarray
        mole fractions, species on the first axis
    shape : tuple of int
        output shape of the differentiated function

    Returns
    -------
    float or numpy.ndarray
        ``shape + (n_species,)`` for 1-D ``xs`` (full Jacobian), else
        ``(n_species,) + shape`` (location-wise)
    """
    terms = [np.broadcast_to(term, shape) for term in terms]
    return as_output(np.stack(terms, axis=-1 if np.ndim(xs) == 1 else 0))


def normalize_jacobian(u):
    r"""
    Jacobian of the mole fractions ``u / sum(u)`` w.r.t. the amounts ``u``.

    .. math:: \partial x / \partial u = (I - x \mathbf{1}^T) / \sum u

    Parameters
    ----------
    u : numpy.ndarray
        amounts or flows, 1-D

    Returns
    -------
    numpy.ndarray
        Jacobian of shape ``(len(u), len(u))``
    """
    u = np.asarray(u, dtype=float)
    total = u.sum()
    x = u / total
    return (np.eye(u.size) - np.outer(x, np.ones(u.size))) / total


def central_difference(func, x, step=None, warn=True):
    """
    Central finite-difference derivative, used where no analytical gradient exists.

    Parameters
    ----------
    func : callable
        function of ``x`` returning a float or array
    x : float or numpy.ndarray
        point to differentiate at
    step : float or numpy.ndarray, optional
        absolute step, or one per component of ``x``; default ``1e-6 * max(|x_i|, 1)`` per component
    warn : bool, optional
        emit a ``RuntimeWarning`` that a numerical fallback is used (False if the caller warns)

    Returns
    -------
    float or numpy.ndarray
        ``f.shape + x.shape`` derivative
    """
    if warn:
        warnings.warn(
            "no analytical gradient available; using central differences",
            RuntimeWarning,
            stacklevel=2,
        )
    x = np.asarray(x, dtype=float)
    steps = None if step is None else np.broadcast_to(np.asarray(step, dtype=float), x.shape)
    columns = []
    for k in range(x.size):
        h = steps.flat[k] if steps is not None else 1e-6 * max(abs(x.flat[k]), 1.0)
        dx = np.zeros_like(x)
        dx.flat[k] = h
        df = (np.asarray(func(x + dx), dtype=float) - np.asarray(func(x - dx), dtype=float)) / (2 * h)
        columns.append(df)
    value = np.stack(columns, axis=-1).reshape(np.shape(columns[0]) + x.shape)
    return as_output(value)
