"""
Array helpers shared by the ``nest.grad`` modules.
"""

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
