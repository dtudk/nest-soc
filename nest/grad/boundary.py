"""
Helpers to treat :class:`nest.boundary.BoundaryData` as a flat parameter vector.
"""

import copy as _copy

import numpy as np

from nest.boundary import BoundaryData

THETA = ("V", "j", "n_fuel", "n_air", "T", "P", "fuel_pressure", "air_pressure")
"""Boundary attributes the gradients are taken with respect to, in vector order."""


def copy(boundary: BoundaryData) -> BoundaryData:
    """
    Independent copy, so gradient routines never mutate the caller's boundary.

    Parameters
    ----------
    boundary : BoundaryData
        boundary to copy

    Returns
    -------
    BoundaryData
        deep copy
    """
    return _copy.deepcopy(boundary)


def theta_slices(boundary: BoundaryData) -> dict:
    """
    Position of each attribute in the flat parameter vector.

    Parameters
    ----------
    boundary : BoundaryData
        boundary giving the number of fuel and air species

    Returns
    -------
    dict
        ``{name: slice}`` following :data:`THETA`
    """
    slices, start = {}, 0
    for name in THETA:
        size = np.size(getattr(boundary, name))
        slices[name] = slice(start, start + size)
        start += size
    return slices


def flatten(boundary: BoundaryData) -> np.ndarray:
    """
    Flat parameter vector of a boundary.

    Parameters
    ----------
    boundary : BoundaryData
        boundary to flatten

    Returns
    -------
    numpy.ndarray
        attributes of :data:`THETA` concatenated
    """
    return np.concatenate(
        [np.ravel(np.asarray(getattr(boundary, name), dtype=float)) for name in THETA]
    )


def unflatten(theta: np.ndarray, template: BoundaryData) -> BoundaryData:
    """
    Boundary with the attributes of :data:`THETA` taken from a flat vector.

    Parameters
    ----------
    theta : numpy.ndarray
        flat parameter vector
    template : BoundaryData
        boundary giving the species counts; not modified

    Returns
    -------
    BoundaryData
        new boundary
    """
    boundary = copy(template)
    for name, part in split(theta, template).items():
        setattr(boundary, name, part)
    return boundary


def split(vector: np.ndarray, boundary: BoundaryData) -> dict:
    """
    Split a flat vector (e.g. a gradient) into the attributes of :data:`THETA`.

    Parameters
    ----------
    vector : numpy.ndarray
        flat vector, parameters on the last axis
    boundary : BoundaryData
        boundary giving the species counts

    Returns
    -------
    dict
        ``{name: value}``; floats for scalar attributes, arrays for flows
    """
    vector = np.asarray(vector, dtype=float)
    parts = {}
    for name, index in theta_slices(boundary).items():
        part = vector[..., index]
        if np.ndim(getattr(boundary, name)) == 0:
            part = part[..., 0]
            part = float(part) if part.ndim == 0 else part
        parts[name] = part
    return parts
