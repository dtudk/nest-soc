"""
Ideal gas properties with analytical gradients.
"""

import numpy as np
from numpy.polynomial.chebyshev import chebder, chebval

from nest import properties
from nest.grad.differentiable import differentiable


def _unravel_specie_init(
    specie: properties.Specie,
) -> tuple[tuple[np.ndarray, np.ndarray], float, float]:
    """
    Constructor arguments of a specie.

    Parameters
    ----------
    specie : nest.properties.Specie
        specie to read from

    Returns
    -------
    tuple
        ``(cs, M, V)`` as accepted by :class:`nest.properties.Specie`
    """
    return specie.cs, specie.M, specie.V

class Specie(properties.Specie):
    """
    Ideal gas specie with analytical property gradients.

    Same constructor as :class:`nest.properties.Specie`; use :meth:`from_specie`
    to wrap an existing specie and reuse its fits.

    Parameters
    ----------
    cs : tuple[numpy.ndarray]
        NASA coefficients and temperature limits
    M : float
        molar mass [g/mol]
    V : float
        special atomic diffusion volumes [cm^3]
    """

    @classmethod
    def from_specie(cls, specie: properties.Specie) -> "Specie":
        """
        Create a gradient-aware copy of an existing specie.

        Parameters
        ----------
        specie : nest.properties.Specie
            specie to wrap, e.g. ``nest.properties.BasicSpecies.H2``

        Returns
        -------
        Specie
            new specie with the same data, reusing the wrapped specie's fits
        """
        new_specie = cls(*_unravel_specie_init(specie))
        return new_specie

    @differentiable
    def viscosity(self, T, P=101325.0):
        """
        Pure-gas viscosity [Pa s], see :meth:`nest.properties.Specie.viscosity`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        P : float or numpy.ndarray, optional
            Pressure [Pa], default 101325. Only affects output broadcasting.

        Returns
        -------
        viscosity : float or numpy.ndarray
            Viscosity [Pa s], with T and P broadcast to a common shape.
        """
        return super().viscosity(T, P)

    @viscosity.grad("T")
    def viscosity(self, T, P=101325.0):
        r"""
        Temperature derivative of the pure-gas viscosity [Pa s/K].

        .. math:: \frac{\partial \mu}{\partial T} = \frac{2}{T_{max} - T_{min}} \sum_k c_k T_k'(\theta)

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        P : float or numpy.ndarray, optional
            Pressure [Pa], default 101325. Only affects output broadcasting.

        Returns
        -------
        float or numpy.ndarray
            Viscosity derivative [Pa s/K], with T and P broadcast to a common shape.
        """
        viscosity_coefficients = self._viscosity_coefficients()
        T_min, T_max = properties._VISCOSITY_T_RANGE
        T = np.asarray(T, dtype=float)
        P = np.asarray(P, dtype=float)
        T, P = np.broadcast_arrays(T, P)
        theta = (2 * T - T_min - T_max) / (T_max - T_min)
        value = chebval(theta, chebder(viscosity_coefficients)) * 2 / (T_max - T_min)
        return float(value) if value.ndim == 0 else value


class BasicSpecies:
    """
    Gradient-aware versions of :class:`nest.properties.BasicSpecies`.
    """

    H2 = Specie.from_specie(properties.BasicSpecies.H2)
    H2O = Specie.from_specie(properties.BasicSpecies.H2O)
    O2 = Specie.from_specie(properties.BasicSpecies.O2)
    N2 = Specie.from_specie(properties.BasicSpecies.N2)
    CO2 = Specie.from_specie(properties.BasicSpecies.CO2)
    CO = Specie.from_specie(properties.BasicSpecies.CO)
    CH4 = Specie.from_specie(properties.BasicSpecies.CH4)
