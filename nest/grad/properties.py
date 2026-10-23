"""
Ideal gas properties with analytical gradients.
"""

import numpy as np
from numpy.polynomial.chebyshev import chebder, chebval

from nest import properties
from nest.constants import P_0, R
from nest.grad._utils import as_output, elementwise, stack_species
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
    def _viscosity_grad_T(self, T, P=101325.0):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.properties.Specie.viscosity`

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

    @differentiable
    def cp(self, T):
        """
        Molar specific heat at constant pressure [J/mol.K], see :meth:`nest.properties.Specie.cp`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].

        Returns
        -------
        float or numpy.ndarray
            Specific heat [J/mol.K], same shape as T.
        """
        return elementwise(super().cp, T)

    @cp.grad("T")
    def _cp_grad_T(self, T):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.properties.Specie.cp`

        Returns
        -------
        float or numpy.ndarray
            Specific heat derivative [J/mol.K^2], same shape as T.
        """

        def dcp_dT(T):
            a = self.a(T)
            return R * (
                -2 * a[0] * T ** (-3)
                - a[1] * T ** (-2)
                + a[3]
                + 2 * a[4] * T
                + 3 * a[5] * T**2
                + 4 * a[6] * T**3
            )

        return elementwise(dcp_dT, T)

    @differentiable
    def h(self, T):
        """
        Molar specific enthalpy [J/mol], see :meth:`nest.properties.Specie.h`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].

        Returns
        -------
        float or numpy.ndarray
            Enthalpy [J/mol], same shape as T.
        """
        return elementwise(super().h, T)

    @h.grad("T")
    def _h_grad_T(self, T):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.properties.Specie.h`

        Returns
        -------
        float or numpy.ndarray
            Enthalpy derivative [J/mol.K] (equal to cp), same shape as T.
        """
        return self.cp(T)

    @differentiable
    def s(self, T, P):
        """
        Molar specific entropy [J/mol.K], see :meth:`nest.properties.Specie.s`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        P : float or numpy.ndarray
            Partial pressure [Pa].

        Returns
        -------
        float or numpy.ndarray
            Entropy [J/mol.K], with T and P broadcast to a common shape.
        """
        return elementwise(super().s, T, P)

    @s.grad("T")
    def _s_grad_T(self, T, P):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.properties.Specie.s`

        Returns
        -------
        float or numpy.ndarray
            Entropy derivative [J/mol.K^2] (cp / T), T and P broadcast.
        """
        return elementwise(lambda T, P: properties.Specie.cp(self, T) / T, T, P)

    @s.grad("P")
    def _s_grad_P(self, T, P):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.properties.Specie.s`

        Returns
        -------
        float or numpy.ndarray
            Entropy derivative [J/mol.K.Pa] (-R / P), T and P broadcast.
        """
        return elementwise(lambda T, P: -R / P, T, P)

    @differentiable
    def g(self, T, P):
        """
        Molar specific Gibbs free energy [J/mol], see :meth:`nest.properties.Specie.g`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        P : float or numpy.ndarray
            Partial pressure [Pa].

        Returns
        -------
        float or numpy.ndarray
            Gibbs free energy [J/mol], with T and P broadcast to a common shape.
        """
        return elementwise(super().g, T, P)

    @g.grad("T")
    def _g_grad_T(self, T, P):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.properties.Specie.g`

        Returns
        -------
        float or numpy.ndarray
            Gibbs free energy derivative [J/mol.K] (-s), T and P broadcast.
        """
        return as_output(-np.asarray(self.s(T, P)))

    @g.grad("P")
    def _g_grad_P(self, T, P):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.properties.Specie.g`

        Returns
        -------
        float or numpy.ndarray
            Gibbs free energy derivative [J/mol.Pa] (R T / P), T and P broadcast.
        """
        return elementwise(lambda T, P: R * T / P, T, P)

    @differentiable
    def gT(self, T):
        """
        Molar specific Gibbs free energy at reference pressure [J/mol], see :meth:`nest.properties.Specie.gT`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].

        Returns
        -------
        float or numpy.ndarray
            Gibbs free energy at P_0 [J/mol], same shape as T.
        """
        return elementwise(super().gT, T)

    @gT.grad("T")
    def _gT_grad_T(self, T):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.properties.Specie.gT`

        Returns
        -------
        float or numpy.ndarray
            Derivative [J/mol.K] (-s at P_0), same shape as T.
        """
        return as_output(-np.asarray(self.s(T, P_0)))


class Mixture(properties.Mixture):
    """
    Ideal gas mixture with analytical property gradients.

    Species are converted to :class:`Specie` so their gradients are available.
    Gradients w.r.t. ``xs`` have shape ``f.shape + (n_species,)`` for 1-D ``xs``,
    and ``(n_species,) + f.shape`` (location-wise) when ``xs`` has extra axes.

    Parameters
    ----------
    species : Specie or tuple of Specie
        species in mixture; :class:`nest.properties.Specie` instances are wrapped
    """

    def __init__(self, species):
        if isinstance(species, properties.Specie):
            species = (species,)
        converted = tuple(
            s if isinstance(s, Specie) else Specie.from_specie(s) for s in species
        )
        super().__init__(converted)
        self.species: tuple[Specie, ...] = converted

    @differentiable
    def viscosity(self, T, xs):
        """
        Gas mixture viscosity [Pa s], see :meth:`nest.properties.Mixture.viscosity`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        xs : numpy.ndarray
            Mole fractions, species on the first axis.

        Returns
        -------
        float or numpy.ndarray
            Viscosity [Pa s].
        """
        return super().viscosity(T, xs)

    def _wilke_terms(self, T, xs):
        """
        Intermediate terms of Wilke's mixing rule and their T-derivatives.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        xs : numpy.ndarray
            Mole fractions, species on the first axis.

        Returns
        -------
        tuple
            ``(mu, dmu, phi, dphi, D, dD)`` as lists indexed by species
        """
        x = np.asarray(xs, dtype=float)
        mu = [np.asarray(s.viscosity(T)) for s in self.species]
        dmu = [np.asarray(s.viscosity.grad_T(T)) for s in self.species]
        n = len(self.species)
        phi = [[None] * n for _ in range(n)]
        dphi = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                r = self.species[j].M / self.species[i].M
                c = r**0.25
                d = np.sqrt(8 * (1 + 1 / r))
                ratio = np.sqrt(mu[i] / mu[j])
                dratio = ratio / 2 * (dmu[i] / mu[i] - dmu[j] / mu[j])
                phi[i][j] = (1 + ratio * c) ** 2 / d
                dphi[i][j] = 2 * (1 + ratio * c) * c * dratio / d
        D = [sum(x[j] * phi[i][j] for j in range(n)) for i in range(n)]
        dD = [sum(x[j] * dphi[i][j] for j in range(n)) for i in range(n)]
        return mu, dmu, phi, dphi, D, dD

    @viscosity.grad("T")
    def _viscosity_grad_T(self, T, xs):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.properties.Mixture.viscosity`

        Returns
        -------
        float or numpy.ndarray
            Viscosity derivative [Pa s/K].
        """
        x = np.asarray(xs, dtype=float)
        mu, dmu, _, _, D, dD = self._wilke_terms(T, xs)
        return as_output(
            sum(x[i] * (dmu[i] * D[i] - mu[i] * dD[i]) / D[i] ** 2 for i in range(len(mu)))
        )

    @viscosity.grad("xs")
    def _viscosity_grad_xs(self, T, xs):
        r"""
        :math:`\nabla_{x}` :meth:`~nest.grad.properties.Mixture.viscosity`

        Returns
        -------
        float or numpy.ndarray
            Viscosity derivative per mole fraction [Pa s].
        """
        x = np.asarray(xs, dtype=float)
        mu, _, phi, _, D, _ = self._wilke_terms(T, xs)
        n = len(mu)
        terms = [
            mu[k] / D[k] - sum(x[i] * mu[i] * phi[i][k] / D[i] ** 2 for i in range(n))
            for k in range(n)
        ]
        return stack_species(terms, x, np.broadcast_shapes(x[0].shape, np.shape(T)))

    @differentiable
    def D_ij(self, i, j, T, P):
        """
        Binary diffusivities [m^2/s], see :meth:`nest.properties.Mixture.D_ij`.

        Parameters
        ----------
        i : int
            Index of the first species.
        j : int
            Index of the second species.
        T : float or numpy.ndarray
            Temperature [K].
        P : float or numpy.ndarray
            Pressure [Pa].

        Returns
        -------
        float or numpy.ndarray
            Binary diffusivity [m^2/s].
        """
        return super().D_ij(i, j, T, P)

    @D_ij.grad("T")
    def _D_ij_grad_T(self, i, j, T, P):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.properties.Mixture.D_ij`

        Returns
        -------
        float or numpy.ndarray
            Diffusivity derivative [m^2/s.K] (1.75 D / T).
        """
        return as_output(1.75 * super().D_ij(i, j, T, P) / np.asarray(T, dtype=float))

    @D_ij.grad("P")
    def _D_ij_grad_P(self, i, j, T, P):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.properties.Mixture.D_ij`

        Returns
        -------
        float or numpy.ndarray
            Diffusivity derivative [m^2/s.Pa] (-D / P).
        """
        return as_output(-super().D_ij(i, j, T, P) / np.asarray(P, dtype=float))

    @differentiable
    def cp(self, T, xs):
        """
        Molar specific heat at constant pressure [J/mol.K], see :meth:`nest.properties.Mixture.cp`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        xs : numpy.ndarray
            Mole fractions, species on the first axis.

        Returns
        -------
        float or numpy.ndarray
            Specific heat [J/mol.K].
        """
        return as_output(super().cp(T, xs))

    @cp.grad("T")
    def _cp_grad_T(self, T, xs):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.properties.Mixture.cp`

        Returns
        -------
        float or numpy.ndarray
            Specific heat derivative [J/mol.K^2].
        """
        x = np.asarray(xs, dtype=float)
        return as_output(sum(x[i] * s.cp.grad_T(T) for i, s in enumerate(self.species)))

    @cp.grad("xs")
    def _cp_grad_xs(self, T, xs):
        r"""
        :math:`\nabla_{x}` :meth:`~nest.grad.properties.Mixture.cp`

        Returns
        -------
        float or numpy.ndarray
            Specific heat per mole fraction [J/mol.K].
        """
        x = np.asarray(xs, dtype=float)
        terms = [s.cp(T) for s in self.species]
        return stack_species(terms, x, np.broadcast_shapes(x[0].shape, np.shape(T)))

    @differentiable
    def h(self, T, xs):
        """
        Molar specific enthalpy [J/mol], see :meth:`nest.properties.Mixture.h`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        xs : numpy.ndarray
            Mole fractions, species on the first axis.

        Returns
        -------
        float or numpy.ndarray
            Enthalpy [J/mol].
        """
        return as_output(super().h(T, xs))

    @h.grad("T")
    def _h_grad_T(self, T, xs):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.properties.Mixture.h`

        Returns
        -------
        float or numpy.ndarray
            Enthalpy derivative [J/mol.K].
        """
        return self.cp(T, xs)

    @h.grad("xs")
    def _h_grad_xs(self, T, xs):
        r"""
        :math:`\nabla_{x}` :meth:`~nest.grad.properties.Mixture.h`

        Returns
        -------
        float or numpy.ndarray
            Enthalpy per mole fraction [J/mol].
        """
        x = np.asarray(xs, dtype=float)
        terms = [s.h(T) for s in self.species]
        return stack_species(terms, x, np.broadcast_shapes(x[0].shape, np.shape(T)))

    @differentiable
    def s(self, T, P, xs):
        """
        Molar specific entropy [J/mol.K], see :meth:`nest.properties.Mixture.s`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        P : float or numpy.ndarray
            Pressure [Pa].
        xs : numpy.ndarray
            Mole fractions, species on the first axis.

        Returns
        -------
        float or numpy.ndarray
            Entropy [J/mol.K].
        """
        return as_output(super().s(T, P, xs))

    @s.grad("T")
    def _s_grad_T(self, T, P, xs):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.properties.Mixture.s`

        Returns
        -------
        float or numpy.ndarray
            Entropy derivative [J/mol.K^2].
        """
        x = np.asarray(xs, dtype=float)
        shape = np.broadcast_shapes(np.shape(T), np.shape(P), x[0].shape)
        value = sum(x[i] * s.s.grad_T(T, P * x[i]) for i, s in enumerate(self.species))
        return as_output(np.broadcast_to(value, shape))

    @s.grad("P")
    def _s_grad_P(self, T, P, xs):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.properties.Mixture.s`

        Returns
        -------
        float or numpy.ndarray
            Entropy derivative [J/mol.K.Pa].
        """
        x = np.asarray(xs, dtype=float)
        shape = np.broadcast_shapes(np.shape(T), np.shape(P), x[0].shape)
        value = -R * sum(x) / np.asarray(P, dtype=float)
        return as_output(np.broadcast_to(value, shape))

    @s.grad("xs")
    def _s_grad_xs(self, T, P, xs):
        r"""
        :math:`\nabla_{x}` :meth:`~nest.grad.properties.Mixture.s`

        Returns
        -------
        float or numpy.ndarray
            Entropy per mole fraction [J/mol.K].
        """
        x = np.asarray(xs, dtype=float)
        terms = [
            np.asarray(s.s(T, P * x[k])) - R for k, s in enumerate(self.species)
        ]
        shape = np.broadcast_shapes(np.shape(T), np.shape(P), x[0].shape)
        return stack_species(terms, x, shape)

    @differentiable
    def g(self, T, P, xs):
        """
        Molar specific Gibbs free energy [J/mol], see :meth:`nest.properties.Mixture.g`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        P : float or numpy.ndarray
            Pressure [Pa].
        xs : numpy.ndarray
            Mole fractions, species on the first axis.

        Returns
        -------
        float or numpy.ndarray
            Gibbs free energy [J/mol].
        """
        return as_output(super().g(T, P, xs))

    @g.grad("T")
    def _g_grad_T(self, T, P, xs):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.properties.Mixture.g`

        Returns
        -------
        float or numpy.ndarray
            Gibbs free energy derivative [J/mol.K].
        """
        x = np.asarray(xs, dtype=float)
        shape = np.broadcast_shapes(np.shape(T), np.shape(P), x[0].shape)
        value = -sum(x[i] * np.asarray(s.s(T, P * x[i])) for i, s in enumerate(self.species))
        return as_output(np.broadcast_to(value, shape))

    @g.grad("P")
    def _g_grad_P(self, T, P, xs):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.properties.Mixture.g`

        Returns
        -------
        float or numpy.ndarray
            Gibbs free energy derivative [J/mol.Pa].
        """
        x = np.asarray(xs, dtype=float)
        shape = np.broadcast_shapes(np.shape(T), np.shape(P), x[0].shape)
        value = R * np.asarray(T, dtype=float) * sum(x) / np.asarray(P, dtype=float)
        return as_output(np.broadcast_to(value, shape))

    @g.grad("xs")
    def _g_grad_xs(self, T, P, xs):
        r"""
        :math:`\nabla_{x}` :meth:`~nest.grad.properties.Mixture.g`

        Returns
        -------
        float or numpy.ndarray
            Gibbs free energy per mole fraction [J/mol].
        """
        x = np.asarray(xs, dtype=float)
        terms = [
            np.asarray(s.g(T, P * x[k])) + R * np.asarray(T, dtype=float)
            for k, s in enumerate(self.species)
        ]
        shape = np.broadcast_shapes(np.shape(T), np.shape(P), x[0].shape)
        return stack_species(terms, x, shape)


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
