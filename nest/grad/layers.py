"""
Electrochemical kinetics with analytical gradients.
"""

import numpy as np

from nest import layers
from nest.constants import F, P_0, R
from nest.grad._utils import as_output
from nest.grad.differentiable import differentiable
from nest.grad.properties import Mixture


class Kinetic(layers.Kinetic):
    """
    Kinetic model with analytical gradients.

    Same parameters as :class:`nest.layers.Kinetic`; the gas is converted to a
    gradient-aware :class:`nest.grad.properties.Mixture`. Gradients w.r.t. a
    multi-location ``Ps`` (species on the first axis) are location-wise, with
    shape ``Ps.shape``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gas = Mixture(self.gas.species)

    @differentiable
    def j_0(self, T, Ps):
        """
        Exchange current density [A/m^2] or current per TPB length [A/m], see :meth:`nest.layers.Kinetic.j_0`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        Ps : numpy.ndarray
            Partial pressures [Pa], species on the first axis.

        Returns
        -------
        float or numpy.ndarray
            Exchange current density.
        """
        return as_output(super().j_0(T, Ps))

    @j_0.grad("T")
    def _j_0_grad_T(self, T, Ps):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.Kinetic.j_0`

        Returns
        -------
        float or numpy.ndarray
            Exchange current density derivative per K.
        """
        T = np.asarray(T, dtype=float)
        return as_output(super().j_0(T, Ps) * (self.theta / T + self.E_act / (R * T**2)))

    @j_0.grad("Ps")
    def _j_0_grad_Ps(self, T, Ps):
        r"""
        :math:`\nabla_{P_s}` :meth:`~nest.grad.layers.Kinetic.j_0`

        Returns
        -------
        float or numpy.ndarray
            Exchange current density derivative per Pa; full Jacobian for 1-D Ps,
            location-wise (shape ``Ps.shape``) otherwise.
        """
        Ps = np.atleast_1d(np.asarray(Ps, dtype=float))
        j0 = np.asarray(super().j_0(T, Ps))
        if Ps.ndim == 1:
            return as_output(np.multiply.outer(j0, self.p / Ps))
        p = self.p.reshape((-1,) + (1,) * (Ps.ndim - 1))
        return as_output(j0 * p / Ps)

    @differentiable
    def mol_flux(self, j):
        """
        Molar flux [mol/(m^2 s)], see :meth:`nest.layers.Kinetic.mol_flux`.

        Parameters
        ----------
        j : float or numpy.ndarray
            Current density [A/m^2].

        Returns
        -------
        numpy.ndarray
            Molar flux of each species.
        """
        return as_output(super().mol_flux(j))

    @mol_flux.grad("j")
    def _mol_flux_grad_j(self, j):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.layers.Kinetic.mol_flux`

        Returns
        -------
        numpy.ndarray
            Molar flux derivative [mol/(A s)] (nu / (n_e F)).
        """
        return as_output(self.nu / (self.n_e * F) * np.ones(np.shape(j)))

    @differentiable
    def V_nernst_half(self, T, Ps):
        """
        Half-reaction Nernst voltage [V], see :meth:`nest.layers.Kinetic.V_nernst_half`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        Ps : numpy.ndarray
            Partial pressures [Pa], species on the first axis.

        Returns
        -------
        float or numpy.ndarray
            Half-reaction Nernst voltage [V].
        """
        return as_output(super().V_nernst_half(T, Ps))

    @V_nernst_half.grad("T")
    def _V_nernst_half_grad_T(self, T, Ps):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.Kinetic.V_nernst_half`

        Returns
        -------
        float or numpy.ndarray
            Voltage derivative [V/K].
        """
        if np.ndim(Ps) > 1:
            value = np.asarray(self.Vt_nernst_half.grad_T(T))
            for nu, partial_pressure in zip(self.nu, Ps):
                if nu != 0:
                    value = value + R / (self.n_e * F) * nu * np.log(partial_pressure / P_0)
            return as_output(value)
        return as_output(
            -sum(
                nu_i * np.asarray(gas_i.s(T, Ps_i))
                for nu_i, gas_i, Ps_i in zip(self.nu, self.gas.species, Ps)
            )
            / (self.n_e * F)
        )

    @V_nernst_half.grad("Ps")
    def _V_nernst_half_grad_Ps(self, T, Ps):
        r"""
        :math:`\nabla_{P_s}` :meth:`~nest.grad.layers.Kinetic.V_nernst_half`

        Returns
        -------
        float or numpy.ndarray
            Voltage derivative [V/Pa]; full Jacobian for 1-D Ps, location-wise
            (shape ``Ps.shape``) otherwise.
        """
        Ps = np.asarray(Ps, dtype=float)
        T = np.asarray(T, dtype=float)
        if Ps.ndim > 1:
            nu = self.nu.reshape((-1,) + (1,) * (Ps.ndim - 1))
            return as_output(np.where(nu != 0, R * T * nu / (self.n_e * F * Ps), 0.0))
        return as_output(np.multiply.outer(T, R * self.nu / (self.n_e * F * Ps)))

    @differentiable
    def V_thermoneutral_half(self, T):
        """
        Half-reaction thermoneutral voltage [V], see :meth:`nest.layers.Kinetic.V_thermoneutral_half`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].

        Returns
        -------
        float or numpy.ndarray
            Half-reaction thermoneutral voltage [V].
        """
        return as_output(super().V_thermoneutral_half(T))

    @V_thermoneutral_half.grad("T")
    def _V_thermoneutral_half_grad_T(self, T):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.Kinetic.V_thermoneutral_half`

        Returns
        -------
        float or numpy.ndarray
            Voltage derivative [V/K].
        """
        return as_output(
            sum(nu_i * np.asarray(gas_i.cp(T)) for nu_i, gas_i in zip(self.nu, self.gas.species))
            / (self.n_e * F)
        )

    @differentiable
    def Vt_nernst_half(self, T):
        """
        Half-reaction voltage at reference pressure [V], see :meth:`nest.layers.Kinetic.Vt_nernst_half`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].

        Returns
        -------
        float or numpy.ndarray
            Half-reaction voltage at reference pressure [V].
        """
        return as_output(super().Vt_nernst_half(T))

    @Vt_nernst_half.grad("T")
    def _Vt_nernst_half_grad_T(self, T):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.Kinetic.Vt_nernst_half`

        Returns
        -------
        float or numpy.ndarray
            Voltage derivative [V/K].
        """
        return as_output(
            -sum(
                nu_i * np.asarray(gas_i.s(T, P_0))
                for nu_i, gas_i in zip(self.nu, self.gas.species)
            )
            / (self.n_e * F)
        )

    @differentiable
    def V(self, j, T, j0):
        """
        Voltage [V]; always 0 for the base kinetic model, see :meth:`nest.layers.Kinetic.V`.

        Parameters
        ----------
        j : float or numpy.ndarray
            Current density [A/m^2].
        T : float or numpy.ndarray
            Temperature [K].
        j0 : float or numpy.ndarray
            Exchange current density [A/m^2].

        Returns
        -------
        float
            Voltage [V]; always 0.
        """
        return super().V(j, T, j0)

    @V.grad("j")
    def _V_grad_j(self, j, T, j0):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.layers.Kinetic.V`

        Returns
        -------
        float or numpy.ndarray
            Always 0, same shape as j.
        """
        return as_output(np.zeros(np.shape(j)))

    @V.grad("T")
    def _V_grad_T(self, j, T, j0):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.Kinetic.V`

        Returns
        -------
        float or numpy.ndarray
            Always 0, same shape as T.
        """
        return as_output(np.zeros(np.shape(T)))

    @V.grad("j0")
    def _V_grad_j0(self, j, T, j0):
        r"""
        :math:`\nabla_{j_0}` :meth:`~nest.grad.layers.Kinetic.V`

        Returns
        -------
        float or numpy.ndarray
            Always 0, same shape as j0.
        """
        return as_output(np.zeros(np.shape(j0)))


class ButlerVolmer(layers.ButlerVolmer, Kinetic):
    """
    Butler-Volmer kinetics with analytical gradients.

    Same parameters as :class:`nest.layers.ButlerVolmer`. All gradients are
    element-wise in ``V``, ``T``, and ``j0``.
    """

    def _rates(self, V, T):
        """
        Forward and reverse exponent factors.

        Parameters
        ----------
        V : float or numpy.ndarray
            Activation overpotential [V].
        T : float or numpy.ndarray
            Temperature [K].

        Returns
        -------
        tuple
            ``(ak, bk, ea, eb)`` with ``ak = alpha n_e F / (R T)``,
            ``bk = beta n_e F / (R T)``, ``ea = exp(ak V)``, ``eb = exp(-bk V)``
        """
        V, T = np.asarray(V, dtype=float), np.asarray(T, dtype=float)
        k = self.n_e * F / (R * T)
        ak, bk = self.alpha * k, self.beta * k
        return ak, bk, np.exp(ak * V), np.exp(-bk * V)

    @differentiable
    def j(self, V, T, j0):
        """
        Current density [A/m^2] or current per TPB length [A/m], see :meth:`nest.layers.ButlerVolmer.j`.

        Parameters
        ----------
        V : float or numpy.ndarray
            Activation overpotential [V].
        T : float or numpy.ndarray
            Temperature [K].
        j0 : float or numpy.ndarray
            Exchange current density.

        Returns
        -------
        float or numpy.ndarray
            Current density.
        """
        return as_output(super().j(V, T, j0))

    @j.grad("V")
    def _j_grad_V(self, V, T, j0):
        r"""
        :math:`\nabla_V` :meth:`~nest.grad.layers.ButlerVolmer.j`

        Returns
        -------
        float or numpy.ndarray
            Current density derivative per V (equal to ``dj_dV``).
        """
        return self.dj_dV(V, T, j0)

    @j.grad("T")
    def _j_grad_T(self, V, T, j0):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.ButlerVolmer.j`

        Returns
        -------
        float or numpy.ndarray
            Current density derivative per K.
        """
        V, T = np.asarray(V, dtype=float), np.asarray(T, dtype=float)
        return as_output(-V / T * np.asarray(self.dj_dV(V, T, j0)))

    @j.grad("j0")
    def _j_grad_j0(self, V, T, j0):
        r"""
        :math:`\nabla_{j_0}` :meth:`~nest.grad.layers.ButlerVolmer.j`

        Returns
        -------
        float or numpy.ndarray
            Current density per unit exchange current density [-].
        """
        ak, bk, _, _ = self._rates(V, T)
        V = np.asarray(V, dtype=float)
        value = np.expm1(ak * V) - np.expm1(-bk * V)
        return as_output(np.broadcast_to(value, np.broadcast_shapes(value.shape, np.shape(j0))))

    @differentiable
    def dj_dV(self, V, T, j0):
        """
        First derivative of current density by voltage, see :meth:`nest.layers.ButlerVolmer.dj_dV`.

        Parameters
        ----------
        V : float or numpy.ndarray
            Activation overpotential [V].
        T : float or numpy.ndarray
            Temperature [K].
        j0 : float or numpy.ndarray
            Exchange current density.

        Returns
        -------
        float or numpy.ndarray
            Current density derivative per V.
        """
        return as_output(super().dj_dV(V, T, j0))

    @dj_dV.grad("V")
    def _dj_dV_grad_V(self, V, T, j0):
        r"""
        :math:`\nabla_V` :meth:`~nest.grad.layers.ButlerVolmer.dj_dV`

        Returns
        -------
        float or numpy.ndarray
            Second derivative per V (equal to ``dj_dV2``).
        """
        return self.dj_dV2(V, T, j0)

    @dj_dV.grad("T")
    def _dj_dV_grad_T(self, V, T, j0):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.ButlerVolmer.dj_dV`

        Returns
        -------
        float or numpy.ndarray
            Derivative per V and K.
        """
        ak, bk, ea, eb = self._rates(V, T)
        V, T = np.asarray(V, dtype=float), np.asarray(T, dtype=float)
        j0 = np.asarray(j0, dtype=float)
        return as_output(-j0 / T * (ak * ea * (1 + ak * V) + bk * eb * (1 - bk * V)))

    @dj_dV.grad("j0")
    def _dj_dV_grad_j0(self, V, T, j0):
        r"""
        :math:`\nabla_{j_0}` :meth:`~nest.grad.layers.ButlerVolmer.dj_dV`

        Returns
        -------
        float or numpy.ndarray
            Derivative per V and unit exchange current density.
        """
        ak, bk, ea, eb = self._rates(V, T)
        value = ak * ea + bk * eb
        return as_output(np.broadcast_to(value, np.broadcast_shapes(value.shape, np.shape(j0))))

    @differentiable
    def dj_dV2(self, V, T, j0):
        """
        Second derivative of current density by voltage, see :meth:`nest.layers.ButlerVolmer.dj_dV2`.

        Parameters
        ----------
        V : float or numpy.ndarray
            Activation overpotential [V].
        T : float or numpy.ndarray
            Temperature [K].
        j0 : float or numpy.ndarray
            Exchange current density.

        Returns
        -------
        float or numpy.ndarray
            Current density second derivative per V^2.
        """
        return as_output(super().dj_dV2(V, T, j0))

    @dj_dV2.grad("V")
    def _dj_dV2_grad_V(self, V, T, j0):
        r"""
        :math:`\nabla_V` :meth:`~nest.grad.layers.ButlerVolmer.dj_dV2`

        Returns
        -------
        float or numpy.ndarray
            Third derivative per V.
        """
        ak, bk, ea, eb = self._rates(V, T)
        return as_output(np.asarray(j0, dtype=float) * (ak**3 * ea + bk**3 * eb))

    @dj_dV2.grad("T")
    def _dj_dV2_grad_T(self, V, T, j0):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.ButlerVolmer.dj_dV2`

        Returns
        -------
        float or numpy.ndarray
            Derivative per V^2 and K.
        """
        ak, bk, ea, eb = self._rates(V, T)
        V, T = np.asarray(V, dtype=float), np.asarray(T, dtype=float)
        j0 = np.asarray(j0, dtype=float)
        return as_output(
            -j0 / T * (ak**2 * ea * (2 + ak * V) - bk**2 * eb * (2 - bk * V))
        )

    @dj_dV2.grad("j0")
    def _dj_dV2_grad_j0(self, V, T, j0):
        r"""
        :math:`\nabla_{j_0}` :meth:`~nest.grad.layers.ButlerVolmer.dj_dV2`

        Returns
        -------
        float or numpy.ndarray
            Derivative per V^2 and unit exchange current density.
        """
        ak, bk, ea, eb = self._rates(V, T)
        value = ak**2 * ea - bk**2 * eb
        return as_output(np.broadcast_to(value, np.broadcast_shapes(value.shape, np.shape(j0))))

    @differentiable
    def V(self, j, T, j0):
        """
        Activation overpotential [V], see :meth:`nest.layers.ButlerVolmer.V`.

        Parameters
        ----------
        j : float or numpy.ndarray
            Current density.
        T : float or numpy.ndarray
            Temperature [K].
        j0 : float or numpy.ndarray
            Exchange current density.

        Returns
        -------
        float or numpy.ndarray
            Activation overpotential [V].
        """
        return as_output(super().V(j, T, j0))

    @V.grad("j")
    def _V_grad_j(self, j, T, j0):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.layers.ButlerVolmer.V`

        Returns
        -------
        float or numpy.ndarray
            Overpotential derivative per unit current density (1 / ``dj_dV``).
        """
        V = self.V(j, T, j0)
        return as_output(1 / np.asarray(self.dj_dV(V, T, j0)))

    @V.grad("T")
    def _V_grad_T(self, j, T, j0):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.ButlerVolmer.V`

        Returns
        -------
        float or numpy.ndarray
            Overpotential derivative [V/K] (V / T).
        """
        return as_output(np.asarray(self.V(j, T, j0)) / np.asarray(T, dtype=float))

    @V.grad("j0")
    def _V_grad_j0(self, j, T, j0):
        r"""
        :math:`\nabla_{j_0}` :meth:`~nest.grad.layers.ButlerVolmer.V`

        Returns
        -------
        float or numpy.ndarray
            Overpotential derivative per unit exchange current density.
        """
        V = self.V(j, T, j0)
        j, j0 = np.asarray(j, dtype=float), np.asarray(j0, dtype=float)
        return as_output(-(j / j0) / np.asarray(self.dj_dV(V, T, j0)))
