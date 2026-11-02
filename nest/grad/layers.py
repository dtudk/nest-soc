"""
Electrochemical kinetics with analytical gradients.
"""

import numpy as np

from nest import layers
from nest.constants import F, P_0, R
from nest.grad._utils import as_output
from nest.grad.degradation import DegradationGradients, NoDegradation
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
        # p broadcasts over Ps like in the base (length-1 p applies to every species);
        # species with p = 0 do not depend on Ps, also where Ps = 0.
        p = np.broadcast_to(self.p.reshape((-1,) + (1,) * (Ps.ndim - 1)), Ps.shape)
        ratio = np.divide(p, Ps, out=np.zeros(Ps.shape), where=p != 0)
        if Ps.ndim == 1:
            return as_output(np.multiply.outer(j0, ratio))
        return as_output(j0 * ratio)

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
        # The base zips nu with Ps (and the species for 1-D Ps): extra Ps entries
        # do not contribute, so nu is zero-padded to len(Ps).
        n_used = min(len(self.nu), len(Ps), len(self.gas.species) if Ps.ndim == 1 else len(Ps))
        nu = np.zeros(len(Ps))
        nu[:n_used] = self.nu[:n_used]
        if Ps.ndim > 1:
            nu = nu.reshape((-1,) + (1,) * (Ps.ndim - 1))
            shape = np.broadcast_shapes(T.shape, Ps.shape)
            numerator = np.broadcast_to(R * T * nu / (self.n_e * F), shape)
            where = np.broadcast_to(nu != 0, shape)
            return as_output(np.divide(numerator, Ps, out=np.zeros(shape), where=where))
        coefficient = R * nu / (self.n_e * F)
        ratio = np.divide(coefficient, Ps, out=np.zeros(Ps.shape), where=coefficient != 0)
        return as_output(np.multiply.outer(T, ratio))

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


class Conductivity(layers.Conductivity):
    """
    Conductivity model with analytical gradients.

    Same parameters as :class:`nest.layers.Conductivity`. Gradients are
    element-wise in ``T``. Use :meth:`resistivity` for derivatives that stay
    finite for the default ``sigma0 = inf`` (no resistance).
    """

    @differentiable
    def sigma(self, T):
        """
        Material conductivity [S/m], see :meth:`nest.layers.Conductivity.sigma`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].

        Returns
        -------
        float or numpy.ndarray
            Conductivity [S/m].
        """
        return as_output(super().sigma(T))

    @sigma.grad("T")
    def _sigma_grad_T(self, T):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.Conductivity.sigma`

        Returns
        -------
        float or numpy.ndarray
            Conductivity derivative [S/m.K]; 0 for ``sigma0 = inf`` (constant).
        """
        T = np.asarray(T, dtype=float)
        if np.isinf(self.sigma0):
            return as_output(np.zeros(T.shape))
        return as_output(super().sigma(T) * (self.theta / T + self.E_act / (R * T**2)))

    @differentiable
    def resistivity(self, T):
        """
        Material resistivity [Ohm m], the inverse of :meth:`sigma`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].

        Returns
        -------
        float or numpy.ndarray
            Resistivity [Ohm m]; 0 for ``sigma0 = inf``.
        """
        return as_output(1 / np.asarray(super().sigma(T), dtype=float))

    @resistivity.grad("T")
    def _resistivity_grad_T(self, T):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.Conductivity.resistivity`

        Returns
        -------
        float or numpy.ndarray
            Resistivity derivative [Ohm m/K].
        """
        T = np.asarray(T, dtype=float)
        rho = np.asarray(self.resistivity(T))
        return as_output(-rho * (self.theta / T + self.E_act / (R * T**2)))


def _grad_mixture(gas):
    """
    Gradient-aware version of a gas mixture.

    Parameters
    ----------
    gas : nest.properties.Mixture
        mixture of species

    Returns
    -------
    Mixture
        ``gas`` itself if it already is a :class:`nest.grad.properties.Mixture`
    """
    return gas if isinstance(gas, Mixture) else Mixture(gas.species)


def _location_jacobian(derivative, x):
    """
    Full Jacobian of a species-first output whose locations depend element-wise on ``x``.

    Parameters
    ----------
    derivative : numpy.ndarray
        element-wise derivative, shape ``(n_species,) + loc``
    x : float or numpy.ndarray
        variable; scalar, or of shape ``loc``

    Returns
    -------
    float or numpy.ndarray
        ``derivative`` for scalar ``x``, else shape ``(n_species,) + loc + loc``
    """
    derivative = np.asarray(derivative, dtype=float)
    x_shape = np.shape(x)
    if not x_shape:
        return as_output(derivative)
    # a derivative without location axes is the same at every location
    derivative = derivative.reshape(derivative.shape + (1,) * (1 + len(x_shape) - derivative.ndim))
    derivative = np.broadcast_to(derivative, derivative.shape[:1] + x_shape)
    n, size = derivative.shape[0], int(np.prod(x_shape))
    full = np.zeros((n, size, size))
    full[:, np.arange(size), np.arange(size)] = derivative.reshape(n, size)
    return as_output(full.reshape((n,) + x_shape + x_shape))


class PorousTransport(layers.PorousTransport):
    """
    Porous transport model with analytical gradients.

    Same parameters as :class:`nest.layers.PorousTransport`. ``dP_dl`` takes a
    single location: scalar ``T``, 1-D ``mol_flux`` and partial pressures ``P``.
    """

    @differentiable
    def D_knudsen(self, T, specie):
        """
        Knudsen diffusion coefficient [m^2/s], see :meth:`nest.layers.PorousTransport.D_knudsen`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        specie : nest.properties.Specie
            Thermodynamic specie model.

        Returns
        -------
        float or numpy.ndarray
            Knudsen diffusion coefficient [m^2/s].
        """
        return as_output(super().D_knudsen(T, specie))

    @D_knudsen.grad("T")
    def _D_knudsen_grad_T(self, T, specie):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.PorousTransport.D_knudsen`

        Returns
        -------
        float or numpy.ndarray
            Diffusivity derivative [m^2/s.K] (D_K / (2 T)), same shape as T.
        """
        T = np.asarray(T, dtype=float)
        return as_output(np.asarray(super().D_knudsen(T, specie)) / (2 * T))

    @differentiable
    def dP_dl(self, mol_flux, T, P, gas, delta):
        """
        Partial pressures at the reaction site [Pa]; unchanged ``P`` in the base model, see :meth:`nest.layers.PorousTransport.dP_dl`.

        Parameters
        ----------
        mol_flux : numpy.ndarray
            Molar flux [mol/(m^2 s)].
        T : float
            Temperature [K].
        P : numpy.ndarray
            Partial pressures [Pa].
        gas : nest.properties.Mixture
            Mixture of species.
        delta : float
            Thickness of the porous layer [m].

        Returns
        -------
        numpy.ndarray
            Partial pressures [Pa].
        """
        return as_output(super().dP_dl(mol_flux, T, P, gas, delta))

    @dP_dl.grad("mol_flux")
    def _dP_dl_grad_mol_flux(self, mol_flux, T, P, gas, delta):
        r"""
        :math:`\nabla_{J}` :meth:`~nest.grad.layers.PorousTransport.dP_dl`

        Returns
        -------
        numpy.ndarray
            Zeros of shape ``P.shape + mol_flux.shape``.
        """
        return as_output(np.zeros(np.shape(P) + np.shape(mol_flux)))

    @dP_dl.grad("T")
    def _dP_dl_grad_T(self, mol_flux, T, P, gas, delta):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.PorousTransport.dP_dl`

        Returns
        -------
        numpy.ndarray
            Zeros of shape ``P.shape + T.shape``.
        """
        return as_output(np.zeros(np.shape(P) + np.shape(T)))

    @dP_dl.grad("P")
    def _dP_dl_grad_P(self, mol_flux, T, P, gas, delta):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.layers.PorousTransport.dP_dl`

        Returns
        -------
        numpy.ndarray
            Identity of shape ``P.shape + P.shape``.
        """
        size = np.size(P)
        return as_output(np.eye(size).reshape(np.shape(P) + np.shape(P)))


class BinaryFick(layers.BinaryFick, PorousTransport):
    """
    Binary Fick diffusion with analytical gradients.

    Same parameters as :class:`nest.layers.BinaryFick`. As in the base model, a
    pure gas (one species) broadcasts its flux and diffusivity over all entries
    of ``P``. ``D_eff`` gradients are full Jacobians (location-wise derivatives
    are spread on the diagonal for array ``T`` or ``P``).
    """

    def _D_eff_terms(self, T, P, gas):
        """
        Effective diffusivity and its T- and P-derivatives (element-wise).

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        P : float or numpy.ndarray
            Pressure [Pa].
        gas : nest.properties.Mixture
            Pure gas or binary mixture.

        Returns
        -------
        tuple of numpy.ndarray
            ``(D, dD_dT, dD_dP)``, each of shape ``(n_species,) + loc``
        """
        gas = _grad_mixture(gas)
        D = np.asarray(layers.BinaryFick.D_eff(self, T, P, gas), dtype=float)
        T = np.asarray(T, dtype=float)
        porosity = self.epsilon / self.tau
        D_K = np.array([np.asarray(self.D_knudsen(T, s)) for s in gas.species])
        D_K = D_K.reshape(D_K.shape + (1,) * (D.ndim - D_K.ndim))  # align with P locations
        dD_K = D_K / (2 * T.reshape(T.shape + (1,) * (D.ndim - 1 - T.ndim)))
        if len(gas.species) == 1:
            # D = porosity * D_K, independent of P
            return D, np.broadcast_to(porosity * dD_K, D.shape), np.zeros(D.shape)
        D_12 = np.asarray(gas.D_ij(0, 1, T, P))
        dD_12_dT = np.asarray(gas.D_ij.grad_T(0, 1, T, P))
        dD_12_dP = np.asarray(gas.D_ij.grad_P(0, 1, T, P))
        dD_dT = D**2 * (dD_K / (porosity * D_K**2) + dD_12_dT / (porosity * D_12**2))
        dD_dP = D**2 * dD_12_dP / (porosity * D_12**2)
        return D, np.broadcast_to(dD_dT, D.shape), np.broadcast_to(dD_dP, D.shape)

    @differentiable
    def D_eff(self, T, P, gas):
        """
        Effective diffusivity of each species [m^2/s], see :meth:`nest.layers.BinaryFick.D_eff`.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K].
        P : float or numpy.ndarray
            Pressure [Pa].
        gas : nest.properties.Mixture
            Pure gas or binary mixture.

        Returns
        -------
        numpy.ndarray
            Effective diffusivities [m^2/s], species on the first axis.
        """
        return as_output(super().D_eff(T, P, gas))

    @D_eff.grad("T")
    def _D_eff_grad_T(self, T, P, gas):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.BinaryFick.D_eff`

        Returns
        -------
        numpy.ndarray
            Diffusivity derivative [m^2/s.K], full Jacobian.
        """
        _, dD_dT, _ = self._D_eff_terms(T, P, gas)
        return _location_jacobian(dD_dT, T)

    @D_eff.grad("P")
    def _D_eff_grad_P(self, T, P, gas):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.layers.BinaryFick.D_eff`

        Returns
        -------
        numpy.ndarray
            Diffusivity derivative [m^2/s.Pa], full Jacobian; zeros for a pure gas.
        """
        D, _, dD_dP = self._D_eff_terms(T, P, gas)
        if len(gas.species) == 1:  # D has no pressure (location) axes
            return as_output(np.zeros(D.shape + np.shape(P)))
        return _location_jacobian(dD_dP, P)

    def _dP_dl_terms(self, mol_flux, T, P, gas, delta):
        """
        Flux, diffusivity and diffusivity derivatives broadcast to the entries of ``P``.

        Parameters
        ----------
        mol_flux : numpy.ndarray
            Molar flux [mol/(m^2 s)].
        T : float
            Temperature [K].
        P : numpy.ndarray
            Partial pressures [Pa], 1-D.
        gas : nest.properties.Mixture
            Pure gas or binary mixture.
        delta : float
            Thickness of the porous layer [m].

        Returns
        -------
        tuple
            ``(J, D, dD_dT, dD_dP, broadcast)``; the first four of shape
            ``P.shape``, ``broadcast`` the Jacobian of ``J`` w.r.t. ``mol_flux``
        """
        P = np.asarray(P, dtype=float)
        mol_flux = np.atleast_1d(np.asarray(mol_flux, dtype=float))
        D, dD_dT, dD_dP = self._D_eff_terms(T, sum(P), gas)
        index = np.broadcast_to(np.arange(mol_flux.size), P.shape)
        J = mol_flux[index]
        broadcast = np.eye(mol_flux.size)[index]
        D, dD_dT, dD_dP = (np.broadcast_to(a, P.shape) for a in (D, dD_dT, dD_dP))
        return J, D, dD_dT, dD_dP, broadcast

    @differentiable
    def dP_dl(self, mol_flux, T, P, gas, delta):
        """
        Partial pressures at the reaction site [Pa], see :meth:`nest.layers.BinaryFick.dP_dl`.

        Parameters
        ----------
        mol_flux : numpy.ndarray
            Molar flux [mol/(m^2 s)].
        T : float
            Temperature [K].
        P : numpy.ndarray
            Partial pressures [Pa].
        gas : nest.properties.Mixture
            Pure gas or binary mixture.
        delta : float
            Thickness of the porous layer [m].

        Returns
        -------
        numpy.ndarray
            Partial pressures at the reaction site [Pa].
        """
        return as_output(super().dP_dl(mol_flux, T, P, gas, delta))

    @dP_dl.grad("mol_flux")
    def _dP_dl_grad_mol_flux(self, mol_flux, T, P, gas, delta):
        r"""
        :math:`\nabla_{J}` :meth:`~nest.grad.layers.BinaryFick.dP_dl`

        Returns
        -------
        numpy.ndarray
            Derivative [Pa m^2 s/mol] of shape ``P.shape + mol_flux.shape``.
        """
        _, D, _, _, broadcast = self._dP_dl_terms(mol_flux, T, P, gas, delta)
        value = (R * T * delta / D)[:, None] * broadcast
        return as_output(value.reshape(np.shape(P) + np.shape(mol_flux)))

    @dP_dl.grad("T")
    def _dP_dl_grad_T(self, mol_flux, T, P, gas, delta):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.BinaryFick.dP_dl`

        Returns
        -------
        numpy.ndarray
            Derivative [Pa/K] of shape ``P.shape``.
        """
        J, D, dD_dT, _, _ = self._dP_dl_terms(mol_flux, T, P, gas, delta)
        return as_output(J * R * delta / D - J * R * T * delta * dD_dT / D**2)

    @dP_dl.grad("P")
    def _dP_dl_grad_P(self, mol_flux, T, P, gas, delta):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.layers.BinaryFick.dP_dl`

        Returns
        -------
        numpy.ndarray
            Jacobian [-] of shape ``P.shape + P.shape``.
        """
        J, D, _, dD_dP, _ = self._dP_dl_terms(mol_flux, T, P, gas, delta)
        n = np.size(P)
        column = J * R * T * delta * dD_dP / D**2  # through the total pressure sum(P)
        return as_output(np.eye(n) - np.outer(column, np.ones(n)))


class StefanMaxwell(layers.StefanMaxwell, PorousTransport):
    """
    Stefan-Maxwell transport; values as :class:`nest.layers.StefanMaxwell`.

    Gradients are registered but not implemented yet; they raise
    :class:`NotImplementedError`.
    """

    @differentiable
    def D_eff(self, T, P, gas):
        """
        Effective binary diffusivities [m^2/s], see :meth:`nest.layers.StefanMaxwell.D_eff`.

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        gas : nest.properties.Mixture
            Mixture of species.

        Returns
        -------
        numpy.ndarray
            Effective diffusivities [m^2/s].
        """
        return as_output(super().D_eff(T, P, gas))

    @D_eff.grad("T")
    def _D_eff_grad_T(self, T, P, gas):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.StefanMaxwell.D_eff`

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("StefanMaxwell.D_eff.grad_T")

    @D_eff.grad("P")
    def _D_eff_grad_P(self, T, P, gas):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.layers.StefanMaxwell.D_eff`

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("StefanMaxwell.D_eff.grad_P")

    @differentiable
    def dP_dl(self, mol_flux, T, P, gas, delta):
        """
        Partial pressures at the reaction site [Pa], see :meth:`nest.layers.StefanMaxwell.dP_dl`.

        Parameters
        ----------
        mol_flux : numpy.ndarray
            Molar flux [mol/(m^2 s)].
        T : float
            Temperature [K].
        P : numpy.ndarray
            Partial pressures [Pa].
        gas : nest.properties.Mixture
            Mixture of species.
        delta : float
            Thickness of the porous layer [m].

        Returns
        -------
        numpy.ndarray
            Partial pressures at the reaction site [Pa].
        """
        return as_output(super().dP_dl(mol_flux, T, P, gas, delta))

    @dP_dl.grad("mol_flux")
    def _dP_dl_grad_mol_flux(self, mol_flux, T, P, gas, delta):
        r"""
        :math:`\nabla_{J}` :meth:`~nest.grad.layers.StefanMaxwell.dP_dl`

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("StefanMaxwell.dP_dl.grad_mol_flux")

    @dP_dl.grad("T")
    def _dP_dl_grad_T(self, mol_flux, T, P, gas, delta):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.StefanMaxwell.dP_dl`

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("StefanMaxwell.dP_dl.grad_T")

    @dP_dl.grad("P")
    def _dP_dl_grad_P(self, mol_flux, T, P, gas, delta):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.layers.StefanMaxwell.dP_dl`

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("StefanMaxwell.dP_dl.grad_P")


def _require(value, cls, name: str):
    """
    Check that a layer component is a gradient-aware model.

    Parameters
    ----------
    value : object
        component to check
    cls : type
        required ``nest.grad`` class
    name : str
        component name, for the error message

    Raises
    ------
    TypeError
        if ``value`` is not an instance of ``cls``
    """
    if not isinstance(value, cls):
        raise TypeError(
            f"{name} must be a {cls.__module__}.{cls.__qualname__}, got "
            f"{type(value).__module__}.{type(value).__qualname__}; build it from nest.grad"
        )


class Layer(layers.Layer):
    """
    Cell layer with analytical gradients.

    Same parameters as :class:`nest.layers.Layer`. The components must be
    ``nest.grad`` models (:class:`Kinetic`, :class:`Conductivity`,
    :class:`PorousTransport`, a :mod:`nest.grad.degradation` model); base
    ``nest`` components raise a :class:`TypeError`, since their gradients are
    missing. ``None`` gives a fresh default component (the base class shares
    one default instance between layers).

    Methods take one location: scalar ``j`` and ``T``, 1-D ``Ps``. A
    ``pol_deg`` or ``ohm_deg`` of ``None`` means the initial material value
    ``degradation.m0``.
    """

    def __init__(
        self,
        delta=0.0,
        kinetic=None,
        conductivity=None,
        transport=None,
        degradation=None,
        elements=2,
        tpb_density=None,
        capacitance_volumetric=None,
    ):
        kinetic = Kinetic() if kinetic is None else kinetic
        conductivity = Conductivity() if conductivity is None else conductivity
        transport = PorousTransport() if transport is None else transport
        degradation = NoDegradation() if degradation is None else degradation
        _require(kinetic, Kinetic, "kinetic")
        _require(conductivity, Conductivity, "conductivity")
        _require(transport, PorousTransport, "transport")
        _require(degradation, DegradationGradients, "degradation")
        super().__init__(
            delta, kinetic, conductivity, transport, degradation, elements,
            tpb_density, capacitance_volumetric,
        )

    def _material(self, m):
        """
        Material variable, with ``None`` meaning the initial value.

        Parameters
        ----------
        m : float or None
            material variable

        Returns
        -------
        float
            ``m``, or ``degradation.m0`` if ``m`` is None
        """
        return self.degradation.m0 if m is None else m

    def _ohm_terms(self, T, ohm_deg):
        """
        Effective thickness and resistivity of the ohmic voltage.

        Parameters
        ----------
        T : float
            temperature [K]
        ohm_deg : float or None
            material variable for ohmic degradation

        Returns
        -------
        tuple of float
            ``(delta, rho, drho_dT)``: thickness including the ohmic degradation [m],
            resistivity [Ohm m] and its temperature derivative [Ohm m/K]
        """
        delta = self.delta
        if self.degradation.ohm_active:
            delta += self.degradation.ohm_deg(self._material(ohm_deg))
        resistivity = self.conductivity.resistivity
        return delta, resistivity(T), resistivity.grad_T(T)

    @differentiable
    def V_ohm(self, j, T, ohm_deg=None, **kwargs):
        """
        Ohmic voltage [V], see :meth:`nest.layers.Layer.V_ohm`.

        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            temperature [K]
        ohm_deg : float, optional
            material variable for ohmic degradation; None means ``degradation.m0``
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            Ohmic voltage [V]
        """
        return as_output(super().V_ohm(j, T, ohm_deg=self._material(ohm_deg)))

    @V_ohm.grad("j")
    def _V_ohm_grad_j(self, j, T, ohm_deg=None, **kwargs):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.layers.Layer.V_ohm`

        Returns
        -------
        float
            area-specific resistance [Ohm m^2]
        """
        delta, rho, _ = self._ohm_terms(T, ohm_deg)
        return as_output(np.zeros(np.shape(j)) + delta * rho)

    @V_ohm.grad("T")
    def _V_ohm_grad_T(self, j, T, ohm_deg=None, **kwargs):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.Layer.V_ohm`

        Returns
        -------
        float
            voltage derivative [V/K]
        """
        delta, _, drho_dT = self._ohm_terms(T, ohm_deg)
        return as_output(np.asarray(j, dtype=float) * delta * drho_dT)

    @V_ohm.grad("ohm_deg")
    def _V_ohm_grad_ohm_deg(self, j, T, ohm_deg=None, **kwargs):
        r"""
        :math:`\nabla_{\omega}` :meth:`~nest.grad.layers.Layer.V_ohm`

        Returns
        -------
        float
            voltage derivative per unit material variable; 0 without active ohmic degradation
        """
        if not self.degradation.ohm_active:
            return as_output(np.zeros(np.shape(j)))
        dh = self.degradation.ohm_deg.grad_m(self._material(ohm_deg))
        return as_output(np.asarray(j, dtype=float) * dh * self.conductivity.resistivity(T))

    def _act_partials(self, j, T, Ps, pol_deg):
        """
        Partial derivatives of the activation voltage.

        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            temperature [K]
        Ps : numpy.ndarray
            partial pressures [Pa], 1-D
        pol_deg : float or None
            material variable for polarization degradation

        Returns
        -------
        dict
            ``{"j", "T", "Ps", "pol_deg"}`` derivatives; ``"Ps"`` of shape ``Ps.shape``.
            All zero unless the kinetic model is a :class:`ButlerVolmer`.
        """
        if getattr(self.kinetic, "exchange_current_prefactor", None) is not None:
            raise ValueError("V_act requires area-based kinetics")
        Ps = np.atleast_1d(np.asarray(Ps, dtype=float))
        if not isinstance(self.kinetic, ButlerVolmer):  # the base kinetic gives V = 0
            return {"j": 0.0, "T": 0.0, "Ps": np.zeros(Ps.shape), "pol_deg": 0.0}
        m = self._material(pol_deg)
        j_0 = self.kinetic.j_0(T, Ps)
        ratio = self.degradation.pol_deg(m) if self.degradation.pol_active else 1.0
        eta = self.kinetic.V(j, T, j_0 * ratio)
        slope = self.kinetic.dj_dV(eta, T, j_0 * ratio)
        deta_dj0 = -(j / (j_0 * ratio)) / slope  # w.r.t. the degraded exchange current
        partials = {
            "j": 1 / slope,
            "T": eta / T + deta_dj0 * ratio * self.kinetic.j_0.grad_T(T, Ps),
            "Ps": deta_dj0 * ratio * np.asarray(self.kinetic.j_0.grad_Ps(T, Ps)),
            "pol_deg": 0.0,
        }
        if self.degradation.pol_active:
            partials["pol_deg"] = deta_dj0 * j_0 * self.degradation.pol_deg.grad_m(m)
        return partials

    @differentiable
    def V_act(self, j, T, Ps=np.array([0]), pol_deg=None, **kwargs):
        """
        Activation voltage [V], see :meth:`nest.layers.Layer.V_act`.

        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            temperature [K]
        Ps : numpy.ndarray
            partial pressures [Pa]
        pol_deg : float, optional
            material variable for polarization degradation; None means ``degradation.m0``
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            Activation voltage [V]
        """
        return as_output(super().V_act(j, T, Ps, pol_deg=self._material(pol_deg)))

    @V_act.grad("j")
    def _V_act_grad_j(self, j, T, Ps=np.array([0]), pol_deg=None, **kwargs):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.layers.Layer.V_act`

        Returns
        -------
        float
            voltage derivative [V m^2/A]
        """
        return as_output(self._act_partials(j, T, Ps, pol_deg)["j"])

    @V_act.grad("T")
    def _V_act_grad_T(self, j, T, Ps=np.array([0]), pol_deg=None, **kwargs):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.Layer.V_act`

        Returns
        -------
        float
            voltage derivative [V/K]
        """
        return as_output(self._act_partials(j, T, Ps, pol_deg)["T"])

    @V_act.grad("Ps")
    def _V_act_grad_Ps(self, j, T, Ps=np.array([0]), pol_deg=None, **kwargs):
        r"""
        :math:`\nabla_{P_s}` :meth:`~nest.grad.layers.Layer.V_act`

        Returns
        -------
        numpy.ndarray
            voltage derivative [V/Pa], shape ``Ps.shape``
        """
        return as_output(self._act_partials(j, T, Ps, pol_deg)["Ps"])

    @V_act.grad("pol_deg")
    def _V_act_grad_pol_deg(self, j, T, Ps=np.array([0]), pol_deg=None, **kwargs):
        r"""
        :math:`\nabla_{\pi}` :meth:`~nest.grad.layers.Layer.V_act`

        Returns
        -------
        float
            voltage derivative per unit material variable; 0 without active
            polarization degradation
        """
        return as_output(self._act_partials(j, T, Ps, pol_deg)["pol_deg"])

    @differentiable
    def Ps_star(self, j, T, Ps):
        """
        Partial pressures at the reaction site [Pa], see :meth:`nest.layers.Layer.Ps_star`.

        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            temperature [K]
        Ps : numpy.ndarray
            partial pressures [Pa], 1-D

        Returns
        -------
        numpy.ndarray
            Partial pressures at the reaction site [Pa]
        """
        return as_output(super().Ps_star(j, T, Ps))

    def _transport_args(self, j, T, Ps):
        """
        Arguments of ``transport.dP_dl`` used by :meth:`Ps_star`.

        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            temperature [K]
        Ps : numpy.ndarray
            partial pressures [Pa]

        Returns
        -------
        tuple
            ``(mol_flux, T, Ps, gas, delta)``
        """
        return self.kinetic.mol_flux(j), T, Ps, self.kinetic.gas, self.delta

    @Ps_star.grad("j")
    def _Ps_star_grad_j(self, j, T, Ps):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.layers.Layer.Ps_star`

        Returns
        -------
        numpy.ndarray
            pressure derivative [Pa m^2/A], shape ``Ps.shape``
        """
        dP_dJ = np.asarray(self.transport.dP_dl.grad_mol_flux(*self._transport_args(j, T, Ps)))
        dJ_dj = np.atleast_1d(self.kinetic.mol_flux.grad_j(j))
        return as_output(dP_dJ.reshape(np.shape(Ps) + dJ_dj.shape) @ dJ_dj)

    @Ps_star.grad("T")
    def _Ps_star_grad_T(self, j, T, Ps):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.Layer.Ps_star`

        Returns
        -------
        numpy.ndarray
            pressure derivative [Pa/K], shape ``Ps.shape``
        """
        return as_output(self.transport.dP_dl.grad_T(*self._transport_args(j, T, Ps)))

    @Ps_star.grad("Ps")
    def _Ps_star_grad_Ps(self, j, T, Ps):
        r"""
        :math:`\nabla_{P_s}` :meth:`~nest.grad.layers.Layer.Ps_star`

        Returns
        -------
        numpy.ndarray
            Jacobian [-] of shape ``Ps.shape + Ps.shape``
        """
        return as_output(self.transport.dP_dl.grad_P(*self._transport_args(j, T, Ps)))

    def _V_partials(self, j, T, Ps=np.array([0]), pol_deg=None, ohm_deg=None):
        """
        Partial derivatives of the total layer voltage.

        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            temperature [K]
        Ps : numpy.ndarray
            partial pressures [Pa], 1-D
        pol_deg : float or None
            material variable for polarization degradation
        ohm_deg : float or None
            material variable for ohmic degradation

        Returns
        -------
        dict
            ``{"j", "T", "Ps", "pol_deg", "ohm_deg"}`` derivatives
        """
        partials = self._act_partials(j, T, Ps, pol_deg)
        delta, rho, drho_dT = self._ohm_terms(T, ohm_deg)
        partials["j"] = partials["j"] + delta * rho
        partials["T"] = partials["T"] + j * delta * drho_dT
        partials["ohm_deg"] = self.V_ohm.grad_ohm_deg(j, T, ohm_deg)
        return partials

    @differentiable
    def V(self, j, T, Ps=np.array([0]), pol_deg=None, ohm_deg=None, **kwargs):
        """
        Total layer voltage [V], see :meth:`nest.layers.Layer.V`.

        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            temperature [K]
        Ps : numpy.ndarray
            partial pressures [Pa]
        pol_deg : float, optional
            material variable for polarization degradation; None means ``degradation.m0``
        ohm_deg : float, optional
            material variable for ohmic degradation; None means ``degradation.m0``
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            Total layer voltage (ohmic + activation) [V]
        """
        return as_output(
            self.V_ohm(j, T, ohm_deg=ohm_deg) + self.V_act(j, T, Ps, pol_deg=pol_deg)
        )

    @V.grad("j")
    def _V_grad_j(self, j, T, Ps=np.array([0]), pol_deg=None, ohm_deg=None, **kwargs):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.layers.Layer.V`

        Returns
        -------
        float
            voltage derivative [V m^2/A]
        """
        return as_output(self._V_partials(j, T, Ps, pol_deg, ohm_deg)["j"])

    @V.grad("T")
    def _V_grad_T(self, j, T, Ps=np.array([0]), pol_deg=None, ohm_deg=None, **kwargs):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.layers.Layer.V`

        Returns
        -------
        float
            voltage derivative [V/K]
        """
        return as_output(self._V_partials(j, T, Ps, pol_deg, ohm_deg)["T"])

    @V.grad("Ps")
    def _V_grad_Ps(self, j, T, Ps=np.array([0]), pol_deg=None, ohm_deg=None, **kwargs):
        r"""
        :math:`\nabla_{P_s}` :meth:`~nest.grad.layers.Layer.V`

        Returns
        -------
        numpy.ndarray
            voltage derivative [V/Pa], shape ``Ps.shape``
        """
        return as_output(self._act_partials(j, T, Ps, pol_deg)["Ps"])

    @V.grad("pol_deg")
    def _V_grad_pol_deg(self, j, T, Ps=np.array([0]), pol_deg=None, ohm_deg=None, **kwargs):
        r"""
        :math:`\nabla_{\pi}` :meth:`~nest.grad.layers.Layer.V`

        Returns
        -------
        float
            voltage derivative per unit polarization material variable
        """
        return as_output(self._act_partials(j, T, Ps, pol_deg)["pol_deg"])

    @V.grad("ohm_deg")
    def _V_grad_ohm_deg(self, j, T, Ps=np.array([0]), pol_deg=None, ohm_deg=None, **kwargs):
        r"""
        :math:`\nabla_{\omega}` :meth:`~nest.grad.layers.Layer.V`

        Returns
        -------
        float
            voltage derivative per unit ohmic material variable
        """
        return self.V_ohm.grad_ohm_deg(j, T, ohm_deg)
