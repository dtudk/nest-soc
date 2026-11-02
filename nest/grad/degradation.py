"""
Degradation models with analytical gradients.
"""

import warnings

import numpy as np

from nest import degradation
from nest.boundary import BoundaryData
from nest.constants import F, R
from nest.grad._utils import as_output, central_difference, normalize_jacobian
from nest.grad.differentiable import differentiable

PARTIALS = ("m", "T", "P", "j", "V", "n_fuel", "eta_fuel")
"""Variables returned by :meth:`DegradationGradients.material_dt_partials`."""


class DegradationGradients:
    """
    Shared gradient helpers of the ``nest.grad`` degradation models.

    Subclasses register the gradients of ``material_dt`` w.r.t. the variables
    their rate depends on; :meth:`material_dt_partials` fills in the rest with
    zeros so the cell model can treat all models alike.
    """

    m_scale = 1.0
    """Typical size of the material variable, used for ODE tolerances."""

    def material_dt_partials(
        self, m: float, state: BoundaryData, time: float, eta_fuel=None
    ) -> dict:
        """
        All partial derivatives of :meth:`material_dt`.

        Parameters
        ----------
        m : float
            material variable
        state : BoundaryData
            state variables
        time : float
            time [h]
        eta_fuel : float, optional
            fuel electrode overpotential [V]

        Returns
        -------
        dict
            ``{name: derivative}`` for every name in :data:`PARTIALS`; floats,
            except ``"n_fuel"`` with shape ``(len(state.n_fuel),)``. Variables
            the rate does not depend on give zeros.
        """
        gradients = type(self).material_dt.gradients
        partials = {}
        for name in PARTIALS:
            if name in gradients:
                partials[name] = gradients[name](self, m, state, time, eta_fuel=eta_fuel)
            elif name == "n_fuel":
                partials[name] = np.zeros(np.size(state.n_fuel))
            else:
                partials[name] = 0.0
        return partials


class NoDegradation(DegradationGradients, degradation.NoDegradation):
    """
    No degradation, with (zero) gradients; see :class:`nest.degradation.NoDegradation`.
    """

    m_scale = 1.0

    @differentiable
    def material_dt(self, m, state, time, eta_fuel=None, **kwargs):
        """
        Material degradation rate, always 0, see :meth:`nest.degradation.NoDegradation.material_dt`.

        Parameters
        ----------
        m : float
            material variable
        state : BoundaryData
            state variables
        time : float
            time [h]
        eta_fuel : float, optional
            not used
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            material degradation rate; always 0
        """
        return as_output(super().material_dt(m, state, time, eta_fuel=eta_fuel, **kwargs))

    @material_dt.grad("m")
    def _material_dt_grad_m(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.NoDegradation.material_dt`

        Returns
        -------
        float
            always 0
        """
        return 0.0

    @differentiable
    def pol_deg(self, m, **kwargs):
        """
        Polarization degradation ratio, always 1, see :meth:`nest.degradation.NoDegradation.pol_deg`.

        Parameters
        ----------
        m : float
            material variable
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            always 1
        """
        return as_output(super().pol_deg(m))

    @pol_deg.grad("m")
    def _pol_deg_grad_m(self, m, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.NoDegradation.pol_deg`

        Returns
        -------
        float
            always 0
        """
        return 0.0

    @differentiable
    def ohm_deg(self, m, **kwargs):
        """
        Ohmic degradation quantity, always 0, see :meth:`nest.degradation.NoDegradation.ohm_deg`.

        Parameters
        ----------
        m : float
            material variable
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            always 0
        """
        return as_output(super().ohm_deg(m))

    @ohm_deg.grad("m")
    def _ohm_deg_grad_m(self, m, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.NoDegradation.ohm_deg`

        Returns
        -------
        float
            always 0
        """
        return 0.0


class NickelAgglomeration(DegradationGradients, degradation.NickelAgglomeration):
    """
    Nickel agglomeration with analytical gradients.

    Same parameters as :class:`nest.degradation.NickelAgglomeration`. Radii
    ``r_i``, ``r_j``, ``r_ed`` are in m, the material variable ``m`` in μm.
    Above ``r_ed_max`` the rate is 0 and so are its gradients.
    """

    m_scale = 0.25

    @differentiable
    def z_ii(self, psi_i, psi_j, r_i, r_j):
        """
        Coordination number between i,i [-], see :meth:`nest.degradation.NickelAgglomeration.z_ii`.

        Parameters
        ----------
        psi_i : float
            solid volume fraction of i [-]
        psi_j : float
            solid volume fraction of j [-]
        r_i : float
            mean particle radius of i [m]
        r_j : float
            mean particle radius of j [m]

        Returns
        -------
        float
            coordination number between i,i [-]
        """
        return as_output(super().z_ii(psi_i, psi_j, r_i, r_j))

    @z_ii.grad("r_i")
    def _z_ii_grad_r_i(self, psi_i, psi_j, r_i, r_j):
        r"""
        :math:`\nabla_{r_i}` :meth:`~nest.grad.degradation.NickelAgglomeration.z_ii`

        Returns
        -------
        float
            derivative [1/m]
        """
        a, b = psi_i / r_i, psi_j / r_j
        return as_output(-self.z_bar * a * b / (r_i * (a + b) ** 2))

    @z_ii.grad("r_j")
    def _z_ii_grad_r_j(self, psi_i, psi_j, r_i, r_j):
        r"""
        :math:`\nabla_{r_j}` :meth:`~nest.grad.degradation.NickelAgglomeration.z_ii`

        Returns
        -------
        float
            derivative [1/m]
        """
        a, b = psi_i / r_i, psi_j / r_j
        return as_output(self.z_bar * a * b / (r_j * (a + b) ** 2))

    @differentiable
    def z_ij(self, psi_i, psi_j, r_i, r_j):
        """
        Coordination number between i,j [-], see :meth:`nest.degradation.NickelAgglomeration.z_ij`.

        Parameters
        ----------
        psi_i : float
            solid volume fraction of i [-]
        psi_j : float
            solid volume fraction of j [-]
        r_i : float
            mean particle radius of i [m]
        r_j : float
            mean particle radius of j [m]

        Returns
        -------
        float
            coordination number between i,j [-]
        """
        return as_output(super().z_ij(psi_i, psi_j, r_i, r_j))

    @z_ij.grad("r_i")
    def _z_ij_grad_r_i(self, psi_i, psi_j, r_i, r_j):
        r"""
        :math:`\nabla_{r_i}` :meth:`~nest.grad.degradation.NickelAgglomeration.z_ij`

        Returns
        -------
        float
            derivative [1/m]
        """
        # z_ij = z_bar * h * g with h = (1 + r_i^2/r_j^2)/2 and g = b/(a + b)
        a, b = psi_i / r_i, psi_j / r_j
        h, g = 0.5 * (1 + r_i**2 / r_j**2), b / (a + b)
        dh, dg = r_i / r_j**2, a * b / (r_i * (a + b) ** 2)
        return as_output(self.z_bar * (dh * g + h * dg))

    @z_ij.grad("r_j")
    def _z_ij_grad_r_j(self, psi_i, psi_j, r_i, r_j):
        r"""
        :math:`\nabla_{r_j}` :meth:`~nest.grad.degradation.NickelAgglomeration.z_ij`

        Returns
        -------
        float
            derivative [1/m]
        """
        a, b = psi_i / r_i, psi_j / r_j
        h, g = 0.5 * (1 + r_i**2 / r_j**2), b / (a + b)
        dh, dg = -(r_i**2) / r_j**3, -a * b / (r_j * (a + b) ** 2)
        return as_output(self.z_bar * (dh * g + h * dg))

    def _p_i_grad_z(self, z):
        """
        Derivative of the percolation probability w.r.t. the coordination number.

        Parameters
        ----------
        z : float
            coordination number between i,i [-]

        Returns
        -------
        float
            derivative; 0 outside the middle branch of
            :meth:`nest.degradation.NickelAgglomeration.p_i`
        """
        if z > 4.236 or z <= 1.764:
            return 0.0
        return 3.7 / 2.472 * ((4.236 - z) / 2.472) ** 2.7

    @differentiable
    def p_i(self, psi_i, psi_j, r_i, r_j):
        """
        Percolation probability of i-particle [-], see :meth:`nest.degradation.NickelAgglomeration.p_i`.

        Parameters
        ----------
        psi_i : float
            solid volume fraction of i [-]
        psi_j : float
            solid volume fraction of j [-]
        r_i : float
            mean particle radius of i [m]
        r_j : float
            mean particle radius of j [m]

        Returns
        -------
        float
            percolation probability of i-particle [-]
        """
        return as_output(super().p_i(psi_i, psi_j, r_i, r_j))

    @p_i.grad("r_i")
    def _p_i_grad_r_i(self, psi_i, psi_j, r_i, r_j):
        r"""
        :math:`\nabla_{r_i}` :meth:`~nest.grad.degradation.NickelAgglomeration.p_i`

        Returns
        -------
        float
            derivative [1/m]; 0 outside the middle (power-law) branch
        """
        z = self.z_ii(psi_i, psi_j, r_i, r_j)
        return as_output(self._p_i_grad_z(z) * self.z_ii.grad_r_i(psi_i, psi_j, r_i, r_j))

    @p_i.grad("r_j")
    def _p_i_grad_r_j(self, psi_i, psi_j, r_i, r_j):
        r"""
        :math:`\nabla_{r_j}` :meth:`~nest.grad.degradation.NickelAgglomeration.p_i`

        Returns
        -------
        float
            derivative [1/m]; 0 outside the middle (power-law) branch
        """
        z = self.z_ii(psi_i, psi_j, r_i, r_j)
        return as_output(self._p_i_grad_z(z) * self.z_ii.grad_r_j(psi_i, psi_j, r_i, r_j))

    @differentiable
    def l_tpb(self, r_ed):
        """
        Triple phase boundary length [1/m^2], see :meth:`nest.degradation.NickelAgglomeration.l_tpb`.

        Parameters
        ----------
        r_ed : float
            mean electrode particle radius [m]

        Returns
        -------
        float
            triple phase boundary length [1/m^2]
        """
        return as_output(super().l_tpb(r_ed))

    @l_tpb.grad("r_ed")
    def _l_tpb_grad_r_ed(self, r_ed):
        r"""
        :math:`\nabla_{r_{ed}}` :meth:`~nest.grad.degradation.NickelAgglomeration.l_tpb`

        Returns
        -------
        float
            derivative [1/m^3]
        """
        # l_tpb = c * z_ed_el(r_ed) / r_ed^2 with c independent of r_ed
        c = 1.5 * np.sin(self.theta) * (1 - self.epsilon) * self.psi_ed
        z = self.z_ij(self.psi_ed, self.psi_el, r_ed, self.r0_el)
        dz = self.z_ij.grad_r_i(self.psi_ed, self.psi_el, r_ed, self.r0_el)
        return as_output(c * (dz / r_ed**2 - 2 * z / r_ed**3))

    @differentiable
    def l_tpb_eff(self, r_ed):
        """
        Effective triple phase boundary length [1/m^2], see :meth:`nest.degradation.NickelAgglomeration.l_tpb_eff`.

        Parameters
        ----------
        r_ed : float
            mean electrode particle radius [m]

        Returns
        -------
        float
            effective triple phase boundary length [1/m^2]
        """
        return as_output(super().l_tpb_eff(r_ed))

    @l_tpb_eff.grad("r_ed")
    def _l_tpb_eff_grad_r_ed(self, r_ed):
        r"""
        :math:`\nabla_{r_{ed}}` :meth:`~nest.grad.degradation.NickelAgglomeration.l_tpb_eff`

        Returns
        -------
        float
            derivative [1/m^3]; the percolation part is constant while
            ``p_ed * p_el`` is clipped at 0.01
        """
        p_ed = self.p_i(self.psi_ed, self.psi_el, r_ed, self.r0_el)
        p_el = self.p_i(self.psi_el, self.psi_ed, self.r0_el, r_ed)
        value = self.l_tpb.grad_r_ed(r_ed) * max(p_ed * p_el, 0.01)
        if p_ed * p_el > 0.01:
            dp_ed = self.p_i.grad_r_i(self.psi_ed, self.psi_el, r_ed, self.r0_el)
            dp_el = self.p_i.grad_r_j(self.psi_el, self.psi_ed, self.r0_el, r_ed)
            value += self.l_tpb(r_ed) * (dp_ed * p_el + p_ed * dp_el)
        return as_output(value)

    @differentiable
    def material_dt(self, m, state, time, eta_fuel=None, **kwargs):
        """
        Ni radius rate of change [μm/h], see :meth:`nest.degradation.NickelAgglomeration.material_dt`.

        Parameters
        ----------
        m : float
            mean Ni radius [μm]
        state : BoundaryData
            state variables; ``state.n_fuel`` gives the TPB mole fractions
            ``[H2, H2O, ...]``
        time : float
            time [h]
        eta_fuel : float, optional
            not used
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            Ni radius rate of change [μm/h]
        """
        return as_output(super().material_dt(m, state, time, eta_fuel=eta_fuel, **kwargs))

    @material_dt.grad("m")
    def _material_dt_grad_m(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.NickelAgglomeration.material_dt`

        Returns
        -------
        float
            derivative [1/h]; 0 above ``r_ed_max``
        """
        rate = self.material_dt(m, state, time)
        if rate == 0:
            return 0.0
        # Only the TPB area A_el depends on the radius via s_el(Z_el_ed);
        # n_tot * xi_el is independent of it.
        rho = m / 1e6
        Z_el_el = self.z_ii(self.psi_el, self.psi_el, self.r0_el, self.r0_el)
        Z_el_ed = self.z_ij(self.psi_el, self.psi_ed, self.r0_el, rho)
        dZ_el_ed = self.z_ij.grad_r_j(self.psi_el, self.psi_ed, self.r0_el, rho)
        s_el = 2 - (1 - np.cos(self.theta)) * (Z_el_el + Z_el_ed)
        ds_el = -(1 - np.cos(self.theta)) * dZ_el_ed
        return as_output(rate * (-7 / m - 1e-6 * ds_el / s_el))

    @material_dt.grad("T")
    def _material_dt_grad_T(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.degradation.NickelAgglomeration.material_dt`

        Returns
        -------
        float
            derivative [μm/(h K)]
        """
        rate = self.material_dt(m, state, time)
        return as_output(rate * self.E_act / (R * state.T**2))

    @material_dt.grad("P")
    def _material_dt_grad_P(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.degradation.NickelAgglomeration.material_dt`

        Returns
        -------
        float
            derivative [μm/(h Pa)]
        """
        rate = self.material_dt(m, state, time)
        return as_output(rate / (2 * state.P))

    @material_dt.grad("n_fuel")
    def _material_dt_grad_n_fuel(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_{n_{fuel}}` :meth:`~nest.grad.degradation.NickelAgglomeration.material_dt`

        Returns
        -------
        numpy.ndarray
            derivative w.r.t. each entry of ``state.n_fuel``, shape ``(len(n_fuel),)``
        """
        n = np.asarray(state.n_fuel, dtype=float)
        rate = self.material_dt(m, state, time)
        if rate == 0:
            return np.zeros(n.size)
        x = n / n.sum()
        dlog_x = np.zeros(n.size)
        dlog_x[0], dlog_x[1] = -0.5 / x[0], 1 / x[1]
        return rate * dlog_x @ normalize_jacobian(n)

    @differentiable
    def pol_deg(self, m, **kwargs):
        """
        Effective TPB length [1/m^2], see :meth:`nest.degradation.NickelAgglomeration.pol_deg`.

        Parameters
        ----------
        m : float
            Nickel radius [μm]
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            effective triple phase boundary length [1/m^2]
        """
        return as_output(super().pol_deg(m))

    @pol_deg.grad("m")
    def _pol_deg_grad_m(self, m, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.NickelAgglomeration.pol_deg`

        Returns
        -------
        float
            derivative [1/(m^2 μm)]
        """
        return as_output(1e-6 * self.l_tpb_eff.grad_r_ed(m / 1e6))

    @differentiable
    def ohm_deg(self, m, **kwargs):
        """
        Ohmic degradation quantity, always 0, see :meth:`nest.degradation.NickelAgglomeration.ohm_deg`.

        Parameters
        ----------
        m : float
            Nickel radius [μm]
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            always 0
        """
        return as_output(super().ohm_deg(m))

    @ohm_deg.grad("m")
    def _ohm_deg_grad_m(self, m, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.NickelAgglomeration.ohm_deg`

        Returns
        -------
        float
            always 0
        """
        return 0.0


class NickelMigration(DegradationGradients, degradation.NickelMigration):
    """
    Ni migration with analytical gradients.

    Same parameters as :class:`nest.degradation.NickelMigration`, plus the
    rate gradients. Without them, central differences are used and a warning
    is emitted once per instance. Switching at the threshold is not differentiated.

    Parameters
    ----------
    rate : callable
        rate(T, j, eta) -> growth rate of the Ni-depleted zone thickness [m/h]
    threshold : callable, optional
        threshold(T, j, eta) -> True when the threshold is reached
    rate_after_threshold : callable, optional
        same as `rate`; used once the threshold is reached
    ohm_active : bool, optional
        activates the ohmic degradation modeling in simulation
    rate_grad : callable, optional
        rate_grad(T, j, eta) -> (d rate/dT, d rate/dj, d rate/deta)
    rate_after_threshold_grad : callable, optional
        same as `rate_grad`, for `rate_after_threshold`
    """

    m_scale = 1e-6

    def __init__(
        self,
        rate,
        threshold=None,
        rate_after_threshold=None,
        ohm_active=True,
        rate_grad=None,
        rate_after_threshold_grad=None,
    ):
        super().__init__(rate, threshold, rate_after_threshold, ohm_active)
        self.rate_grad = rate_grad
        self.rate_after_threshold_grad = rate_after_threshold_grad
        self._warned = False

    def rate_partials(self, T: float, j: float, eta: float) -> tuple:
        """
        Derivatives of the active rate w.r.t. temperature, current density and overpotential.

        Parameters
        ----------
        T : float
            temperature [K]
        j : float
            current density [A/m^2]
        eta : float
            fuel electrode overpotential [V]

        Returns
        -------
        tuple of float
            ``(d/dT, d/dj, d/deta)`` of the rate used by :meth:`material_dt`
        """
        if self.threshold_reached(T, j, eta):
            rate, rate_grad = self.rate_after_threshold, self.rate_after_threshold_grad
        else:
            rate, rate_grad = self.rate, self.rate_grad
        if rate_grad is not None:
            return tuple(float(g) for g in rate_grad(T, j, eta))
        if not self._warned:
            warnings.warn(
                "NickelMigration: no rate gradient given; using central differences",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned = True
        steps = np.array([1e-6 * T, 1e-6 * max(abs(j), 1.0), 1e-7])
        grad = central_difference(lambda x: rate(*x), np.array([T, j, eta], dtype=float), steps, warn=False)
        return tuple(float(g) for g in grad)

    @differentiable
    def material_dt(self, m, state, time, eta_fuel=None, **kwargs):
        """
        Ni-depleted zone growth rate [m/h], see :meth:`nest.degradation.NickelMigration.material_dt`.

        Parameters
        ----------
        m : float
            Ni-depleted zone thickness [m]
        state : BoundaryData
            state variables
        time : float
            time [h]
        eta_fuel : float
            fuel electrode overpotential [V]
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            Ni-depleted zone growth rate [m/h]
        """
        return as_output(super().material_dt(m, state, time, eta_fuel=eta_fuel, **kwargs))

    @material_dt.grad("m")
    def _material_dt_grad_m(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.NickelMigration.material_dt`

        Returns
        -------
        float
            always 0
        """
        return 0.0

    @material_dt.grad("T")
    def _material_dt_grad_T(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.degradation.NickelMigration.material_dt`

        Returns
        -------
        float
            derivative [m/(h K)]
        """
        return self.rate_partials(state.T, state.j, eta_fuel)[0]

    @material_dt.grad("j")
    def _material_dt_grad_j(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.degradation.NickelMigration.material_dt`

        Returns
        -------
        float
            derivative [m^3/(h A)]
        """
        return self.rate_partials(state.T, state.j, eta_fuel)[1]

    @material_dt.grad("eta_fuel")
    def _material_dt_grad_eta_fuel(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_{\eta_{fuel}}` :meth:`~nest.grad.degradation.NickelMigration.material_dt`

        Returns
        -------
        float
            derivative [m/(h V)]
        """
        return self.rate_partials(state.T, state.j, eta_fuel)[2]

    @differentiable
    def pol_deg(self, m, **kwargs):
        """
        Polarization degradation ratio, always 0, see :meth:`nest.degradation.NickelMigration.pol_deg`.

        Parameters
        ----------
        m : float
            Ni-depleted zone thickness [m]
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            always 0
        """
        return as_output(super().pol_deg(m))

    @pol_deg.grad("m")
    def _pol_deg_grad_m(self, m, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.NickelMigration.pol_deg`

        Returns
        -------
        float
            always 0
        """
        return 0.0

    @differentiable
    def ohm_deg(self, m, **kwargs):
        """
        Ni-depleted zone thickness [m], see :meth:`nest.degradation.NickelMigration.ohm_deg`.

        Parameters
        ----------
        m : float
            Ni-depleted zone thickness [m]
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            Ni-depleted zone thickness [m]
        """
        return as_output(super().ohm_deg(m))

    @ohm_deg.grad("m")
    def _ohm_deg_grad_m(self, m, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.NickelMigration.ohm_deg`

        Returns
        -------
        float
            always 1
        """
        return 1.0


class InterconnectOxidation(DegradationGradients, degradation.InterconnectOxidation):
    """
    Interconnect oxidation with analytical gradients.

    Same parameters as :class:`nest.degradation.InterconnectOxidation`. The
    ohmic degradation ``sqrt(m)`` has an infinite gradient at ``m = 0``.
    """

    m_scale = 1e-12

    @differentiable
    def material_dt(self, m, state, time, eta_fuel=None, **kwargs):
        """
        Oxide thickness squared rate [m^2/h], see :meth:`nest.degradation.InterconnectOxidation.material_dt`.

        Parameters
        ----------
        m : float
            oxide layer thickness squared [m^2]
        state : BoundaryData
            state variables
        time : float
            time [h]
        eta_fuel : float, optional
            not used
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            rate of change of oxide layer thickness squared [m^2/h]
        """
        return as_output(super().material_dt(m, state, time, eta_fuel=eta_fuel, **kwargs))

    @material_dt.grad("m")
    def _material_dt_grad_m(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.InterconnectOxidation.material_dt`

        Returns
        -------
        float
            always 0
        """
        return 0.0

    @material_dt.grad("T")
    def _material_dt_grad_T(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.degradation.InterconnectOxidation.material_dt`

        Returns
        -------
        float
            derivative [m^2/(h K)]
        """
        rate = self.material_dt(m, state, time)
        return as_output(rate * self.E_ox / (R * state.T**2))

    @differentiable
    def pol_deg(self, m, **kwargs):
        """
        Polarization degradation ratio, always 0, see :meth:`nest.degradation.InterconnectOxidation.pol_deg`.

        Parameters
        ----------
        m : float
            oxide layer thickness squared [m^2]
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            always 0
        """
        return as_output(super().pol_deg(m))

    @pol_deg.grad("m")
    def _pol_deg_grad_m(self, m, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.InterconnectOxidation.pol_deg`

        Returns
        -------
        float
            always 0
        """
        return 0.0

    @differentiable
    def ohm_deg(self, m, **kwargs):
        """
        Oxide layer thickness [m], see :meth:`nest.degradation.InterconnectOxidation.ohm_deg`.

        Parameters
        ----------
        m : float
            oxide layer thickness squared [m^2]
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            oxide layer thickness [m]
        """
        return as_output(super().ohm_deg(m))

    @ohm_deg.grad("m")
    def _ohm_deg_grad_m(self, m, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.InterconnectOxidation.ohm_deg`

        Returns
        -------
        float
            derivative [1/m]; infinite at ``m = 0``
        """
        with np.errstate(divide="ignore"):
            return as_output(0.5 / np.sqrt(np.asarray(m, dtype=float)))


class ChromiumPoison(DegradationGradients, degradation.ChromiumPoison):
    """
    Chromium poisoning with analytical gradients.

    Same parameters as :class:`nest.degradation.ChromiumPoison`.
    """

    m_scale = 1.0

    # Constants of nest.degradation.ChromiumPoison.material_dt
    _H_TPB = 35e-9  # m
    _M_CR2O3 = 151.99  # g/mol
    _RHO_CR2O3 = 5.22 / (1e-2) ** 3  # g/m^3
    _E_CRO2 = 6.7e4  # J/mol

    def _rate_terms(self, state: BoundaryData) -> tuple:
        """
        Pieces of the poisoning rate ``-K * m * A * 2 sinh(phi)``.

        Parameters
        ----------
        state : BoundaryData
            layer state conditions (voltage = activation voltage)

        Returns
        -------
        tuple of float
            ``(K, A, phi)`` with the rate constant K [m^2/(A h)], current
            prefactor A [A/m^2] and phi = F V / (2 R T) [-]
        """
        K = self._M_CR2O3 / (6 * F * self._RHO_CR2O3 * self._H_TPB) * 3600
        P_CrO2 = (
            2.26e-2
            * (state.P * self.x_H2O) ** 0.992
            * np.exp(-self._E_CRO2 / (R * state.T))
        )
        A = (
            self.j0
            * np.exp(-self.E_act / (R * state.T))
            * (P_CrO2 / state.P) ** 0.5
            * self.x_H2O**0.5
        )
        phi = F / (2 * R * state.T) * state.V
        return K, A, phi

    @differentiable
    def material_dt(self, m, state, time, eta_fuel=None, **kwargs):
        """
        Chromium poisoning rate [1/h], see :meth:`nest.degradation.ChromiumPoison.material_dt`.

        Parameters
        ----------
        m : float
            polarization degradation ratio [1 = begining of life]
        state : BoundaryData
            layer state conditions (voltage = activation voltage)
        time : float
            time [h]
        eta_fuel : float, optional
            not used
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            polarization degradation ratio rate of change [1/h]
        """
        return as_output(super().material_dt(m, state, time, eta_fuel=eta_fuel, **kwargs))

    @material_dt.grad("m")
    def _material_dt_grad_m(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.ChromiumPoison.material_dt`

        Returns
        -------
        float
            derivative [1/h]
        """
        K, A, phi = self._rate_terms(state)
        return as_output(-K * A * 2 * np.sinh(phi))

    @material_dt.grad("V")
    def _material_dt_grad_V(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_V` :meth:`~nest.grad.degradation.ChromiumPoison.material_dt`

        Returns
        -------
        float
            derivative [1/(h V)]
        """
        K, A, phi = self._rate_terms(state)
        return as_output(-K * m * A * 2 * np.cosh(phi) * F / (2 * R * state.T))

    @material_dt.grad("T")
    def _material_dt_grad_T(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.degradation.ChromiumPoison.material_dt`

        Returns
        -------
        float
            derivative [1/(h K)]
        """
        K, A, phi = self._rate_terms(state)
        dlog_A = (self.E_act + 0.5 * self._E_CRO2) / (R * state.T**2)
        return as_output(
            -K * m * A * (2 * np.sinh(phi) * dlog_A - 2 * np.cosh(phi) * phi / state.T)
        )

    @material_dt.grad("P")
    def _material_dt_grad_P(self, m, state, time, eta_fuel=None, **kwargs):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.degradation.ChromiumPoison.material_dt`

        Returns
        -------
        float
            derivative [1/(h Pa)]; the rate scales as P^(-0.004)
        """
        K, A, phi = self._rate_terms(state)
        return as_output(-K * m * A * 2 * np.sinh(phi) * 0.5 * (0.992 - 1) / state.P)

    @differentiable
    def pol_deg(self, m, **kwargs):
        """
        Polarization degradation ratio, see :meth:`nest.degradation.ChromiumPoison.pol_deg`.

        Parameters
        ----------
        m : float
            polarization degradation ratio [1 = begining of life]
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            polarization degradation ratio (equal to `m`) [-]
        """
        return as_output(super().pol_deg(m))

    @pol_deg.grad("m")
    def _pol_deg_grad_m(self, m, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.ChromiumPoison.pol_deg`

        Returns
        -------
        float
            always 1
        """
        return 1.0

    @differentiable
    def ohm_deg(self, m, **kwargs):
        """
        Ohmic degradation quantity, always 0, see :meth:`nest.degradation.ChromiumPoison.ohm_deg`.

        Parameters
        ----------
        m : float
            polarization degradation ratio [1 = begining of life]
        **kwargs
            not used (compatibility with the base call sites)

        Returns
        -------
        float
            always 0
        """
        return as_output(super().ohm_deg(m))

    @ohm_deg.grad("m")
    def _ohm_deg_grad_m(self, m, **kwargs):
        r"""
        :math:`\nabla_m` :meth:`~nest.grad.degradation.ChromiumPoison.ohm_deg`

        Returns
        -------
        float
            always 0
        """
        return 0.0
