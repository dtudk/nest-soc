"""
Cell model and channel reactions with analytical gradients.
"""

import warnings
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import newton

from nest import cell
from nest.boundary import BoundaryData
from nest.constants import R
from nest.grad._utils import as_output, central_difference, normalize_jacobian
from nest.grad.boundary import THETA, flatten, split, theta_slices, unflatten
from nest.grad.boundary import copy as copy_boundary
from nest.grad.degradation import NickelAgglomeration, NickelMigration
from nest.grad.differentiable import differentiable
from nest.grad.layers import Layer, StefanMaxwell, _require


class NoReaction(cell.NoReaction):
    """
    No channel reaction, with (zero) gradients; see :class:`nest.cell.NoReaction`.
    """

    @differentiable
    def rate(self, T, Ps, gases=None):
        """
        Kinetic reaction rate, always 0, see :meth:`nest.cell.NoReaction.rate`.

        Parameters
        ----------
        T : float
            temperature [K]
        Ps : numpy.ndarray
            partial pressures [Pa]
        gases : Mixture, optional
            not used

        Returns
        -------
        float
            always 0
        """
        return as_output(super().rate(T, Ps, gases))

    @rate.grad("T")
    def _rate_grad_T(self, T, Ps, gases=None):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.cell.NoReaction.rate`

        Returns
        -------
        float
            always 0
        """
        return 0.0

    @rate.grad("Ps")
    def _rate_grad_Ps(self, T, Ps, gases=None):
        r"""
        :math:`\nabla_{P_s}` :meth:`~nest.grad.cell.NoReaction.rate`

        Returns
        -------
        numpy.ndarray
            zeros of shape ``Ps.shape``
        """
        return np.zeros(np.shape(Ps))

    @differentiable
    def dn(self, T, Ps):
        """
        Mole flow variation from reaction, always 0, see :meth:`nest.cell.Reaction.dn`.

        Parameters
        ----------
        T : float
            temperature [K]
        Ps : numpy.ndarray
            partial pressures [Pa]

        Returns
        -------
        numpy.ndarray
            zeros of shape ``nu.shape``
        """
        return as_output(super().dn(T, Ps))

    @dn.grad("T")
    def _dn_grad_T(self, T, Ps):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.cell.NoReaction.dn`

        Returns
        -------
        numpy.ndarray
            zeros of shape ``nu.shape``
        """
        return np.zeros(np.shape(self.nu))

    @dn.grad("Ps")
    def _dn_grad_Ps(self, T, Ps):
        r"""
        :math:`\nabla_{P_s}` :meth:`~nest.grad.cell.NoReaction.dn`

        Returns
        -------
        numpy.ndarray
            zeros of shape ``nu.shape + Ps.shape``
        """
        return np.zeros(np.shape(self.nu) + np.shape(Ps))


class WaterGasShift(cell.WaterGasShift):
    """
    Water gas shift reaction with analytical gradients.

    Same parameters as :class:`nest.cell.WaterGasShift`. Gradients take one
    location: scalar ``T`` and 1-D ``Ps``.
    """

    def _terms(self, T, Ps):
        """
        Pieces of the rate ``k_T (P_CO P_H2O - P_H2 P_CO2 / K)``.

        Parameters
        ----------
        T : float
            temperature [K]
        Ps : numpy.ndarray
            partial pressures [Pa]

        Returns
        -------
        tuple
            ``(k_T, K, dlnK_dT)``: rate constant, equilibrium constant and the
            temperature derivative of ``log(K)`` [1/K]
        """
        Z = 1000 / T - 1
        K = np.exp(-0.2935 * Z**3 + 0.6351 * Z**2 + 4.1788 * Z + 0.3169)
        dlnK_dT = (-0.8805 * Z**2 + 1.2702 * Z + 4.1788) * (-1000 / T**2)
        return self.rate_T(T), K, dlnK_dT

    @differentiable
    def rate(self, T, Ps, gases=None):
        """
        Kinetic reaction rate, see :meth:`nest.cell.WaterGasShift.rate`.

        Parameters
        ----------
        T : float
            temperature [K]
        Ps : numpy.ndarray
            partial pressures [Pa]
        gases : Mixture, optional
            not used

        Returns
        -------
        float
            kinetic reaction rate
        """
        return as_output(super().rate(T, Ps, gases))

    @rate.grad("T")
    def _rate_grad_T(self, T, Ps, gases=None):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.cell.WaterGasShift.rate`

        Returns
        -------
        float
            rate derivative per K
        """
        H2_i, H2O_i, CO2_i, CO_i = self.index
        k_T, K, dlnK_dT = self._terms(T, Ps)
        reverse = k_T * Ps[H2_i] * Ps[CO2_i] / K
        return as_output(
            self.rate(T, Ps) * self.E / (R * T**2) + reverse * dlnK_dT
        )

    @rate.grad("Ps")
    def _rate_grad_Ps(self, T, Ps, gases=None):
        r"""
        :math:`\nabla_{P_s}` :meth:`~nest.grad.cell.WaterGasShift.rate`

        Returns
        -------
        numpy.ndarray
            rate derivative per Pa, shape ``Ps.shape``
        """
        H2_i, H2O_i, CO2_i, CO_i = self.index
        Ps = np.asarray(Ps, dtype=float)
        k_T, K, _ = self._terms(T, Ps)
        grad = np.zeros(Ps.shape)
        grad[CO_i] += k_T * Ps[H2O_i]
        grad[H2O_i] += k_T * Ps[CO_i]
        grad[H2_i] -= k_T * Ps[CO2_i] / K
        grad[CO2_i] -= k_T * Ps[H2_i] / K
        return grad

    @differentiable
    def dn(self, T, Ps):
        """
        Mole flow variation from reaction, see :meth:`nest.cell.Reaction.dn`.

        Parameters
        ----------
        T : float
            temperature [K]
        Ps : numpy.ndarray
            partial pressures [Pa]

        Returns
        -------
        numpy.ndarray
            mole flow variation for each specie
        """
        return as_output(super().dn(T, Ps))

    @dn.grad("T")
    def _dn_grad_T(self, T, Ps):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.cell.WaterGasShift.dn`

        Returns
        -------
        numpy.ndarray
            derivative per K, shape ``nu.shape``
        """
        return as_output(np.asarray(self.nu, dtype=float) * self.rate.grad_T(T, Ps))

    @dn.grad("Ps")
    def _dn_grad_Ps(self, T, Ps):
        r"""
        :math:`\nabla_{P_s}` :meth:`~nest.grad.cell.WaterGasShift.dn`

        Returns
        -------
        numpy.ndarray
            derivative per Pa, shape ``nu.shape + Ps.shape``
        """
        return as_output(np.multiply.outer(np.asarray(self.nu, dtype=float), self.rate.grad_Ps(T, Ps)))


def _pad(value, size: int) -> np.ndarray:
    """
    Zero-pad (along the first axis) a per-species array, as the base source loop does.

    Parameters
    ----------
    value : numpy.ndarray
        per-species values, species on the first axis
    size : int
        number of species of the flow

    Returns
    -------
    numpy.ndarray
        array with ``size`` rows
    """
    value = np.atleast_1d(np.asarray(value, dtype=float))
    padded = np.zeros((size,) + value.shape[1:])
    padded[: len(value)] = value
    return padded


def _product(partial, tangent) -> np.ndarray:
    """
    Element-wise ``partial * tangent`` with ``0 * inf = 0``.

    The ohmic degradation of interconnect oxidation has an infinite gradient
    at ``m = 0``; it must only reach the tangent columns that move ``m``.

    Parameters
    ----------
    partial : float or numpy.ndarray
        partial derivative(s), broadcast against ``tangent``
    tangent : numpy.ndarray
        tangent rows

    Returns
    -------
    numpy.ndarray
        product, 0 wherever either factor is 0
    """
    partial = np.asarray(partial, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    with np.errstate(invalid="ignore"):
        product = partial * tangent
    return np.where((partial != 0) & (tangent != 0), product, 0.0)


def _heat_capacities(gas, size: int, T) -> tuple:
    """
    Species heat capacities of a flow and their temperature derivatives.

    Parameters
    ----------
    gas : Mixture
        gas species of the kinetic model
    size : int
        number of flows; species beyond it are ignored and missing ones are 0,
        like the ``zip`` in :meth:`nest.cell.Cell.heat_capacity_flow`
    T : float
        temperature [K]

    Returns
    -------
    tuple of numpy.ndarray
        ``(cp, dcp_dT)`` of shape ``(size,)`` [J/mol.K], [J/mol.K^2]
    """
    cp, dcp_dT = np.zeros(size), np.zeros(size)
    for i, specie in enumerate(gas.species[:size]):
        cp[i] = specie.cp(T)
        dcp_dT[i] = specie.cp.grad_T(T)
    return cp, dcp_dT


class _Segment(NamedTuple):
    """Piece of the forward degradation solve between restarts at internal events."""

    t0: float
    """start time [h]"""
    t1: float
    """end time [h]"""
    sol: object
    """dense output of ``solve_ivp``"""
    t: np.ndarray
    """accepted times [h]"""
    y: np.ndarray
    """states at the accepted times"""
    start_slot: object
    """index of ``y`` whose Ni radius crossed ``r_ed_max`` at ``t0``, or None"""
    crossed: tuple
    """indices of ``y`` whose Ni radius is above ``r_ed_max`` (rate 0)"""


class _Forward(NamedTuple):
    """Forward degradation solve used by the adjoint."""

    segments: list
    """:class:`_Segment` pieces in time order"""
    t: float
    """final time [h] (``t_max`` or the terminal event)"""
    y: np.ndarray
    """final state"""
    terminal: object
    """index of the user event that stopped the solve, or None"""
    events: list
    """user events"""
    work: BoundaryData
    """boundary copy passed to the right-hand side and the events"""


def _user_event(event):
    """
    User event called as ``event(t, y, boundary, mode)`` from a segment solve.

    Parameters
    ----------
    event : callable
        ``solve_ivp`` event of :meth:`Cell.solve_for_time`

    Returns
    -------
    callable
        event taking the segment arguments, with the same ``terminal`` and ``direction``
    """

    def wrapped(t, y, work, mode, crossed):
        return event(t, y, work, mode)

    wrapped.terminal = getattr(event, "terminal", False)
    wrapped.direction = getattr(event, "direction", 0)
    return wrapped


def _fired(sol, events: list):
    """
    Terminal event that stopped a ``solve_ivp`` run.

    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        result of ``solve_ivp``
    events : list of callable
        events passed to ``solve_ivp``

    Returns
    -------
    int or None
        index of the event, None if the run reached the end of the interval
    """
    if sol.status != 1:
        return None
    for i, event in enumerate(events):
        hits = sol.t_events[i]
        if getattr(event, "terminal", False) and len(hits) and hits[-1] == sol.t[-1]:
            return i
    return None


def _key(value):
    """
    Hashable cache key of an option value (arrays by content).

    Parameters
    ----------
    value : object
        option value

    Returns
    -------
    object
        hashable key
    """
    if isinstance(value, np.ndarray):
        return (value.shape, value.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(_key(v) for v in value)
    return value if isinstance(value, (str, int, float, bool, type(None))) else id(value)


class Cell(cell.Cell):
    """
    Electrochemical cell model with analytical gradients.

    Same parameters as :class:`nest.cell.Cell`. Layers must be
    :class:`nest.grad.layers.Layer` and reactions ``nest.grad.cell`` reactions
    (a :class:`TypeError` is raised otherwise); ``reactions=None`` means
    ``(NoReaction(),)``. Degradation vectors ``pol_deg`` and ``ohm_deg`` are
    indexed by layer: fuel electrode, air electrode, then electrolyte layers;
    ``None`` means each layer's initial material value.

    The element and voltage solves use tight Newton tolerances, so the
    implicit gradients match the computed values. Gradients of the
    degradation simulation (:meth:`solve_for_time_gradients`) use the adjoint
    method with :attr:`ivp_options`.

    Gradients of the steady solves (:meth:`solve_for_voltage`,
    :meth:`solve_for_current`) and of the degradation rates
    (:meth:`advance_time_step`) come from forward tangents propagated element
    by element. Tangent columns are the boundary attributes in the flat order
    of :data:`nest.grad.boundary.THETA`, followed by the degradation variables
    being differentiated. The Newton guesses (``boundary.j`` in voltage mode,
    ``boundary.V`` in current mode) and ``fuel_pressure`` / ``air_pressure``
    have exactly zero gradients. Gradient methods work on copies of
    ``boundary``; the value methods mutate it like the base model.
    """

    step_tol = 1e-9
    """Newton tolerance of the element current density [A/m^2]."""
    voltage_tol = 1e-12
    """Newton tolerance of the cell voltage in current mode [V]."""
    newton_maxiter = 100
    """Maximum Newton iterations."""
    ivp_options = {"method": "RK45", "rtol": 1e-8}
    """Default ``solve_ivp`` options of the gradients of :meth:`solve_for_time`; ``atol`` follows from ``rtol``."""
    adjoint_t_min = 1e-8
    """End of the backward adjoint integration, relative to ``t_max``."""

    def __init__(
        self,
        area: float,
        electrode_fuel: Layer,
        electrolyte: tuple[Layer, ...],
        electrode_air: Layer,
        elements: int = 10,
        reactions=None,
        delta_channel=0.001,
        fuel_channel=None,
        air_channel=None,
        fuel_support=None,
        energy_balance: bool = False,
    ):
        reactions = (NoReaction(),) if reactions is None else tuple(reactions)
        electrolyte = tuple(electrolyte)
        _require(electrode_fuel, Layer, "electrode_fuel")
        _require(electrode_air, Layer, "electrode_air")
        for layer in electrolyte:
            _require(layer, Layer, "electrolyte layer")
        for reaction in reactions:
            if not isinstance(reaction, (NoReaction, WaterGasShift)):
                raise TypeError(
                    "reactions must be nest.grad.cell.NoReaction or WaterGasShift, got "
                    f"{type(reaction).__module__}.{type(reaction).__qualname__}"
                )
        super().__init__(
            area, electrode_fuel, electrolyte, electrode_air, elements, reactions,
            delta_channel, fuel_channel, air_channel, fuel_support, energy_balance,
        )

    def _layers(self) -> tuple:
        """
        Layers in degradation-vector order.

        Returns
        -------
        tuple of Layer
            fuel electrode, air electrode, then the electrolyte layers
        """
        return (self.electrode_fuel, self.electrode_air) + tuple(self.electrolyte)

    @differentiable
    def V_nernst(self, T, Ps_fuel, Ps_air):
        """
        Thermodynamic voltage [V], see :meth:`nest.cell.Cell.V_nernst`.

        Parameters
        ----------
        T : float
            temperature [K]
        Ps_fuel : numpy.ndarray
            partial pressures fuel side [Pa]
        Ps_air : numpy.ndarray
            partial pressures air side [Pa]

        Returns
        -------
        float
            Nernst voltage [V]
        """
        return as_output(super().V_nernst(T, Ps_fuel, Ps_air))

    @V_nernst.grad("T")
    def _V_nernst_grad_T(self, T, Ps_fuel, Ps_air):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.cell.Cell.V_nernst`

        Returns
        -------
        float
            voltage derivative [V/K]
        """
        return as_output(
            -(
                np.asarray(self.electrode_fuel.kinetic.V_nernst_half.grad_T(T, Ps_fuel))
                + self.electrode_air.kinetic.V_nernst_half.grad_T(T, Ps_air)
            )
        )

    @V_nernst.grad("Ps_fuel")
    def _V_nernst_grad_Ps_fuel(self, T, Ps_fuel, Ps_air):
        r"""
        :math:`\nabla_{P_{fuel}}` :meth:`~nest.grad.cell.Cell.V_nernst`

        Returns
        -------
        numpy.ndarray
            voltage derivative [V/Pa], shape ``Ps_fuel.shape``
        """
        return as_output(-np.asarray(self.electrode_fuel.kinetic.V_nernst_half.grad_Ps(T, Ps_fuel)))

    @V_nernst.grad("Ps_air")
    def _V_nernst_grad_Ps_air(self, T, Ps_fuel, Ps_air):
        r"""
        :math:`\nabla_{P_{air}}` :meth:`~nest.grad.cell.Cell.V_nernst`

        Returns
        -------
        numpy.ndarray
            voltage derivative [V/Pa], shape ``Ps_air.shape``
        """
        return as_output(-np.asarray(self.electrode_air.kinetic.V_nernst_half.grad_Ps(T, Ps_air)))

    @differentiable
    def V_thermoneutral(self, T):
        """
        Thermoneutral voltage [V], see :meth:`nest.cell.Cell.V_thermoneutral`.

        Parameters
        ----------
        T : float
            temperature [K]

        Returns
        -------
        float
            thermoneutral voltage [V]
        """
        return as_output(super().V_thermoneutral(T))

    @V_thermoneutral.grad("T")
    def _V_thermoneutral_grad_T(self, T):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.cell.Cell.V_thermoneutral`

        Returns
        -------
        float
            voltage derivative [V/K]
        """
        return as_output(
            -(
                np.asarray(self.electrode_fuel.kinetic.V_thermoneutral_half.grad_T(T))
                + self.electrode_air.kinetic.V_thermoneutral_half.grad_T(T)
            )
        )

    def _degradation(self, pol_deg, ohm_deg) -> list:
        """
        Per-layer degradation keywords of :meth:`V`.

        Parameters
        ----------
        pol_deg : numpy.ndarray or None
            polarization material variables, indexed by layer
        ohm_deg : numpy.ndarray or None
            ohmic material variables, indexed by layer

        Returns
        -------
        list of dict
            ``{"pol_deg", "ohm_deg"}`` per layer, None where not given
        """
        return [
            {
                "pol_deg": None if pol_deg is None else pol_deg[k],
                "ohm_deg": None if ohm_deg is None else ohm_deg[k],
            }
            for k in range(len(self._layers()))
        ]

    def _V_partials(self, j, T, Ps_fuel, Ps_air, pol_deg=None, ohm_deg=None) -> dict:
        """
        Partial derivatives of the cell voltage :meth:`V`.

        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            temperature [K]
        Ps_fuel : numpy.ndarray
            partial pressures fuel side [Pa]
        Ps_air : numpy.ndarray
            partial pressures air side [Pa]
        pol_deg : numpy.ndarray, optional
            polarization material variables, indexed by layer
        ohm_deg : numpy.ndarray, optional
            ohmic material variables, indexed by layer

        Returns
        -------
        dict
            ``{"j", "T", "Ps_fuel", "Ps_air", "pol_deg", "ohm_deg"}``; the
            degradation entries have one value per layer
        """
        layers = self._layers()
        degradation = self._degradation(pol_deg, ohm_deg)
        partials = {
            "j": 0.0, "T": 0.0,
            "pol_deg": np.zeros(len(layers)), "ohm_deg": np.zeros(len(layers)),
        }
        # Electrodes: losses and Nernst voltage at the reaction-site pressures
        for k, (name, Ps) in enumerate((("Ps_fuel", Ps_fuel), ("Ps_air", Ps_air))):
            layer = layers[k]
            Ps_star = layer.Ps_star(j, T, Ps)
            nernst_T = layer.kinetic.V_nernst_half.grad_T(T, Ps_star)
            nernst_Ps = np.asarray(layer.kinetic.V_nernst_half.grad_Ps(T, Ps_star))
            loss = layer._V_partials(j, T, Ps_star, **degradation[k])
            a = -nernst_Ps - loss["Ps"]  # derivative w.r.t. the reaction-site pressures
            partials["j"] += a @ layer.Ps_star.grad_j(j, T, Ps) - loss["j"]
            partials["T"] += a @ layer.Ps_star.grad_T(j, T, Ps) - nernst_T - loss["T"]
            partials[name] = a @ np.asarray(layer.Ps_star.grad_Ps(j, T, Ps))
            partials["pol_deg"][k] = -loss["pol_deg"]
            partials["ohm_deg"][k] = -loss["ohm_deg"]
        for k, layer in enumerate(layers[2:], start=2):
            loss = layer._V_partials(j, T, **degradation[k])
            partials["j"] -= loss["j"]
            partials["T"] -= loss["T"]
            partials["pol_deg"][k] = -loss["pol_deg"]
            partials["ohm_deg"][k] = -loss["ohm_deg"]
        return partials

    @differentiable
    def V(self, j, T, Ps_fuel, Ps_air, pol_deg=None, ohm_deg=None):
        """
        Cell voltage [V], see :meth:`nest.cell.Cell.V`.

        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            temperature [K]
        Ps_fuel : numpy.ndarray
            partial pressures fuel side [Pa]
        Ps_air : numpy.ndarray
            partial pressures air side [Pa]
        pol_deg : numpy.ndarray, optional
            polarization material variables, indexed by layer; None means ``m0``
        ohm_deg : numpy.ndarray, optional
            ohmic material variables, indexed by layer; None means ``m0``

        Returns
        -------
        float
            cell voltage [V]
        """
        kwargs = {
            key: value
            for key, value in (("pol_deg", pol_deg), ("ohm_deg", ohm_deg))
            if value is not None
        }
        return as_output(super().V(j, T, Ps_fuel, Ps_air, **kwargs))

    @V.grad("j")
    def _V_grad_j(self, j, T, Ps_fuel, Ps_air, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.cell.Cell.V`

        Returns
        -------
        float
            voltage derivative [V m^2/A]
        """
        return as_output(self._V_partials(j, T, Ps_fuel, Ps_air, pol_deg, ohm_deg)["j"])

    @V.grad("T")
    def _V_grad_T(self, j, T, Ps_fuel, Ps_air, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.cell.Cell.V`

        Returns
        -------
        float
            voltage derivative [V/K]
        """
        return as_output(self._V_partials(j, T, Ps_fuel, Ps_air, pol_deg, ohm_deg)["T"])

    @V.grad("Ps_fuel")
    def _V_grad_Ps_fuel(self, j, T, Ps_fuel, Ps_air, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_{P_{fuel}}` :meth:`~nest.grad.cell.Cell.V`

        Returns
        -------
        numpy.ndarray
            voltage derivative [V/Pa], shape ``Ps_fuel.shape``
        """
        return as_output(self._V_partials(j, T, Ps_fuel, Ps_air, pol_deg, ohm_deg)["Ps_fuel"])

    @V.grad("Ps_air")
    def _V_grad_Ps_air(self, j, T, Ps_fuel, Ps_air, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_{P_{air}}` :meth:`~nest.grad.cell.Cell.V`

        Returns
        -------
        numpy.ndarray
            voltage derivative [V/Pa], shape ``Ps_air.shape``
        """
        return as_output(self._V_partials(j, T, Ps_fuel, Ps_air, pol_deg, ohm_deg)["Ps_air"])

    @V.grad("pol_deg")
    def _V_grad_pol_deg(self, j, T, Ps_fuel, Ps_air, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_{\pi}` :meth:`~nest.grad.cell.Cell.V`

        Returns
        -------
        numpy.ndarray
            voltage derivative per unit material variable of each layer,
            shape ``pol_deg.shape`` (one entry per layer if None)
        """
        partials = self._V_partials(j, T, Ps_fuel, Ps_air, pol_deg, ohm_deg)["pol_deg"]
        return _pad(partials, len(partials) if pol_deg is None else len(pol_deg))

    @V.grad("ohm_deg")
    def _V_grad_ohm_deg(self, j, T, Ps_fuel, Ps_air, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_{\omega}` :meth:`~nest.grad.cell.Cell.V`

        Returns
        -------
        numpy.ndarray
            voltage derivative per unit material variable of each layer,
            shape ``ohm_deg.shape`` (one entry per layer if None)
        """
        partials = self._V_partials(j, T, Ps_fuel, Ps_air, pol_deg, ohm_deg)["ohm_deg"]
        return _pad(partials, len(partials) if ohm_deg is None else len(ohm_deg))

    def _dn_fuel_partials(self, j, boundary: BoundaryData) -> dict:
        """
        Partial derivatives of :meth:`dn_fuel`.

        Parameters
        ----------
        j : float
            current density [A/m^2]
        boundary : BoundaryData
            inlet state variables of the element

        Returns
        -------
        dict
            ``{"j", "n_fuel", "T", "P"}`` derivatives, species on the first axis
        """
        area = self.area / self.elements
        n_fuel = np.asarray(boundary.n_fuel, dtype=float)
        # Intermediate flows after electrochemistry, with the flux broadcast to the flows
        flux = np.atleast_1d(self.electrode_fuel.kinetic.mol_flux(j))
        k = np.atleast_1d(self.electrode_fuel.kinetic.mol_flux.grad_j(j))
        q = n_fuel + flux * area
        x = q / sum(q)
        Ps = boundary.P * x
        # Reaction source, zero-padded to the flows like the base loop
        source_T = np.zeros(len(Ps))
        source_Ps = np.zeros((len(Ps), len(Ps)))
        for reaction in self.reactions:
            source_T += _pad(reaction.dn.grad_T(boundary.T, Ps), len(Ps))
            source_Ps += _pad(reaction.dn.grad_Ps(boundary.T, Ps), len(Ps))
        shape = np.broadcast_shapes(k.shape, source_T.shape)
        dq_dj = np.broadcast_to(k, q.shape) * area
        dPs_dq = boundary.P * normalize_jacobian(q)
        source_dq = source_Ps @ dPs_dq
        return {
            "j": np.broadcast_to(area * (k + source_dq @ dq_dj), shape),
            "n_fuel": np.broadcast_to(area * source_dq, shape + n_fuel.shape),
            "T": np.broadcast_to(area * source_T, shape),
            "P": np.broadcast_to(area * source_Ps @ x, shape),
        }

    @differentiable
    def dn_fuel(self, j, boundary):
        """
        Net molar flow per element, fuel side [mol/s], see :meth:`nest.cell.Cell.dn_fuel`.

        Parameters
        ----------
        j : float
            current density [A/m^2]
        boundary : BoundaryData
            inlet state variables of the element

        Returns
        -------
        numpy.ndarray
            net molar flow per element for each fuel-side specie [mol/s]
        """
        return as_output(super().dn_fuel(j, boundary))

    @dn_fuel.grad("j")
    def _dn_fuel_grad_j(self, j, boundary):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.cell.Cell.dn_fuel`

        Returns
        -------
        numpy.ndarray
            derivative [mol m^2/(s A)] per species
        """
        return as_output(self._dn_fuel_partials(j, boundary)["j"])

    @dn_fuel.grad("n_fuel")
    def _dn_fuel_grad_n_fuel(self, j, boundary):
        r"""
        :math:`\nabla_{n_{fuel}}` :meth:`~nest.grad.cell.Cell.dn_fuel`

        Returns
        -------
        numpy.ndarray
            Jacobian [-] of shape ``(n_species, len(boundary.n_fuel))``
        """
        return as_output(self._dn_fuel_partials(j, boundary)["n_fuel"])

    @dn_fuel.grad("T")
    def _dn_fuel_grad_T(self, j, boundary):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.cell.Cell.dn_fuel`

        Returns
        -------
        numpy.ndarray
            derivative [mol/(s K)] per species
        """
        return as_output(self._dn_fuel_partials(j, boundary)["T"])

    @dn_fuel.grad("P")
    def _dn_fuel_grad_P(self, j, boundary):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.cell.Cell.dn_fuel`

        Returns
        -------
        numpy.ndarray
            derivative [mol/(s Pa)] per species
        """
        return as_output(self._dn_fuel_partials(j, boundary)["P"])

    @differentiable
    def dn_air(self, j):
        """
        Net molar flow per element, air side [mol/s], see :meth:`nest.cell.Cell.dn_air`.

        Parameters
        ----------
        j : float
            current density [A/m^2]

        Returns
        -------
        numpy.ndarray
            net molar flow per element for each air-side specie [mol/s]
        """
        return as_output(super().dn_air(j))

    @dn_air.grad("j")
    def _dn_air_grad_j(self, j):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.cell.Cell.dn_air`

        Returns
        -------
        numpy.ndarray
            derivative [mol m^2/(s A)] per species
        """
        k = np.atleast_1d(self.electrode_air.kinetic.mol_flux.grad_j(j))
        return as_output(k * self.area / self.elements)

    def _Ps_fuel_partials(self, boundary: BoundaryData, j) -> dict:
        """
        Partial derivatives of :meth:`Ps_fuel`.

        Parameters
        ----------
        boundary : BoundaryData
            boundary state variables
        j : float
            current density [A/m^2]

        Returns
        -------
        dict
            ``{"j", "n_fuel", "T", "P"}`` derivatives, species on the first axis
        """
        n = np.asarray(boundary.n_fuel, dtype=float)
        dn = self._dn_fuel_partials(j, boundary)
        w = n + self.dn_fuel(j, boundary)
        N_w = 0.5 * boundary.P * normalize_jacobian(w)
        return {
            "j": N_w @ dn["j"],
            "n_fuel": 0.5 * boundary.P * normalize_jacobian(n) + N_w @ (np.eye(len(n)) + dn["n_fuel"]),
            "T": N_w @ dn["T"],
            "P": 0.5 * (n / sum(n) + w / sum(w)) + N_w @ dn["P"],
        }

    @differentiable
    def Ps_fuel(self, boundary, j):
        """
        Mean partial pressures of the fuel channel element [Pa], see :meth:`nest.cell.Cell.Ps_fuel`.

        Parameters
        ----------
        boundary : BoundaryData
            boundary state variables
        j : float
            current density [A/m^2]

        Returns
        -------
        numpy.ndarray
            partial pressures for the fuel channel [Pa]
        """
        return as_output(super().Ps_fuel(boundary, j))

    @Ps_fuel.grad("j")
    def _Ps_fuel_grad_j(self, boundary, j):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.cell.Cell.Ps_fuel`

        Returns
        -------
        numpy.ndarray
            pressure derivative [Pa m^2/A] per species
        """
        return as_output(self._Ps_fuel_partials(boundary, j)["j"])

    @Ps_fuel.grad("n_fuel")
    def _Ps_fuel_grad_n_fuel(self, boundary, j):
        r"""
        :math:`\nabla_{n_{fuel}}` :meth:`~nest.grad.cell.Cell.Ps_fuel`

        Returns
        -------
        numpy.ndarray
            Jacobian [Pa s/mol] of shape ``(n_species, len(boundary.n_fuel))``
        """
        return as_output(self._Ps_fuel_partials(boundary, j)["n_fuel"])

    @Ps_fuel.grad("T")
    def _Ps_fuel_grad_T(self, boundary, j):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.cell.Cell.Ps_fuel`

        Returns
        -------
        numpy.ndarray
            pressure derivative [Pa/K] per species
        """
        return as_output(self._Ps_fuel_partials(boundary, j)["T"])

    @Ps_fuel.grad("P")
    def _Ps_fuel_grad_P(self, boundary, j):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.cell.Cell.Ps_fuel`

        Returns
        -------
        numpy.ndarray
            pressure derivative [-] per species
        """
        return as_output(self._Ps_fuel_partials(boundary, j)["P"])

    def _Ps_air_partials(self, boundary: BoundaryData, j) -> dict:
        """
        Partial derivatives of :meth:`Ps_air`.

        Parameters
        ----------
        boundary : BoundaryData
            boundary state variables
        j : float
            current density [A/m^2]

        Returns
        -------
        dict
            ``{"j", "n_air", "T", "P"}`` derivatives, species on the first axis
        """
        n = np.asarray(boundary.n_air, dtype=float)
        # The air flux broadcasts over all flows (e.g. O2-only kinetics with O2/N2 flows)
        w = n + self.dn_air(j)
        dw_dj = np.broadcast_to(self.dn_air.grad_j(j), w.shape)
        N_w = 0.5 * boundary.P * normalize_jacobian(w)
        return {
            "j": N_w @ dw_dj,
            "n_air": 0.5 * boundary.P * normalize_jacobian(n) + N_w,
            "T": np.zeros(w.shape),
            "P": 0.5 * (n / sum(n) + w / sum(w)),
        }

    @differentiable
    def Ps_air(self, boundary, j):
        """
        Mean partial pressures of the air channel element [Pa], see :meth:`nest.cell.Cell.Ps_air`.

        Parameters
        ----------
        boundary : BoundaryData
            boundary state variables
        j : float
            current density [A/m^2]

        Returns
        -------
        numpy.ndarray
            partial pressures for the air channel [Pa]
        """
        return as_output(super().Ps_air(boundary, j))

    @Ps_air.grad("j")
    def _Ps_air_grad_j(self, boundary, j):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.cell.Cell.Ps_air`

        Returns
        -------
        numpy.ndarray
            pressure derivative [Pa m^2/A] per species
        """
        return as_output(self._Ps_air_partials(boundary, j)["j"])

    @Ps_air.grad("n_air")
    def _Ps_air_grad_n_air(self, boundary, j):
        r"""
        :math:`\nabla_{n_{air}}` :meth:`~nest.grad.cell.Cell.Ps_air`

        Returns
        -------
        numpy.ndarray
            Jacobian [Pa s/mol] of shape ``(n_species, n_species)``
        """
        return as_output(self._Ps_air_partials(boundary, j)["n_air"])

    @Ps_air.grad("T")
    def _Ps_air_grad_T(self, boundary, j):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.cell.Cell.Ps_air`

        Returns
        -------
        numpy.ndarray
            always 0, per species
        """
        return as_output(self._Ps_air_partials(boundary, j)["T"])

    @Ps_air.grad("P")
    def _Ps_air_grad_P(self, boundary, j):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.cell.Cell.Ps_air`

        Returns
        -------
        numpy.ndarray
            pressure derivative [-] per species
        """
        return as_output(self._Ps_air_partials(boundary, j)["P"])

    def advance_step_area(self, boundary: BoundaryData, pol_deg=None, ohm_deg=None) -> BoundaryData:
        """
        Solves the current density of one element, see :meth:`nest.cell.Cell.advance_step_area`.

        Uses the secant method with the tolerance :attr:`step_tol`.

        Parameters
        ----------
        boundary : BoundaryData
            inlet state variables of the element
        pol_deg : numpy.ndarray, optional
            polarization material variables of the element, indexed by layer
        ohm_deg : numpy.ndarray, optional
            ohmic material variables of the element, indexed by layer

        Returns
        -------
        BoundaryData
            outlet state variables of the element with the solved current density
        """

        def residual(x):
            return boundary.V - self.V(
                x,
                boundary.T,
                self.Ps_fuel(boundary, x),
                self.Ps_air(boundary, x),
                pol_deg,
                ohm_deg,
            )

        # Secant method: the analytical slope costs more than the iterations it saves
        j = newton(residual, boundary.j, tol=self.step_tol, maxiter=self.newton_maxiter)

        # Record solution (as in the base model)
        n_out_fuel = boundary.n_fuel + self.dn_fuel(j, boundary)
        n_out_air = boundary.n_air + self.dn_air(j)
        T_out = boundary.T
        if self.energy_balance:
            C_fuel = self.heat_capacity_flow(
                self.electrode_fuel.kinetic.gas, n_out_fuel, boundary.T
            )
            C_air = self.heat_capacity_flow(
                self.electrode_air.kinetic.gas, n_out_air, boundary.T
            )
            heat = (
                j
                * self.area
                / self.elements
                * (self.V_thermoneutral(boundary.T) - boundary.V)
            )
            T_out = boundary.T + heat / (C_fuel + C_air)
        return BoundaryData(
            V=boundary.V,
            j=j,
            n_fuel=n_out_fuel,
            n_air=n_out_air,
            T=T_out,
            P=boundary.P,
            fuel_pressure=boundary.fuel_pressure,
            air_pressure=boundary.air_pressure,
        )

    @staticmethod
    def _deg_kwargs(pol_deg, ohm_deg) -> dict:
        """
        Degradation keywords of the base solves, without the ones not given.

        Parameters
        ----------
        pol_deg : numpy.ndarray or None
            polarization material variables, indexed by element and then by layer
        ohm_deg : numpy.ndarray or None
            ohmic material variables, indexed by element and then by layer

        Returns
        -------
        dict
            ``{"pol_deg", "ohm_deg"}`` entries that are not None
        """
        return {
            key: value
            for key, value in (("pol_deg", pol_deg), ("ohm_deg", ohm_deg))
            if value is not None
        }

    def _deg_matrices(self, pol_deg, ohm_deg) -> tuple:
        """
        Degradation matrices of the solves, with None meaning the initial material values.

        Parameters
        ----------
        pol_deg : numpy.ndarray or None
            polarization material variables, indexed by element and then by layer
        ohm_deg : numpy.ndarray or None
            ohmic material variables, indexed by element and then by layer

        Returns
        -------
        tuple of numpy.ndarray
            ``(pol_deg, ohm_deg)``, each of shape ``(elements, layers)``
        """
        m0 = np.array([layer.degradation.m0 for layer in self._layers()], dtype=float)
        return tuple(
            np.tile(m0, (self.elements, 1)) if value is None else np.array(value, dtype=float)
            for value in (pol_deg, ohm_deg)
        )

    def _element_tangent(self, inlet, outlet, pol_deg, ohm_deg, d_inlet, d_deg) -> tuple:
        r"""
        Forward tangents of one solved element, see :meth:`advance_step_area`.

        The current density solves the residual ``V - V(j, T, Ps_fuel, Ps_air)``
        at the inlet state; its tangent follows from the implicit function
        theorem. The recorded overpotentials are evaluated at the outlet state,
        as in :meth:`nest.cell.Cell.solve_for_voltage`.

        .. math:: \dot{j} = -(r_z \dot{z} + \dot{V} + r_d \dot{d}) / r_j

        Parameters
        ----------
        inlet : BoundaryData
            inlet state of the element
        outlet : BoundaryData
            solved outlet state of the element (holds the current density)
        pol_deg : numpy.ndarray
            polarization material variables of the element, indexed by layer
        ohm_deg : numpy.ndarray
            ohmic material variables of the element, indexed by layer
        d_inlet : dict
            inlet tangents ``{"V", "T", "P"}`` of shape ``(n_c,)`` and
            ``{"n_fuel", "n_air"}`` of shape ``(n_species, n_c)``
        d_deg : numpy.ndarray
            tangents of ``(pol_deg, ohm_deg)``, shape ``(2, layers, n_c)``

        Returns
        -------
        tuple
            ``(d_outlet, rows)``: outlet tangents (same keys as ``d_inlet``) and
            the tangents of the element's solution column, shape ``(n_var, n_c)``
        """
        j = outlet.j
        area = self.area / self.elements
        dV, dT, dP = d_inlet["V"], d_inlet["T"], d_inlet["P"]
        d_pol, d_ohm = d_deg

        # Current density: implicit derivative of the residual at the inlet state
        Ps_fuel = self.Ps_fuel(inlet, j)
        Ps_air = self.Ps_air(inlet, j)
        U = self._V_partials(j, inlet.T, Ps_fuel, Ps_air, pol_deg, ohm_deg)
        fuel = self._Ps_fuel_partials(inlet, j)
        air = self._Ps_air_partials(inlet, j)
        r_j = -(U["j"] + U["Ps_fuel"] @ fuel["j"] + U["Ps_air"] @ air["j"])
        d_Ps_fuel = fuel["n_fuel"] @ d_inlet["n_fuel"] + np.outer(fuel["T"], dT) + np.outer(fuel["P"], dP)
        d_Ps_air = air["n_air"] @ d_inlet["n_air"] + np.outer(air["T"], dT) + np.outer(air["P"], dP)
        d_residual = (
            dV
            - U["T"] * dT
            - U["Ps_fuel"] @ d_Ps_fuel
            - U["Ps_air"] @ d_Ps_air
            - _product(U["pol_deg"][:, None], d_pol).sum(axis=0)
            - _product(U["ohm_deg"][:, None], d_ohm).sum(axis=0)
        )
        dj = -d_residual / r_j

        # Outlet flows and temperature
        dn = self._dn_fuel_partials(j, inlet)
        d_n_fuel = (
            d_inlet["n_fuel"]
            + dn["n_fuel"] @ d_inlet["n_fuel"]
            + np.outer(dn["j"], dj)
            + np.outer(dn["T"], dT)
            + np.outer(dn["P"], dP)
        )
        k_air = np.broadcast_to(self.dn_air.grad_j(j), np.shape(inlet.n_air))
        d_n_air = d_inlet["n_air"] + np.outer(k_air, dj)
        dT_out = dT
        if self.energy_balance:
            # T_out = T + heat / C with the heat capacities at the inlet temperature
            V_tn = self.V_thermoneutral(inlet.T)
            heat = j * area * (V_tn - inlet.V)
            cp_fuel, dcp_fuel = _heat_capacities(self.electrode_fuel.kinetic.gas, len(outlet.n_fuel), inlet.T)
            cp_air, dcp_air = _heat_capacities(self.electrode_air.kinetic.gas, len(outlet.n_air), inlet.T)
            C = cp_fuel @ outlet.n_fuel + cp_air @ outlet.n_air
            d_heat = area * (
                (V_tn - inlet.V) * dj + j * self.V_thermoneutral.grad_T(inlet.T) * dT - j * dV
            )
            dC = (
                cp_fuel @ d_n_fuel
                + cp_air @ d_n_air
                + (dcp_fuel @ outlet.n_fuel + dcp_air @ outlet.n_air) * dT
            )
            dT_out = dT + d_heat / C - heat * dC / C**2

        # Recorded overpotentials: outlet state, channel pressures (base behaviour)
        layers = self._layers()
        aux = 4 + len(outlet.n_fuel) + len(outlet.n_air)
        rows = np.zeros((aux + len(layers), len(dV)))
        channels = (
            (self.Ps_fuel(outlet, j), self._Ps_fuel_partials(outlet, j), "n_fuel", d_n_fuel),
            (self.Ps_air(outlet, j), self._Ps_air_partials(outlet, j), "n_air", d_n_air),
        )
        for k, (Ps, partials, name, d_n) in enumerate(channels):
            d_Ps = (
                np.outer(partials["j"], dj)
                + partials[name] @ d_n
                + np.outer(partials["T"], dT_out)
                + np.outer(partials["P"], dP)
            )
            loss = layers[k]._V_partials(j, outlet.T, Ps, pol_deg[k], ohm_deg[k])
            rows[aux + k] = (
                loss["j"] * dj
                + loss["T"] * dT_out
                + np.asarray(loss["Ps"]) @ d_Ps
                + _product(loss["pol_deg"], d_pol[k])
                + _product(loss["ohm_deg"], d_ohm[k])
            )
        for k, layer in enumerate(layers[2:], start=2):
            V_ohm = layer.V_ohm
            rows[aux + k] = (
                V_ohm.grad_j(j, outlet.T, ohm_deg[k]) * dj
                + V_ohm.grad_T(j, outlet.T, ohm_deg[k]) * dT_out
                + _product(V_ohm.grad_ohm_deg(j, outlet.T, ohm_deg[k]), d_ohm[k])
            )
        n_fuel = len(outlet.n_fuel)
        rows[0], rows[1], rows[2], rows[3] = dV, dj, dT_out, dP
        rows[4 : 4 + n_fuel] = d_n_fuel
        rows[4 + n_fuel : aux] = d_n_air
        d_outlet = {"V": dV, "n_fuel": d_n_fuel, "n_air": d_n_air, "T": dT_out, "P": dP}
        return d_outlet, rows

    def _voltage_tangent(self, boundary: BoundaryData, pol_deg, ohm_deg, deg_seed, solution=None) -> tuple:
        """
        Voltage-mode solution matrix and its forward tangents.

        Parameters
        ----------
        boundary : BoundaryData
            initial value conditions; not modified
        pol_deg : numpy.ndarray
            polarization material variables, shape ``(elements, layers)``
        ohm_deg : numpy.ndarray
            ohmic material variables, shape ``(elements, layers)``
        deg_seed : numpy.ndarray
            derivatives of ``(pol_deg, ohm_deg)`` w.r.t. the ``n_d`` degradation
            columns, shape ``(elements, 2, layers, n_d)``
        solution : numpy.ndarray, optional
            solution matrix of :meth:`solve_for_voltage` at ``boundary``, if known

        Returns
        -------
        tuple of numpy.ndarray
            ``(S, dS)``: solution matrix ``(n_var, elements)`` and tangents
            ``(n_var, elements, n_theta + n_d)``; columns follow
            :data:`nest.grad.boundary.THETA`, then the degradation columns.
            The ``j`` column (Newton guess) is 0.
        """
        if solution is None:
            solution = self.solve_for_voltage(copy_boundary(boundary), pol_deg=pol_deg, ohm_deg=ohm_deg)
        slices = theta_slices(boundary)
        n_theta = slices[THETA[-1]].stop
        n_columns = n_theta + deg_seed.shape[-1]
        n_fuel, n_air = len(boundary.n_fuel), len(boundary.n_air)
        identity = np.eye(n_columns)
        d_inlet = {name: identity[slices[name]] for name in ("n_fuel", "n_air")}
        d_inlet.update({name: identity[slices[name].start] for name in ("V", "T", "P")})
        d_deg = np.zeros((2, pol_deg.shape[1], n_columns))
        d_solution = np.zeros(solution.shape + (n_columns,))
        inlet = boundary
        for e in range(self.elements):
            outlet = BoundaryData(
                V=solution[0][e],
                j=solution[1][e],
                n_fuel=solution[4 : 4 + n_fuel, e],
                n_air=solution[4 + n_fuel : 4 + n_fuel + n_air, e],
                T=solution[2][e],
                P=solution[3][e],
                fuel_pressure=boundary.fuel_pressure,
                air_pressure=boundary.air_pressure,
            )
            d_deg[..., n_theta:] = deg_seed[e]
            d_inlet, d_solution[:, e] = self._element_tangent(
                inlet, outlet, pol_deg[e], ohm_deg[e], d_inlet, d_deg
            )
            inlet = outlet
        return solution, d_solution

    def _current_tangent(self, d_solution: np.ndarray, boundary: BoundaryData) -> np.ndarray:
        r"""
        Current-mode tangents from the voltage-mode ones at the solved voltage.

        The voltage solves ``j - mean(j_e(V, q)) = 0``, so it moves with the
        target current ``j`` and every other column ``q``.

        .. math:: \partial V^* / \partial q = -\partial_q \bar{j} / \partial_V \bar{j}

        Parameters
        ----------
        d_solution : numpy.ndarray
            voltage-mode tangents, see :meth:`_voltage_tangent`
        boundary : BoundaryData
            boundary giving the column layout

        Returns
        -------
        numpy.ndarray
            current-mode tangents, same shape; the ``V`` column (Newton guess) is 0
        """
        slices = theta_slices(boundary)
        i_V, i_j = slices["V"].start, slices["j"].start
        dj_mean = np.sum(d_solution[1], axis=0) / self.elements
        dV_star = -dj_mean / dj_mean[i_V]  # -1 in the V column cancels it
        dV_star[i_j] += 1 / dj_mean[i_V]
        return d_solution + _product(d_solution[..., i_V : i_V + 1], dV_star)

    def _solve_tangent(self, boundary: BoundaryData, mode: str, pol_deg, ohm_deg, deg_seed) -> tuple:
        """
        Steady solution matrix of either mode and its forward tangents.

        Parameters
        ----------
        boundary : BoundaryData
            initial value conditions; not modified
        mode : str
            operational mode: "current" or "voltage" constant
        pol_deg : numpy.ndarray
            polarization material variables, shape ``(elements, layers)``
        ohm_deg : numpy.ndarray
            ohmic material variables, shape ``(elements, layers)``
        deg_seed : numpy.ndarray
            see :meth:`_voltage_tangent`

        Returns
        -------
        tuple of numpy.ndarray
            ``(S, dS)``, see :meth:`_voltage_tangent`
        """
        boundary = copy_boundary(boundary)
        if mode == "voltage":
            return self._voltage_tangent(boundary, pol_deg, ohm_deg, deg_seed)
        if mode != "current":
            raise ValueError("Undefined mode of operation")
        solution = self.solve_for_current(boundary, pol_deg=pol_deg, ohm_deg=ohm_deg)
        solution, d_solution = self._voltage_tangent(boundary, pol_deg, ohm_deg, deg_seed, solution)
        return solution, self._current_tangent(d_solution, boundary)

    def _solve_gradient(self, mode: str, boundary: BoundaryData, pol_deg, ohm_deg, variable: str):
        """
        Gradient of a steady solve w.r.t. a boundary attribute or a degradation matrix.

        Parameters
        ----------
        mode : str
            "voltage" (:meth:`solve_for_voltage`) or "current" (:meth:`solve_for_current`)
        boundary : BoundaryData
            initial value conditions; not modified
        pol_deg : numpy.ndarray or None
            polarization material variables, indexed by element and then by layer
        ohm_deg : numpy.ndarray or None
            ohmic material variables, indexed by element and then by layer
        variable : str
            boundary attribute, ``"pol_deg"`` or ``"ohm_deg"``

        Returns
        -------
        numpy.ndarray
            full Jacobian of shape ``S.shape + x.shape``; a degradation matrix
            given as None is differentiated at the initial material values
        """
        pol_deg, ohm_deg = self._deg_matrices(pol_deg, ohm_deg)
        n_elements, n_layers = pol_deg.shape
        deg_seed = np.zeros((n_elements, 2, n_layers, 0))
        if variable in ("pol_deg", "ohm_deg"):
            deg_seed = np.zeros((n_elements, 2, n_layers, n_elements * n_layers))
            k = 0 if variable == "pol_deg" else 1
            for e in range(n_elements):
                for layer in range(n_layers):
                    deg_seed[e, k, layer, e * n_layers + layer] = 1.0
        solution, d_solution = self._solve_tangent(boundary, mode, pol_deg, ohm_deg, deg_seed)
        n_theta = theta_slices(boundary)[THETA[-1]].stop
        if variable in ("pol_deg", "ohm_deg"):
            return d_solution[..., n_theta:].reshape(solution.shape + pol_deg.shape)
        return split(d_solution[..., :n_theta], boundary)[variable]

    @differentiable
    def solve_for_voltage(self, boundary, pol_deg=None, ohm_deg=None):
        """
        Solves the cell at the boundary voltage, see :meth:`nest.cell.Cell.solve_for_voltage`.

        Parameters
        ----------
        boundary : BoundaryData
            initial value conditions
        pol_deg : numpy.ndarray, optional
            polarization material variables, indexed by element and then by layer;
            None means ``m0``
        ohm_deg : numpy.ndarray, optional
            ohmic material variables, indexed by element and then by layer;
            None means ``m0``

        Returns
        -------
        numpy.ndarray
            solution matrix (variables x elements): V, j, T, P, fuel and air
            molar flows, and layer overpotentials
        """
        return as_output(super().solve_for_voltage(boundary, **self._deg_kwargs(pol_deg, ohm_deg)))

    @solve_for_voltage.grad("V")
    def _solve_for_voltage_grad_V(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_V` :meth:`~nest.grad.cell.Cell.solve_for_voltage`

        Returns
        -------
        numpy.ndarray
            derivative per V, shape ``S.shape``
        """
        return self._solve_gradient("voltage", boundary, pol_deg, ohm_deg, "V")

    @solve_for_voltage.grad("j")
    def _solve_for_voltage_grad_j(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.cell.Cell.solve_for_voltage`

        Returns
        -------
        numpy.ndarray
            zeros of shape ``S.shape`` (``boundary.j`` is only the Newton guess)
        """
        return self._solve_gradient("voltage", boundary, pol_deg, ohm_deg, "j")

    @solve_for_voltage.grad("n_fuel")
    def _solve_for_voltage_grad_n_fuel(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_{n_{fuel}}` :meth:`~nest.grad.cell.Cell.solve_for_voltage`

        Returns
        -------
        numpy.ndarray
            derivative per mol/s, shape ``S.shape + n_fuel.shape``
        """
        return self._solve_gradient("voltage", boundary, pol_deg, ohm_deg, "n_fuel")

    @solve_for_voltage.grad("n_air")
    def _solve_for_voltage_grad_n_air(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_{n_{air}}` :meth:`~nest.grad.cell.Cell.solve_for_voltage`

        Returns
        -------
        numpy.ndarray
            derivative per mol/s, shape ``S.shape + n_air.shape``
        """
        return self._solve_gradient("voltage", boundary, pol_deg, ohm_deg, "n_air")

    @solve_for_voltage.grad("T")
    def _solve_for_voltage_grad_T(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.cell.Cell.solve_for_voltage`

        Returns
        -------
        numpy.ndarray
            derivative per K, shape ``S.shape``
        """
        return self._solve_gradient("voltage", boundary, pol_deg, ohm_deg, "T")

    @solve_for_voltage.grad("P")
    def _solve_for_voltage_grad_P(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.cell.Cell.solve_for_voltage`

        Returns
        -------
        numpy.ndarray
            derivative per Pa, shape ``S.shape``
        """
        return self._solve_gradient("voltage", boundary, pol_deg, ohm_deg, "P")

    @solve_for_voltage.grad("pol_deg")
    def _solve_for_voltage_grad_pol_deg(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_{\pi}` :meth:`~nest.grad.cell.Cell.solve_for_voltage`

        Returns
        -------
        numpy.ndarray
            derivative per unit material variable, shape ``S.shape + (elements, layers)``
        """
        return self._solve_gradient("voltage", boundary, pol_deg, ohm_deg, "pol_deg")

    @solve_for_voltage.grad("ohm_deg")
    def _solve_for_voltage_grad_ohm_deg(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_{\omega}` :meth:`~nest.grad.cell.Cell.solve_for_voltage`

        Returns
        -------
        numpy.ndarray
            derivative per unit material variable, shape ``S.shape + (elements, layers)``
        """
        return self._solve_gradient("voltage", boundary, pol_deg, ohm_deg, "ohm_deg")

    @differentiable
    def solve_for_current(self, boundary, pol_deg=None, ohm_deg=None):
        """
        Solves the cell at the voltage matching the boundary current density, see :meth:`nest.cell.Cell.solve_for_current`.

        Uses the secant method with the tolerance :attr:`voltage_tol`. Sets
        ``boundary.V`` to the solved voltage.

        Parameters
        ----------
        boundary : BoundaryData
            initial value conditions
        pol_deg : numpy.ndarray, optional
            polarization material variables, indexed by element and then by layer;
            None means ``m0``
        ohm_deg : numpy.ndarray, optional
            ohmic material variables, indexed by element and then by layer;
            None means ``m0``

        Returns
        -------
        numpy.ndarray
            solution matrix (variables x elements)
        """
        kwargs = self._deg_kwargs(pol_deg, ohm_deg)

        def find_voltage(V):
            boundary.V = V
            solution = self.solve_for_voltage(boundary, **kwargs)
            return boundary.j - np.sum(solution[1]) / self.elements

        # Secant method: as fast as the analytical slope, which needs a full tangent
        boundary.V = newton(
            find_voltage, boundary.V, tol=self.voltage_tol, maxiter=self.newton_maxiter
        )
        return as_output(self.solve_for_voltage(boundary, **kwargs))

    @solve_for_current.grad("V")
    def _solve_for_current_grad_V(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_V` :meth:`~nest.grad.cell.Cell.solve_for_current`

        Returns
        -------
        numpy.ndarray
            zeros of shape ``S.shape`` (``boundary.V`` is only the Newton guess)
        """
        return self._solve_gradient("current", boundary, pol_deg, ohm_deg, "V")

    @solve_for_current.grad("j")
    def _solve_for_current_grad_j(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.cell.Cell.solve_for_current`

        Returns
        -------
        numpy.ndarray
            derivative per A/m^2, shape ``S.shape``
        """
        return self._solve_gradient("current", boundary, pol_deg, ohm_deg, "j")

    @solve_for_current.grad("n_fuel")
    def _solve_for_current_grad_n_fuel(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_{n_{fuel}}` :meth:`~nest.grad.cell.Cell.solve_for_current`

        Returns
        -------
        numpy.ndarray
            derivative per mol/s, shape ``S.shape + n_fuel.shape``
        """
        return self._solve_gradient("current", boundary, pol_deg, ohm_deg, "n_fuel")

    @solve_for_current.grad("n_air")
    def _solve_for_current_grad_n_air(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_{n_{air}}` :meth:`~nest.grad.cell.Cell.solve_for_current`

        Returns
        -------
        numpy.ndarray
            derivative per mol/s, shape ``S.shape + n_air.shape``
        """
        return self._solve_gradient("current", boundary, pol_deg, ohm_deg, "n_air")

    @solve_for_current.grad("T")
    def _solve_for_current_grad_T(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.cell.Cell.solve_for_current`

        Returns
        -------
        numpy.ndarray
            derivative per K, shape ``S.shape``
        """
        return self._solve_gradient("current", boundary, pol_deg, ohm_deg, "T")

    @solve_for_current.grad("P")
    def _solve_for_current_grad_P(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.cell.Cell.solve_for_current`

        Returns
        -------
        numpy.ndarray
            derivative per Pa, shape ``S.shape``
        """
        return self._solve_gradient("current", boundary, pol_deg, ohm_deg, "P")

    @solve_for_current.grad("pol_deg")
    def _solve_for_current_grad_pol_deg(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_{\pi}` :meth:`~nest.grad.cell.Cell.solve_for_current`

        Returns
        -------
        numpy.ndarray
            derivative per unit material variable, shape ``S.shape + (elements, layers)``
        """
        return self._solve_gradient("current", boundary, pol_deg, ohm_deg, "pol_deg")

    @solve_for_current.grad("ohm_deg")
    def _solve_for_current_grad_ohm_deg(self, boundary, pol_deg=None, ohm_deg=None):
        r"""
        :math:`\nabla_{\omega}` :meth:`~nest.grad.cell.Cell.solve_for_current`

        Returns
        -------
        numpy.ndarray
            derivative per unit material variable, shape ``S.shape + (elements, layers)``
        """
        return self._solve_gradient("current", boundary, pol_deg, ohm_deg, "ohm_deg")

    def _deg_slots(self) -> tuple:
        """
        Where each degradation variable of ``y`` enters the cell, as in :meth:`nest.cell.Cell.solve_time_step`.

        Returns
        -------
        tuple of tuple
            ``(index, name, element, layer)``: ``y[index]`` is the ``name``
            (``"pol_deg"`` or ``"ohm_deg"``) material variable of ``layer``
            (degradation-vector order) in ``element``
        """
        n_pol_deg = self.count_pol_deg()
        n_ohm_deg = self.count_ohm_deg()
        slots = []
        for i in range(self.elements):
            aux = 0
            if self.electrode_fuel.degradation.pol_active:
                slots.append((i * n_pol_deg, "pol_deg", i, 0))
                aux += 1
            if self.electrode_air.degradation.pol_active:
                slots.append((i * n_pol_deg + aux, "pol_deg", i, 1))
            aux = 0
            for k, layer in enumerate(self.electrolyte):
                if layer.degradation.ohm_active:
                    index = aux + i * n_ohm_deg + n_pol_deg * self.elements
                    slots.append((index, "ohm_deg", i, k + 2))
                    aux += 1
        return tuple(slots)

    def _deg_layout(self, y: np.ndarray) -> tuple:
        """
        Degradation keywords of the cell solves from the ODE state, see :meth:`_deg_slots`.

        Parameters
        ----------
        y : numpy.ndarray
            material variables (and power) of the degradation ODE

        Returns
        -------
        tuple of numpy.ndarray
            ``(pol_deg, ohm_deg)``, each of shape ``(elements, layers)``;
            entries without an active model are 0
        """
        n_layers = 2 + len(self.electrolyte)
        layout = {
            "pol_deg": np.zeros((self.elements, n_layers)),
            "ohm_deg": np.zeros((self.elements, n_layers)),
        }
        for index, name, element, layer in self._deg_slots():
            layout[name][element][layer] = y[index]
        return layout["pol_deg"], layout["ohm_deg"]

    def solve_time_step(self, y: np.ndarray, boundary: BoundaryData, mode: str) -> np.ndarray:
        """
        Steady-state solution matrix for the material state ``y``, see :meth:`nest.cell.Cell.solve_time_step`.

        Parameters
        ----------
        y : numpy.ndarray
            material variables of the degradation ODE
        boundary : BoundaryData
            state variable conditions of the cell; ``j`` (voltage mode) or ``V``
            (current mode) is set to the solved mean value
        mode : str
            operational mode: "current" or "voltage" constant

        Returns
        -------
        numpy.ndarray
            steady-state solution matrix (variables x elements)
        """
        pol_deg, ohm_deg = self._deg_layout(y)
        if mode == "voltage":
            steady_solution = self.solve_for_voltage(boundary, pol_deg=pol_deg, ohm_deg=ohm_deg)
            boundary.j = np.sum(steady_solution[1]) / self.elements
        elif mode == "current":
            steady_solution = self.solve_for_current(boundary, pol_deg=pol_deg, ohm_deg=ohm_deg)
            boundary.V = np.sum(steady_solution[0]) / self.elements
        else:
            raise ValueError("Undefined mode of operation")
        return steady_solution

    def _steady_tangent(self, y: np.ndarray, boundary: BoundaryData, mode: str) -> tuple:
        """
        Steady solution matrix of :meth:`solve_time_step` and its forward tangents.

        Parameters
        ----------
        y : numpy.ndarray
            material variables (and power) of the degradation ODE
        boundary : BoundaryData
            state variable conditions of the cell; not modified
        mode : str
            operational mode: "current" or "voltage" constant

        Returns
        -------
        tuple of numpy.ndarray
            ``(S, dS)``: solution matrix ``(n_var, elements)`` and tangents
            ``(n_var, elements, n_theta + len(y))``. Columns are the boundary
            attributes in :data:`nest.grad.boundary.THETA` flat order, then
            ``y`` (the power column is 0).
        """
        pol_deg, ohm_deg = self._deg_layout(y)
        deg_seed = np.zeros(pol_deg.shape[:1] + (2,) + pol_deg.shape[1:] + (len(y),))
        for index, name, element, layer in self._deg_slots():
            deg_seed[element, 0 if name == "pol_deg" else 1, layer, index] = 1.0
        return self._solve_tangent(boundary, mode, pol_deg, ohm_deg, deg_seed)

    def _rhs_jacobians(self, t, y: np.ndarray, boundary: BoundaryData, mode: str) -> tuple:
        """
        Degradation rates of :meth:`advance_time_step` and their Jacobians.

        Each rate is chained through the steady solution: the layer state is
        rebuilt as in :meth:`nest.cell.Cell.advance_time_step`, with the
        reaction-site pressures of the outlet state.

        Parameters
        ----------
        t : float
            time [h]
        y : numpy.ndarray
            material variables (and power) of the degradation ODE
        boundary : BoundaryData
            state variable conditions of the cell; not modified
        mode : str
            operational mode: "current" or "voltage" constant

        Returns
        -------
        tuple of numpy.ndarray
            ``(f, J_y, J_theta)``: rates ``(n_y,)``, Jacobian w.r.t. ``y``
            ``(n_y, n_y)`` (power column 0) and w.r.t. the boundary attributes
            ``(n_y, n_theta)`` in :data:`nest.grad.boundary.THETA` flat order
        """
        solution, d_solution = self._steady_tangent(y, boundary, mode)
        slices = theta_slices(boundary)
        n_theta = slices[THETA[-1]].stop
        n_fuel, n_air = len(boundary.n_fuel), len(boundary.n_air)
        aux = 4 + n_fuel + n_air
        fuel, air = self.electrode_fuel, self.electrode_air
        f = np.zeros(len(y))
        J = np.zeros((len(y), n_theta + len(y)))

        # Layer states per element: reaction-site pressures and their tangents
        states = []
        for e in range(self.elements):
            V, j, T, P = solution[0][e], solution[1][e], solution[2][e], solution[3][e]
            dj, dT, dP = d_solution[1, e], d_solution[2, e], d_solution[3, e]
            outlet = BoundaryData(
                V, j, solution[4 : 4 + n_fuel, e], solution[4 + n_fuel : aux, e], T, P,
                fuel_pressure=boundary.fuel_pressure, air_pressure=boundary.air_pressure,
            )
            tpb = []
            channels = (
                (fuel, self.Ps_fuel(outlet, j), self._Ps_fuel_partials(outlet, j), "n_fuel",
                 d_solution[4 : 4 + n_fuel, e]),
                (air, self.Ps_air(outlet, j), self._Ps_air_partials(outlet, j), "n_air",
                 d_solution[4 + n_fuel : aux, e]),
            )
            for layer, Ps, partials, name, d_n in channels:
                d_Ps = (
                    np.outer(partials["j"], dj)
                    + partials[name] @ d_n
                    + np.outer(partials["T"], dT)
                    + np.outer(partials["P"], dP)
                )
                d_tpb = (
                    np.outer(layer.Ps_star.grad_j(j, T, Ps), dj)
                    + np.outer(layer.Ps_star.grad_T(j, T, Ps), dT)
                    + np.asarray(layer.Ps_star.grad_Ps(j, T, Ps)) @ d_Ps
                )
                tpb.append((layer.Ps_star(j, T, Ps), d_tpb))
            state = BoundaryData(
                V, j, tpb[0][0], tpb[1][0], T, P,
                fuel_pressure=boundary.fuel_pressure, air_pressure=boundary.air_pressure,
            )
            states.append((state, tpb[0][1]))

        # Rates: the layer overpotential is the state voltage, as in the base model
        layers = self._layers()
        for index, name, element, k in self._deg_slots():
            state, d_Ps_tpb = states[element]
            state.V = solution[aux + k][element]
            eta_fuel = solution[aux][element] if name == "ohm_deg" else None
            kwargs = {} if eta_fuel is None else {"eta_fuel": eta_fuel}
            model = layers[k].degradation
            f[index] = model.material_dt(y[index], state, t, **kwargs)
            partials = model.material_dt_partials(y[index], state, t, eta_fuel=eta_fuel)
            J[index] = (
                partials["T"] * d_solution[2, element]
                + partials["P"] * d_solution[3, element]
                + partials["j"] * d_solution[1, element]
                + partials["V"] * d_solution[aux + k, element]
                + partials["eta_fuel"] * d_solution[aux, element]
                + np.asarray(partials["n_fuel"]) @ d_Ps_tpb
            )
            J[index, n_theta + index] += partials["m"]

        # Power density: mean current at the fixed voltage, or the reverse
        if mode == "voltage":
            j_mean = np.sum(solution[1]) / self.elements
            f[-1] = j_mean * boundary.V
            J[-1] = boundary.V * np.sum(d_solution[1], axis=0) / self.elements
            J[-1, slices["V"].start] += j_mean
        else:
            V_mean = np.sum(solution[0]) / self.elements
            f[-1] = boundary.j * V_mean
            J[-1] = boundary.j * np.sum(d_solution[0], axis=0) / self.elements
            J[-1, slices["j"].start] += V_mean
        return f, J[:, n_theta:], J[:, :n_theta]

    def _advance_time_step_theta(self, t, y, boundary, mode, name: str):
        """
        Gradient of :meth:`advance_time_step` w.r.t. one boundary attribute.

        Parameters
        ----------
        t : float
            time [h]
        y : numpy.ndarray
            material variables (and power) of the degradation ODE
        boundary : BoundaryData
            state variable conditions of the cell; not modified
        mode : str
            operational mode: "current" or "voltage" constant
        name : str
            attribute in :data:`nest.grad.boundary.THETA`

        Returns
        -------
        numpy.ndarray
            Jacobian of shape ``(len(y),) + x.shape``
        """
        return split(self._rhs_jacobians(t, y, boundary, mode)[2], boundary)[name]

    @differentiable
    def advance_time_step(self, t, y, boundary, mode):
        """
        Material change rates from degradation, see :meth:`nest.cell.Cell.advance_time_step`.

        Parameters
        ----------
        t : float
            time [h]
        y : numpy.ndarray
            material variables (pol_deg and ohm_deg) and power of the degradation ODE
        boundary : BoundaryData
            state variables of the cell; ``j`` (voltage mode) or ``V`` (current
            mode) is set to the solved mean value
        mode : str
            operational mode: "current" or "voltage" constant

        Returns
        -------
        numpy.ndarray
            rates of change of pol_deg, ohm_deg and power density [W/m^2]
        """
        return as_output(super().advance_time_step(t, y, boundary, mode))

    @advance_time_step.grad("y")
    def _advance_time_step_grad_y(self, t, y, boundary, mode):
        r"""
        :math:`\nabla_y` :meth:`~nest.grad.cell.Cell.advance_time_step`

        Returns
        -------
        numpy.ndarray
            Jacobian of shape ``(len(y), len(y))``; the power column is 0
        """
        return self._rhs_jacobians(t, y, boundary, mode)[1]

    @advance_time_step.grad("V")
    def _advance_time_step_grad_V(self, t, y, boundary, mode):
        r"""
        :math:`\nabla_V` :meth:`~nest.grad.cell.Cell.advance_time_step`

        Returns
        -------
        numpy.ndarray
            derivative per V, shape ``(len(y),)``; 0 in current mode (Newton guess)
        """
        return self._advance_time_step_theta(t, y, boundary, mode, "V")

    @advance_time_step.grad("j")
    def _advance_time_step_grad_j(self, t, y, boundary, mode):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.cell.Cell.advance_time_step`

        Returns
        -------
        numpy.ndarray
            derivative per A/m^2, shape ``(len(y),)``; 0 in voltage mode (Newton guess)
        """
        return self._advance_time_step_theta(t, y, boundary, mode, "j")

    @advance_time_step.grad("n_fuel")
    def _advance_time_step_grad_n_fuel(self, t, y, boundary, mode):
        r"""
        :math:`\nabla_{n_{fuel}}` :meth:`~nest.grad.cell.Cell.advance_time_step`

        Returns
        -------
        numpy.ndarray
            derivative per mol/s, shape ``(len(y),) + n_fuel.shape``
        """
        return self._advance_time_step_theta(t, y, boundary, mode, "n_fuel")

    @advance_time_step.grad("n_air")
    def _advance_time_step_grad_n_air(self, t, y, boundary, mode):
        r"""
        :math:`\nabla_{n_{air}}` :meth:`~nest.grad.cell.Cell.advance_time_step`

        Returns
        -------
        numpy.ndarray
            derivative per mol/s, shape ``(len(y),) + n_air.shape``
        """
        return self._advance_time_step_theta(t, y, boundary, mode, "n_air")

    @advance_time_step.grad("T")
    def _advance_time_step_grad_T(self, t, y, boundary, mode):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.cell.Cell.advance_time_step`

        Returns
        -------
        numpy.ndarray
            derivative per K, shape ``(len(y),)``
        """
        return self._advance_time_step_theta(t, y, boundary, mode, "T")

    @advance_time_step.grad("P")
    def _advance_time_step_grad_P(self, t, y, boundary, mode):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.cell.Cell.advance_time_step`

        Returns
        -------
        numpy.ndarray
            derivative per Pa, shape ``(len(y),)``
        """
        return self._advance_time_step_theta(t, y, boundary, mode, "P")

    @advance_time_step.grad("fuel_pressure")
    def _advance_time_step_grad_fuel_pressure(self, t, y, boundary, mode):
        r"""
        :math:`\nabla_{P_{fuel}}` :meth:`~nest.grad.cell.Cell.advance_time_step`

        Returns
        -------
        numpy.ndarray
            zeros of shape ``(len(y),)`` (the fuel-side pressure is not used)
        """
        return self._advance_time_step_theta(t, y, boundary, mode, "fuel_pressure")

    @advance_time_step.grad("air_pressure")
    def _advance_time_step_grad_air_pressure(self, t, y, boundary, mode):
        r"""
        :math:`\nabla_{P_{air}}` :meth:`~nest.grad.cell.Cell.advance_time_step`

        Returns
        -------
        numpy.ndarray
            zeros of shape ``(len(y),)`` (the air-side pressure is not used)
        """
        return self._advance_time_step_theta(t, y, boundary, mode, "air_pressure")

    def _initial_state(self) -> np.ndarray:
        """
        Initial state of the degradation ODE, as in :meth:`nest.cell.Cell.solve_for_time`.

        The ohmic entries are set like the base model, which only fills the
        first element (harmless while every ohmic model starts at 0).

        Returns
        -------
        numpy.ndarray
            initial material variables and power
        """
        n_pol_deg = self.count_pol_deg()
        n_ohm_deg = self.count_ohm_deg()
        pol_deg_0 = np.zeros(self.elements * n_pol_deg)
        ohm_deg_0 = np.zeros(self.elements * n_ohm_deg)
        for i in range(self.elements):
            count = 0
            if self.electrode_fuel.degradation.pol_active:
                pol_deg_0[i * n_pol_deg] = self.electrode_fuel.degradation.m0
                count += 1
            if self.electrode_air.degradation.pol_active:
                pol_deg_0[count + i * n_pol_deg] = self.electrode_air.degradation.m0
            k = 0
            for layer in self.electrolyte:
                if layer.degradation.ohm_active:
                    ohm_deg_0[k] = layer.degradation.m0  # base indexing
                    k += 1
        return np.concatenate((pol_deg_0, ohm_deg_0, np.zeros(1)))

    @differentiable
    def solve_for_time(self, boundary, mode, t_max, events=None, **ivp_options):
        """
        Solves the degradation ODE, see :meth:`nest.cell.Cell.solve_for_time`.

        Without ``ivp_options`` this is the base solve (``RK23`` with the
        ``solve_ivp`` default tolerances); options given are passed to
        ``solve_ivp`` on top of ``method="RK23"``. The gradients are those of
        the objective of :meth:`solve_for_time_gradients`; they integrate with
        :attr:`ivp_options` unless options are given, so pass the same options
        here for values consistent with them.

        Parameters
        ----------
        boundary : BoundaryData
            state variables of the cell; mutated like the base model
        mode : str
            operational mode: "current" or "voltage" constant
        t_max : float
            final time [h]
        events : callable or list of callable, optional
            ``solve_ivp`` events, called as ``event(t, y, boundary, mode)``
        **ivp_options
            options of ``scipy.integrate.solve_ivp``

        Returns
        -------
        scipy.integrate.OdeResult
            solution object from ``solve_ivp``
        """
        if not ivp_options:
            return super().solve_for_time(boundary, mode, t_max, events)
        return solve_ivp(
            self.advance_time_step,
            (0, t_max),
            self._initial_state(),
            args=(boundary, mode),
            events=events,
            **{"method": "RK23", **ivp_options},
        )

    def _check_gradients(self):
        """
        Raise if a layer has no analytical gradients.

        Raises
        ------
        NotImplementedError
            for Stefan-Maxwell transport
        """
        if any(isinstance(layer.transport, StefanMaxwell) for layer in self._layers()):
            raise NotImplementedError("StefanMaxwell transport has no analytical gradients")

    def _ivp_settings(self, boundary: BoundaryData, t_max, ivp_options: dict) -> tuple:
        """
        ``solve_ivp`` options of the gradient computations and the state scales.

        Parameters
        ----------
        boundary : BoundaryData
            initial value conditions
        t_max : float
            final time [h]
        ivp_options : dict
            options overriding :attr:`ivp_options`

        Returns
        -------
        tuple
            ``(options, scale)``: options (``atol = rtol * scale`` unless given)
            and the scale of each state entry: the degradation model's
            ``m_scale``, and ``|j V| t_max`` for the power
        """
        options = {**self.ivp_options, **ivp_options}
        for key in ("dense_output", "t_eval", "events", "args", "vectorized"):
            options.pop(key, None)
        scale = np.ones(len(self._initial_state()))
        layers = self._layers()
        for index, _, _, k in self._deg_slots():
            scale[index] = layers[k].degradation.m_scale
        scale[-1] = abs(boundary.j * boundary.V) * t_max or 1.0
        if "atol" not in ivp_options:
            options["atol"] = options.get("rtol", 1e-3) * scale
        return options, scale

    def _internal_events(self, y: np.ndarray) -> tuple:
        """
        Ni radii below and above ``r_ed_max``, where the agglomeration rate drops to 0.

        Parameters
        ----------
        y : numpy.ndarray
            state of the degradation ODE

        Returns
        -------
        tuple
            ``(pending, crossed)``: ``(index, limit, event)`` for each radius
            still below its limit [μm], with a terminal ``solve_ivp`` event for
            the upward crossing, and the state indices of the radii above it
        """
        layers = self._layers()
        pending, crossed = [], []
        for index, _, _, k in self._deg_slots():
            model = layers[k].degradation
            if not isinstance(model, NickelAgglomeration):
                continue
            limit = model.r_ed_max * 1e6
            if y[index] > limit:
                crossed.append(index)
                continue

            def event(t, y, *args, index=index, limit=limit):
                return y[index] - limit

            event.terminal, event.direction = True, 1
            pending.append((index, limit, event))
        return pending, tuple(crossed)

    @contextmanager
    def _smooth_agglomeration(self):
        """
        Context without the ``r_ed_max`` cut-off of the Ni agglomeration rates.

        Within a segment the rates are continued smoothly past the limit, so a
        step straddling the crossing does not spoil its location; the radii
        that already crossed are set to 0 by the caller.
        """
        models = [
            layer.degradation for layer in self._layers()
            if isinstance(layer.degradation, NickelAgglomeration)
        ]
        limits = [model.r_ed_max for model in models]
        try:
            for model in models:
                model.r_ed_max = np.inf
            yield
        finally:
            for model, limit in zip(models, limits):
                model.r_ed_max = limit

    def _segment_rhs(self, t, y, work: BoundaryData, mode: str, crossed: tuple) -> np.ndarray:
        """
        Degradation rates within a segment, see :meth:`_smooth_agglomeration`.

        Parameters
        ----------
        t : float
            time [h]
        y : numpy.ndarray
            state of the degradation ODE
        work : BoundaryData
            boundary copy; its Newton guess is updated like the base model
        mode : str
            operational mode: "current" or "voltage" constant
        crossed : tuple of int
            state indices of the Ni radii above ``r_ed_max``

        Returns
        -------
        numpy.ndarray
            rates of :meth:`advance_time_step`
        """
        with self._smooth_agglomeration():
            f = np.array(self.advance_time_step(t, y, work, mode))
        f[list(crossed)] = 0.0
        return f

    def _segment_jacobians(self, t, y, work: BoundaryData, mode: str, crossed: tuple) -> tuple:
        """
        Rates and Jacobians within a segment, see :meth:`_smooth_agglomeration` and :meth:`_rhs_jacobians`.

        Parameters
        ----------
        t : float
            time [h]
        y : numpy.ndarray
            state of the degradation ODE
        work : BoundaryData
            boundary copy; its Newton guess is set to the solved value
        mode : str
            operational mode: "current" or "voltage" constant
        crossed : tuple of int
            state indices of the Ni radii above ``r_ed_max``

        Returns
        -------
        tuple of numpy.ndarray
            ``(f, J_y, J_theta)``; the rows of the crossed radii are 0
        """
        with self._smooth_agglomeration():
            f, J_y, J_theta = self._rhs_jacobians(t, y, work, mode)
        self._warm_start(work, mode, f)
        for index in crossed:
            f[index], J_y[index], J_theta[index] = 0.0, 0.0, 0.0
        return f, J_y, J_theta

    def _forward(self, boundary: BoundaryData, mode: str, t_max, events, options: dict) -> _Forward:
        """
        Forward degradation solve with dense output, restarted at every ``r_ed_max`` crossing.

        The values are those of :meth:`advance_time_step`; within a segment the
        rates are evaluated with :meth:`_segment_rhs` so the crossings are
        located accurately.

        Parameters
        ----------
        boundary : BoundaryData
            initial value conditions; not modified
        mode : str
            operational mode: "current" or "voltage" constant
        t_max : float
            final time [h]
        events : callable or list of callable or None
            user events
        options : dict
            ``solve_ivp`` options

        Returns
        -------
        _Forward
            segments, final time and state, and the terminal user event
        """
        work = copy_boundary(boundary)
        user = [] if events is None else ([events] if callable(events) else list(events))
        y = self._initial_state()
        t0, start_slot, segments, terminal = 0.0, None, [], None
        while True:
            pending, crossed = self._internal_events(y)
            all_events = [_user_event(event) for event in user] + [event for _, _, event in pending]
            sol = solve_ivp(
                self._segment_rhs, (t0, t_max), y, args=(work, mode, crossed),
                events=all_events or None, dense_output=True, **options,
            )
            if sol.status == -1:
                raise RuntimeError(f"degradation solve failed: {sol.message}")
            segments.append(_Segment(t0, sol.t[-1], sol.sol, sol.t, sol.y, start_slot, crossed))
            y = sol.y[:, -1].copy()
            fired = _fired(sol, all_events)
            if fired is None:
                break
            if fired < len(user):
                terminal = fired
                break
            # The rate is 0 above the limit: restart just above it
            start_slot, limit, _ = pending[fired - len(user)]
            y[start_slot] = np.nextafter(limit, np.inf)
            t0 = sol.t[-1]
        return _Forward(segments, segments[-1].t1, y, terminal, user, work)

    def _warn_threshold_flips(self, forward: _Forward, boundary: BoundaryData, mode: str):
        """
        Warn if a Ni migration threshold switches during the forward solve (not differentiated).

        Parameters
        ----------
        forward : _Forward
            forward solve
        boundary : BoundaryData
            initial value conditions; not modified
        mode : str
            operational mode: "current" or "voltage" constant
        """
        layers = self._layers()
        switching = [
            (element, k)
            for _, _, element, k in self._deg_slots()
            if isinstance(layers[k].degradation, NickelMigration)
            and layers[k].degradation.threshold is not None
            and layers[k].degradation.rate_after_threshold is not None
        ]
        if not switching:
            return
        aux = 4 + len(boundary.n_fuel) + len(boundary.n_air)
        work = copy_boundary(boundary)
        previous = None
        for segment in forward.segments:
            for t, y in zip(segment.t, segment.y.T):
                S = self.solve_time_step(y, work, mode)
                flags = tuple(
                    layers[k].degradation.threshold_reached(S[2][e], S[1][e], S[aux][e])
                    for e, k in switching
                )
                if previous is not None and flags != previous:
                    warnings.warn(
                        f"NickelMigration threshold switched near t = {t:g} h; "
                        "the switch is not differentiated",
                        RuntimeWarning,
                        stacklevel=4,
                    )
                    return
                previous = flags

    @staticmethod
    def _warm_start(work: BoundaryData, mode: str, f: np.ndarray):
        """
        Set the Newton guess of ``work`` from the power rate, as :meth:`solve_time_step` does.

        Parameters
        ----------
        work : BoundaryData
            boundary copy reused by successive steady solves
        mode : str
            operational mode: "current" or "voltage" constant
        f : numpy.ndarray
            degradation rates; the last one is the power density
        """
        if mode == "voltage" and work.V != 0:
            work.j = f[-1] / work.V
        elif mode == "current" and work.j != 0:
            work.V = f[-1] / work.j

    def _objective_gradient(self, objective, objective_grad, y: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        Gradient of the objective w.r.t. the final state.

        Parameters
        ----------
        objective : callable or None
            ``objective(y)``; None means the integrated power density ``y[-1]``
        objective_grad : callable or None
            ``objective_grad(y)``; central differences (with a warning) if None
        y : numpy.ndarray
            final state
        scale : numpy.ndarray
            state scales, used for the difference steps

        Returns
        -------
        numpy.ndarray
            gradient of shape ``y.shape``
        """
        if objective_grad is not None:
            return np.asarray(objective_grad(y), dtype=float).reshape(y.shape)
        if objective is None:
            grad = np.zeros(y.shape)
            grad[-1] = 1.0
            return grad
        warnings.warn(
            "no objective_grad given; using central differences of the objective",
            RuntimeWarning,
            stacklevel=4,
        )
        return central_difference(objective, y, 1e-6 * np.maximum(np.abs(y), scale), warn=False)

    def _event_transversality(self, forward: _Forward, boundary, mode, grad_g, scale, theta_ref) -> tuple:
        r"""
        Adjoint end conditions at a terminal user event.

        The event time moves with the parameters; the event derivatives are
        taken by central differences (with a warning).

        .. math:: \kappa = \nabla g \cdot f / (\psi_t + \psi_y \cdot f)

        Parameters
        ----------
        forward : _Forward
            forward solve stopped by the event
        boundary : BoundaryData
            initial value conditions; not modified
        mode : str
            operational mode: "current" or "voltage" constant
        grad_g : numpy.ndarray
            objective gradient at the final state
        scale : numpy.ndarray
            state scales, used for the difference steps
        theta_ref : numpy.ndarray
            parameter scales

        Returns
        -------
        tuple of numpy.ndarray
            ``(lambda, mu)``: adjoint state and scaled parameter adjoint at the event
        """
        warnings.warn(
            "terminal event: event derivatives by central differences",
            RuntimeWarning,
            stacklevel=4,
        )
        event = forward.events[forward.terminal]
        t, y, work = forward.t, forward.y, forward.work
        f = self._segment_rhs(t, y, copy_boundary(work), mode, forward.segments[-1].crossed)
        psi_t = central_difference(lambda s: event(float(s), y, work, mode), t, warn=False)
        psi_y = central_difference(
            lambda v: event(t, v, work, mode), y, 1e-6 * np.maximum(np.abs(y), scale), warn=False
        )
        psi_theta = central_difference(
            lambda v: event(t, y, unflatten(v, boundary), mode), flatten(boundary), 1e-6 * theta_ref, warn=False
        )
        kappa = (grad_g @ f) / (psi_t + psi_y @ f)
        return grad_g - kappa * psi_y, -kappa * psi_theta * theta_ref

    def _adjoint(self, boundary, mode, t_max, events, objective, objective_grad, ivp_options) -> np.ndarray:
        r"""
        Adjoint gradient of the objective w.r.t. the flat boundary parameters.

        The adjoint runs backward from the final time to ``adjoint_t_min * t_max``
        in the variable ``s = sqrt(t)``, which removes the ``t^(-1/2)``
        singularity of ``J_y`` from interconnect oxidation; the rest of the
        interval is added with the end values. The parameter adjoint is scaled
        by ``theta_ref = |theta|`` (1 where 0).

        .. math:: \lambda' = -J_y^T \lambda, \quad \mu' = -J_\theta^T \lambda

        Parameters
        ----------
        boundary, mode, t_max, events, objective, objective_grad, ivp_options
            see :meth:`solve_for_time_gradients`

        Returns
        -------
        numpy.ndarray
            gradient in :data:`nest.grad.boundary.THETA` flat order
        """
        self._check_gradients()
        if mode not in ("voltage", "current"):
            raise ValueError("Undefined mode of operation")
        options, scale = self._ivp_settings(boundary, t_max, ivp_options)
        forward = self._forward(boundary, mode, t_max, events, options)
        self._warn_threshold_flips(forward, boundary, mode)
        theta = flatten(boundary)
        theta_ref = np.where(theta != 0, np.abs(theta), 1.0)
        n_y = len(forward.y)

        # End conditions, with the transversality correction of a terminal event
        grad_g = self._objective_gradient(objective, objective_grad, forward.y, scale)
        lam, mu = grad_g, np.zeros(theta.size)
        if forward.terminal is not None:
            lam, mu = self._event_transversality(forward, boundary, mode, grad_g, scale, theta_ref)
        g_scale = max(np.abs(grad_g) @ scale, np.abs(lam) @ scale, np.finfo(float).tiny)
        rtol = options.get("rtol", 1e-3)
        backward = {
            "method": options.get("method", "RK45"),
            "rtol": rtol,
            "atol": rtol * g_scale * np.concatenate((1 / scale, np.ones(theta.size))),
        }
        if "max_step" in options:
            backward["max_step"] = options["max_step"]

        # Jacobians along the forward solution, cached per segment and time
        work = copy_boundary(boundary)
        cache = {}

        def jacobians(i, t):
            if (i, t) not in cache:
                segment = forward.segments[i]
                _, J_y, J_theta = self._segment_jacobians(t, segment.sol(t), work, mode, segment.crossed)
                cache[(i, t)] = (J_y, J_theta * theta_ref)
            return cache[(i, t)]

        t_eps = min(self.adjoint_t_min * t_max, 0.5 * forward.t)
        a = np.concatenate((lam, mu))
        first = 0
        for i in reversed(range(len(forward.segments))):
            segment = forward.segments[i]
            t_low = max(segment.t0, t_eps)
            if segment.t1 > t_low:
                # In s = sqrt(t) the t^(-1/2) singularity of J_y at t = 0 cancels

                def rhs(s, a, i=i):
                    J_y, J_theta = jacobians(i, s**2)
                    return -2 * s * np.concatenate((J_y.T @ a[:n_y], J_theta.T @ a[:n_y]))

                sol = solve_ivp(rhs, (np.sqrt(segment.t1), np.sqrt(t_low)), a, **backward)
                if sol.status == -1:
                    raise RuntimeError(f"adjoint solve failed: {sol.message}")
                a = sol.y[:, -1].copy()
            if segment.t0 <= t_eps:
                first = i
                break
            if segment.start_slot is not None:
                # The agglomeration rate drops to 0 at r_ed_max: the radius adjoint jumps to 0
                a[segment.start_slot] = 0.0

        # Remaining interval (0, t_eps) with the end values
        _, J_theta = jacobians(first, t_eps)
        a[n_y:] += J_theta.T @ a[:n_y] * t_eps
        return a[n_y:] / theta_ref

    def solve_for_time_gradients(
        self, boundary, mode, t_max, events=None, objective=None, objective_grad=None, **ivp_options
    ) -> dict:
        r"""
        Gradient of an objective of the final degradation state w.r.t. the boundary attributes.

        The objective ``g(y(t_f))`` is evaluated at ``t_max`` or at a terminal
        event. The gradient comes from the adjoint method: a forward solve with
        dense output, then a backward solve of the adjoint with the Jacobians of
        :meth:`advance_time_step`. A Ni radius reaching ``r_ed_max`` is handled
        as a jump of its adjoint to 0; a terminal event adds the event-time
        correction, with the event derivatives by central differences (with a
        warning). A Ni migration threshold switch is not differentiated (a
        warning is emitted). The last result is cached, so the ``grad_<var>``
        methods of :meth:`solve_for_time` share one computation.

        .. math:: \nabla_\theta g = \int_0^{t_f} J_\theta^T \lambda \, dt

        Parameters
        ----------
        boundary : BoundaryData
            state variables of the cell; not modified
        mode : str
            operational mode: "current" or "voltage" constant
        t_max : float
            final time [h]
        events : callable or list of callable, optional
            ``solve_ivp`` events, called as ``event(t, y, boundary, mode)``
        objective : callable, optional
            ``objective(y)`` of the final state; None means the integrated
            power density ``y[-1]``
        objective_grad : callable, optional
            ``objective_grad(y)``, gradient of ``objective``; central
            differences (with a warning) if None
        **ivp_options
            ``solve_ivp`` options overriding :attr:`ivp_options`; without
            ``atol``, it is ``rtol`` times the degradation models' ``m_scale``
            (``|j V| t_max`` for the power)

        Returns
        -------
        dict
            ``{name: gradient}`` for the names of :data:`nest.grad.boundary.THETA`;
            floats for scalar attributes, arrays for the flows. The Newton
            guess (``j`` in voltage mode, ``V`` in current mode),
            ``fuel_pressure`` and ``air_pressure`` have zero gradients.

        Raises
        ------
        NotImplementedError
            if a layer uses Stefan-Maxwell transport
        """
        key = (
            flatten(boundary).tobytes(), mode, float(t_max), id(events), id(objective),
            id(objective_grad), tuple(sorted((k, _key(v)) for k, v in ivp_options.items())),
        )
        cache = getattr(self, "_adjoint_cache", None)
        if cache is None or cache[0] != key:
            gradient = self._adjoint(boundary, mode, t_max, events, objective, objective_grad, ivp_options)
            # The callables are kept so that their ids stay unique
            self._adjoint_cache = (key, (events, objective, objective_grad), gradient)
        return split(self._adjoint_cache[2].copy(), boundary)

    def _forward_sensitivity(self, boundary: BoundaryData, mode: str, t_max, **ivp_options) -> np.ndarray:
        r"""
        Final-state sensitivities by forward integration (test oracle of the adjoint).

        Integrates the state and its scaled sensitivities from
        ``adjoint_t_min * t_max``, starting from the same end correction as the
        adjoint; a Ni radius reaching ``r_ed_max`` resets its row to 0.

        .. math:: S' = J_y S + J_\theta

        Parameters
        ----------
        boundary : BoundaryData
            initial value conditions; not modified
        mode : str
            operational mode: "current" or "voltage" constant
        t_max : float
            final time [h]
        **ivp_options
            ``solve_ivp`` options, see :meth:`solve_for_time_gradients`

        Returns
        -------
        numpy.ndarray
            ``dy(t_max)/dtheta`` of shape ``(len(y), n_theta)`` in
            :data:`nest.grad.boundary.THETA` flat order
        """
        self._check_gradients()
        options, scale = self._ivp_settings(boundary, t_max, ivp_options)
        theta = flatten(boundary)
        theta_ref = np.where(theta != 0, np.abs(theta), 1.0)
        n_y, n_theta = len(scale), theta.size
        work = copy_boundary(boundary)
        t_eps = self.adjoint_t_min * t_max
        y = self._initial_state()
        crossed = self._internal_events(y)[1]
        y = solve_ivp(self._segment_rhs, (0, t_eps), y, args=(work, mode, crossed), **options).y[:, -1]
        S = self._segment_jacobians(t_eps, y, work, mode, crossed)[2] * theta_ref * t_eps
        rtol = options.get("rtol", 1e-3)
        sensitivity = {
            "method": options.get("method", "RK45"),
            "rtol": rtol,
            "atol": rtol * np.concatenate((scale, np.repeat(scale, n_theta))),
        }

        def rhs(t, z, crossed):
            f, J_y, J_theta = self._segment_jacobians(t, z[:n_y], work, mode, crossed)
            S = z[n_y:].reshape(n_y, n_theta)
            return np.concatenate((f, (J_y @ S + J_theta * theta_ref).ravel()))

        t0 = t_eps
        while True:
            pending, crossed = self._internal_events(y)
            events = [event for _, _, event in pending]
            sol = solve_ivp(
                rhs, (t0, t_max), np.concatenate((y, S.ravel())), args=(crossed,),
                events=events or None, **sensitivity,
            )
            if sol.status == -1:
                raise RuntimeError(f"sensitivity solve failed: {sol.message}")
            y = sol.y[:n_y, -1].copy()
            S = sol.y[n_y:, -1].reshape(n_y, n_theta).copy()
            fired = _fired(sol, events)
            if fired is None:
                break
            index, limit, _ = pending[fired]
            y[index] = np.nextafter(limit, np.inf)
            S[index] = 0.0
            t0 = sol.t[-1]
        return S / theta_ref

    def _solve_for_time_gradient(self, name, boundary, mode, t_max, events, objective, objective_grad, ivp_options):
        """
        One entry of :meth:`solve_for_time_gradients`.

        Returns
        -------
        float or numpy.ndarray
            gradient w.r.t. the boundary attribute ``name``
        """
        return self.solve_for_time_gradients(
            boundary, mode, t_max, events, objective, objective_grad, **ivp_options
        )[name]

    @solve_for_time.grad("V")
    def _solve_for_time_grad_V(
        self, boundary, mode, t_max, events=None, objective=None, objective_grad=None, **ivp_options
    ):
        r"""
        :math:`\nabla_V` :meth:`~nest.grad.cell.Cell.solve_for_time`

        Returns
        -------
        float
            objective derivative per V, see :meth:`solve_for_time_gradients`;
            0 in current mode (Newton guess)
        """
        return self._solve_for_time_gradient(
            "V", boundary, mode, t_max, events, objective, objective_grad, ivp_options
        )

    @solve_for_time.grad("j")
    def _solve_for_time_grad_j(
        self, boundary, mode, t_max, events=None, objective=None, objective_grad=None, **ivp_options
    ):
        r"""
        :math:`\nabla_j` :meth:`~nest.grad.cell.Cell.solve_for_time`

        Returns
        -------
        float
            objective derivative per A/m^2, see :meth:`solve_for_time_gradients`;
            0 in voltage mode (Newton guess)
        """
        return self._solve_for_time_gradient(
            "j", boundary, mode, t_max, events, objective, objective_grad, ivp_options
        )

    @solve_for_time.grad("n_fuel")
    def _solve_for_time_grad_n_fuel(
        self, boundary, mode, t_max, events=None, objective=None, objective_grad=None, **ivp_options
    ):
        r"""
        :math:`\nabla_{n_{fuel}}` :meth:`~nest.grad.cell.Cell.solve_for_time`

        Returns
        -------
        numpy.ndarray
            objective derivative per mol/s, shape ``n_fuel.shape``
        """
        return self._solve_for_time_gradient(
            "n_fuel", boundary, mode, t_max, events, objective, objective_grad, ivp_options
        )

    @solve_for_time.grad("n_air")
    def _solve_for_time_grad_n_air(
        self, boundary, mode, t_max, events=None, objective=None, objective_grad=None, **ivp_options
    ):
        r"""
        :math:`\nabla_{n_{air}}` :meth:`~nest.grad.cell.Cell.solve_for_time`

        Returns
        -------
        numpy.ndarray
            objective derivative per mol/s, shape ``n_air.shape``
        """
        return self._solve_for_time_gradient(
            "n_air", boundary, mode, t_max, events, objective, objective_grad, ivp_options
        )

    @solve_for_time.grad("T")
    def _solve_for_time_grad_T(
        self, boundary, mode, t_max, events=None, objective=None, objective_grad=None, **ivp_options
    ):
        r"""
        :math:`\nabla_T` :meth:`~nest.grad.cell.Cell.solve_for_time`

        Returns
        -------
        float
            objective derivative per K
        """
        return self._solve_for_time_gradient(
            "T", boundary, mode, t_max, events, objective, objective_grad, ivp_options
        )

    @solve_for_time.grad("P")
    def _solve_for_time_grad_P(
        self, boundary, mode, t_max, events=None, objective=None, objective_grad=None, **ivp_options
    ):
        r"""
        :math:`\nabla_P` :meth:`~nest.grad.cell.Cell.solve_for_time`

        Returns
        -------
        float
            objective derivative per Pa
        """
        return self._solve_for_time_gradient(
            "P", boundary, mode, t_max, events, objective, objective_grad, ivp_options
        )

    @solve_for_time.grad("fuel_pressure")
    def _solve_for_time_grad_fuel_pressure(
        self, boundary, mode, t_max, events=None, objective=None, objective_grad=None, **ivp_options
    ):
        r"""
        :math:`\nabla_{P_{fuel}}` :meth:`~nest.grad.cell.Cell.solve_for_time`

        Returns
        -------
        float
            always 0 (the fuel-side pressure is not used)
        """
        return self._solve_for_time_gradient(
            "fuel_pressure", boundary, mode, t_max, events, objective, objective_grad, ivp_options
        )

    @solve_for_time.grad("air_pressure")
    def _solve_for_time_grad_air_pressure(
        self, boundary, mode, t_max, events=None, objective=None, objective_grad=None, **ivp_options
    ):
        r"""
        :math:`\nabla_{P_{air}}` :meth:`~nest.grad.cell.Cell.solve_for_time`

        Returns
        -------
        float
            always 0 (the air-side pressure is not used)
        """
        return self._solve_for_time_gradient(
            "air_pressure", boundary, mode, t_max, events, objective, objective_grad, ivp_options
        )
