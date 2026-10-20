"""Steady-state and transient solvers."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult, root

from nest.constants import F, R

from .model import Model
from .protocol import _Protocol
from .circuit import ExternalCircuit

__all__ = ["solve_steady", "solve_transient"]


def solve_steady(
    model: Model,
    voltage: float,
    initial_state: np.ndarray | None = None,
    tol: float = 1e-7,
    maxiter: int = 150,
) -> OptimizeResult:
    """Solve the cell equations at a fixed voltage.

    Parameters
    ----------
    model : Model
        Cell equations and parameters.
    voltage : float
        Voltage [V].
    initial_state : array_like, optional
        Initial state vector; defaults to model.initial_state().
    tol : float, optional
        Maximum allowed residual of the steady-state equations [A/m^2].
        Default is 1e-7.
    maxiter : int, optional
        Controls the maximum number of residual evaluations per channel column.
        The limit is maxiter * (number of unknowns + 1). Default is 150.

    Returns
    -------
    solution : OptimizeResult
        Steady-state solution (x), time derivatives (fun), and maximum
        scaled residual (residual_norm) [A/m^2].
    """
    
    y0 = model.initial_state() if initial_state is None else np.asarray(initial_state, dtype=float).copy()
    state_fields = model.unpack_state(y0)

    c_ref_f = model.fuel_pressure / (R * model.T)
    c_ref_a = model.air_pressure / (R * model.T)
    # psi_f, psi_a, c_pf, c_pa, c_cf, c_ca, c_ps
    y_scales = (0.1, 0.1, c_ref_f, c_ref_a, c_ref_f, c_ref_a, c_ref_f)
    y_scale = np.repeat(y_scales, [field.size for field in state_fields])

    flux_scales = (
        model.cell.electrode_fuel.capacitance_volumetric * model.grid.dz_fuel,
        model.cell.electrode_air.capacitance_volumetric * model.grid.dz_air,
        F * model.cell.electrode_fuel.transport.epsilon * model.grid.dz_fuel,
        F * model.cell.electrode_air.transport.epsilon * model.grid.dz_air,
        F * model.cell.fuel_channel.width * model.cell.fuel_channel.height / model.width_active,
        F * model.cell.air_channel.width * model.cell.air_channel.height
        / (model.cell.area / model.cell.air_channel.length),
        F * model.support_transport.epsilon * model.grid.dz_support,
    )
    flux_scale = np.repeat(flux_scales, [field.size for field in state_fields])

    initial_derivatives = model.time_derivatives(0.0, y0, voltage)
    initial_max_residual = float(np.max(np.abs(flux_scale * initial_derivatives)))

    if initial_max_residual <= tol:
        return OptimizeResult(
            x=y0, fun=initial_derivatives, success=True,
            message="Initial state satisfies the steady equations",
            nfev=1, residual_norm=initial_max_residual,
        )


    z = (y0 / y_scale).reshape(-1, model.n_x).copy()
    nfev = 0
    for column in range(model.n_x):
        def _residual(column_state):
            z[:, column] = column_state
            y = z.ravel() * y_scale
            derivatives = model.time_derivatives(0.0, y, voltage)
            residuals = (flux_scale * derivatives).reshape(z.shape)
            return residuals[:, column]

        solution = root(_residual, z[:, column].copy(), method="hybr",
                        options={"xtol": 1e-12, "maxfev": maxiter * (z.shape[0] + 1)})
        nfev += solution.nfev
        max_residual = float(np.max(np.abs(_residual(solution.x))))
        if not np.isfinite(max_residual) or max_residual > tol:
            raise RuntimeError(f"Steady solver failed at channel column (n_x) {column}")

    y = z.ravel() * y_scale
    derivatives = model.time_derivatives(0.0, y, voltage)
    max_residual = float(np.max(np.abs(flux_scale * derivatives)))

    solution.x = y
    solution.fun = derivatives
    solution.residual_norm = max_residual
    solution.success = True
    solution.nfev = nfev
    # Discard matrix details from the last column solve.
    for key in ("fjac", "r", "qtf"):
        solution.pop(key, None)
    return solution




def solve_transient(
    model: Model,
    protocol: _Protocol,
    t_span: tuple[float, float],
    initial_state: np.ndarray | None = None,
    t_eval: np.ndarray | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    circuit: ExternalCircuit | None = None,
    source_voltage: float = 0.0,
):
    """Integrate the cell response using BDF.

    Parameters
    ----------
    model : Model
        Cell equations and parameters.
    protocol : callable
        Command created by voltage_ramp or current_interrupt.
    t_span : tuple of float
        Start and end times [s], with start < end.
    initial_state : array_like, optional
        Initial state vector; defaults to model.initial_state().
    t_eval : array_like, optional
        Output times [s]. Default uses the solver's time steps.
    rtol : float, optional
        Relative tolerance. Default is 1e-5.
    atol : float or array_like, optional
        Absolute tolerance per state. Default is 1e-7.
        For an array, include I [A] and V_C [V] when using a circuit.
    circuit : ExternalCircuit, optional
        External RLC circuit used to calculate non-ideal current interrupt.
    source_voltage : float, optional
        Constant source voltage opposing the cell [V]. Default is 0.

    Returns
    -------
    solution : OdeResult
        Times [s] in t and cell states in y.
        With a circuit, also includes current [A], current_density [A/m^2],
        voltage [V], and capacitor_voltage [V].
    """
    y0 = model.initial_state() if initial_state is None else np.asarray(initial_state, dtype=float).copy()
    start, end = t_span
    if circuit is not None:
        y0 = np.r_[y0, protocol.current_before * model.cell.area, 0.0]

    if protocol.change_time is not None and start < protocol.change_time < end:
        segment_edges = [start, protocol.change_time, end]
    else:
        segment_edges = [start, end]
    times, states = [], []
    nfev = njev = nlu = 0

    for i in range(len(segment_edges) - 1):
        segment_start = segment_edges[i]
        segment_end = segment_edges[i + 1]
        circuit_active = circuit is not None and segment_start >= protocol.change_time
        def _time_derivatives(t: float, y: np.ndarray) -> np.ndarray:
            """Calculate state time derivatives using the protocol's voltage or current."""
            if circuit is not None:
                cell_state, current, capacitor_voltage = y[:-2], y[-2], y[-1]
                voltage = model.voltage_for_current(cell_state, current / model.cell.area)
                cell_rate = model.time_derivatives(t, cell_state, voltage)
                circuit_rate = circuit.derivatives(
                    current, capacitor_voltage, voltage, source_voltage
                ) if circuit_active else (0.0, 0.0)
                return np.r_[cell_rate, circuit_rate]
            if protocol.control_mode == "current":
                if t == segment_end == protocol.change_time:
                    t = np.nextafter(segment_end, segment_start)
                voltage = model.voltage_for_current(y, protocol(t))
            else:
                voltage = protocol(t)
            return model.time_derivatives(t, y, float(voltage))

        output_times = integration_times = None

        if t_eval is not None:
            t_eval = np.asarray(t_eval)
            output_times = t_eval[(t_eval >= segment_start) & (t_eval <= segment_end)]

            if i > 0:
                output_times = output_times[output_times != segment_start]

            integration_times = np.unique(np.append(output_times, segment_end))

        solution = solve_ivp(
            _time_derivatives, (segment_start, segment_end), y0, 
            method="BDF", t_eval=integration_times, rtol=rtol, atol=atol,
        )

        if not solution.success:
            raise RuntimeError(f"Transient solver did not converge: {solution.message}")
        y0 = solution.y[:, -1]
        if output_times is None:
            times.append(solution.t[1:] if i > 0 else solution.t)
            states.append(solution.y[:, 1:] if i > 0 else solution.y)
        else:
            times.append(output_times)
            states.append(solution.y[:, np.isin(solution.t, output_times)])
        nfev += solution.nfev
        njev += solution.njev
        nlu += solution.nlu

    solution.t, solution.y = np.concatenate(times), np.concatenate(states, axis=1)
    solution.nfev, solution.njev, solution.nlu = nfev, njev, nlu
    if circuit is not None:
        solution.current = solution.y[-2].copy()
        solution.capacitor_voltage = solution.y[-1].copy()
        solution.y = solution.y[:-2]
        solution.current_density = solution.current / model.cell.area
        solution.voltage = np.array([
            model.voltage_for_current(state, current_density)
            for state, current_density in zip(solution.y.T, solution.current_density)
        ])
    return solution
