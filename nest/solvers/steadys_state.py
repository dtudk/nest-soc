from .abstract import Solver, NestResult, Mode

import numpy as np
from scipy.optimize import root

from nest._temp.class_updates import Problem
from nest.constants import F, R
from dataclasses import dataclass, field

@dataclass
class SSResult(NestResult):
    """Steady-state solution with its full field layout.

    Fields are reached as ``result.fields["psi_f"]`` or ``result["psi_f"]``;
    ``coords`` gives the matching cell-center and face coordinates.
    """

    mode: Mode
    y: np.ndarray | None = None
    current_density_profile: np.ndarray | None = None
    residual_norm: float | None = None
    nfev: int | None = None
    fields: dict = field(default_factory=dict)
    coords: dict = field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.fields[name]

    def __str__(self) -> str:
        layout = ", ".join(f"{name}{np.shape(value)}" for name, value in self.fields.items())
        return (f"Steady State Result ({self.mode} mode): "
                f"voltage={self.voltage}, current_density={self.current_density}\n"
                f"  fields: {layout}")

    __repr__ = __str__


class SteadyStateSolver(Solver):
    """Steady state of a :class:`Problem` at fixed voltage or current density.

    Solves ``f(y, V) = 0`` of ``M dy/dt = f`` (:meth:`Problem.fluxes`) with the
    unknowns scaled by :meth:`Problem.typical_state`. With downstream coupling
    the columns are solved one at a time along the flow, else all at once.

    Parameters
    ----------
    mode : Mode or str
        "voltage" (V given) or "current" (mean j given)
    tol : float
        maximum ``|f|`` [A/m^2]
    maxiter : int
        residual evaluations per unknown and column
    j_tol : float
        relative current-density tolerance (current mode)
    max_outer : int
        voltage updates (current mode)
    warm_start : bool, optional
        start each column from the solution upstream; default True for purely
        algebraic problems (``storage() == 0``), whose initial state is only a guess
    initial_voltage : float, optional
        first voltage of a current-controlled solve; default ``boundary.V``
        (the voltage guess of the boundary data)
    """

    def __init__(self, mode: Mode | str = "current", *, tol: float = 1e-7,
                 maxiter: int = 150, j_tol: float = 1e-6, max_outer: int = 25,
                 warm_start: bool | None = None, initial_voltage: float | None = None) -> None:
        self.mode = Mode(mode)
        self.tol = tol
        self.maxiter = maxiter
        self.j_tol = j_tol
        self.max_outer = max_outer
        self.warm_start = warm_start
        self.initial_voltage = initial_voltage

    def solve(self, problem: Problem, initial_state=None) -> SSResult:
        y0 = problem.initial_state() if initial_state is None else np.array(initial_state, dtype=float)
        if y0.size != problem.state_size:
            raise ValueError(
                f"initial_state has size {y0.size}, expected {problem.state_size} for this problem"
            )
        if self.mode is Mode.VOLTAGE:
            return self._solve_voltage(problem, problem.boundary.V, y0)
        return self._solve_current(problem, y0)

    def _solve_voltage(self, problem: Problem, voltage: float, y0) -> SSResult:
        """Potentiostatic solve: root of ``f(y, V)``, one column at a time with downstream coupling."""
        y_scale = problem.typical_state()
        if not problem.downstream_coupling:
            return self._solve_all(problem, voltage, y0, y_scale)
        warm_start = (
            not np.any(problem.storage()) if self.warm_start is None else self.warm_start
        )
        n_columns = problem.n_columns
        z = (y0 / y_scale).reshape(-1, n_columns).copy()
        nfev = 0
        for column in range(n_columns):
            if warm_start and column:
                z[:, column] = z[:, column - 1]  # upstream solution as the guess

            def residual(column_state):
                z[:, column] = column_state
                return problem.fluxes(z.ravel() * y_scale, voltage, column)

            nfev += 1
            if np.max(np.abs(residual(z[:, column]))) <= self.tol:
                continue  # column already converged
            z[:, column], n = self._root(residual, z[:, column].copy(), f"column {column}")
            nfev += n
        return self._result(problem, z.ravel() * y_scale, voltage, Mode.VOLTAGE, nfev)

    def _solve_all(self, problem: Problem, voltage: float, y0, y_scale) -> SSResult:
        """Potentiostatic solve of all unknowns at once (no column structure)."""
        z, nfev = self._root(lambda z: problem.fluxes(z * y_scale, voltage), y0 / y_scale, "the full state")
        return self._result(problem, z * y_scale, voltage, Mode.VOLTAGE, nfev)

    def _root(self, residual, z0, label: str) -> tuple:
        """Solve ``residual(z) = 0`` from ``z0``; returns the solution and the evaluations used."""
        solution = root(
            residual, z0, method="hybr",
            options={"xtol": 1e-12, "maxfev": self.maxiter * (z0.size + 1)},
        )
        max_residual = float(np.max(np.abs(residual(solution.x))))
        if not np.isfinite(max_residual) or max_residual > self.tol:
            raise RuntimeError(f"Steady solver failed at {label}")
        return solution.x, solution.nfev

    def _solve_current(self, problem: Problem, y) -> SSResult:
        """Galvanostatic solve: secant iteration on the voltage over potentiostatic solves."""
        j_target = problem.boundary.j
        V = float(problem.boundary.V if self.initial_voltage is None else self.initial_voltage)
        j_tol = max(self.j_tol * abs(j_target), 1e-9)
        nfev, j = 0, np.nan
        V_prev = j_prev = None
        for _ in range(self.max_outer):
            result = self._solve_voltage(problem, V, y)
            nfev += result.nfev
            j = result.current_density
            if abs(j - j_target) <= j_tol:
                result.mode, result.nfev = Mode.CURRENT, nfev
                return result
            # dV/dj: ohmic path (j = (OCV - V) / R) first, then the secant
            slope = -problem.resistance if j_prev is None or j == j_prev else (V - V_prev) / (j - j_prev)
            V_prev, j_prev = V, j
            V += (j_target - j) * slope
            y = result.y  # warm start
        raise RuntimeError(
            f"Current-controlled steady state did not converge after {self.max_outer} "
            f"iterations (last: V={V:.6f} V, j={j:.1f} A/m^2, target j={j_target:.1f} A/m^2)"
        )

    @staticmethod
    def _result(problem: Problem, y, voltage: float, mode: Mode, nfev: int) -> SSResult:
        j_profile = problem.current_density(y, voltage)
        return SSResult(
            time=0.0, voltage=float(voltage), current_density=float(np.mean(j_profile)),
            mode=mode, y=y, current_density_profile=j_profile,
            residual_norm=float(np.max(np.abs(problem.fluxes(y, voltage)))), nfev=nfev,
            fields=problem.fields(y, voltage), coords=problem.coords(),
        )
