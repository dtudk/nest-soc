"""
Module providing classes and functions for creating the differential equation problem
"""

import numpy as np
from scipy.optimize import newton
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod

from nest.constants import R
from nest.layers import Layer
from nest.boundary import BoundaryData
from nest.properties import Mixture


class Reaction(ABC):
    """
    Reaction model for advection flow

    Parameters
    ----------
    k : float
        kinetic rate constant [mol/m2]
    E : float
        activation energy for temperature dependency [J/mol.K]
    nu : np.ndarray
        stoichiometric factors for reaction and expected gas mixture [-]
    index : np.ndarray
        array indexes for species to match internal reaction rate equations [-]
    """

    def __init__(self, k: float, E: float, nu: np.ndarray, index: np.ndarray):
        self.k = k
        self.E = E
        self.nu = nu
        self.index = index

    def rate_T(self, T: float):
        """
        Reaction rate constant for specific temperature [mol/m2]

        Parameters
        ----------
        T : float
            temperature [K]
        """
        return self.k * np.exp(-self.E / (R * T))

    @abstractmethod
    def rate(self, T: float, Ps: np.ndarray, gases: Mixture):
        """
        Kinetic reaction rate [mol/m2]

        Parameters
        ----------
        T: float
            temperature [K]
        Ps: np.ndarray
            partial pressures for each specie [Pa]
        gases: Mixture
            information on gas mixture components
        """
        return 0

    def dn(self, T: float, Ps: np.ndarray):
        """
        Mole flow variation from reaction [mol/s]

        Parameters
        ----------
        T: float
            temperature [K]
        Ps: np.ndarray
            partial pressure for each specie [Pa]
        """
        return self.nu * self.rate(T, Ps)


class NoReaction(Reaction):
    """
    Default class without reaction
    """

    def __init__(self):
        self.k = 0
        self.E = 0
        self.nu = np.array([0])
        self.index = np.array([0])

    def rate(self, T, Ps):
        return 0


class WaterGasShift(Reaction):
    """
    Water gas shift reaction

    Notes
    -----
    * The original reaction assumes homogenous reaction along porous media [mol/m3].
    * Here the reaction is assumed to be heterogenous [mol/m2] by multiplying a thickness length [m] to the kinetic rate.
    * The user defines the thickness based on the size of the support layer.

    References
    ----------
    1. https://doi.org/10.1016/j.ijheatmasstransfer.2004.04.010
    """

    def rate(self, T, Ps):
        H2_i, H2O_i, CO2_i, CO_i = self.index
        P = sum(Ps)
        x = Ps / P
        Z = 1000 / T - 1
        K_eq = np.exp(-0.2935 * Z**3 + 0.6351 * Z**2 + 4.1788 * Z + 0.3169)
        return self.rate_T(T) * (x[CO_i] * x[H2O_i] - x[H2_i] * x[CO2_i] / K_eq) * P**2


class Cell:
    """
    Electrochemical cell model

    Parameters
    ----------
    area : float
        cell active area [m^2]
    electrode_fuel : Layer
        fuel-side electrode layer
    electrolyte : tuple[Layer]
        tuple of electrolyte-like layers
    electrode_air : Layer
        air-side electrode layer
    elements : int, optional
        number of elements used in finite-element method
    """

    def __init__(
        self,
        area: float,
        electrode_fuel: Layer,
        electrolyte: Layer,
        electrode_air: Layer,
        elements: int = 10,
        reactions: tuple[Reaction] = (NoReaction(),),
        delta_channel = 0.001,
    ):
        self.area = area
        self.electrode_fuel = electrode_fuel
        self.electrolyte = electrolyte
        self.electrode_air = electrode_air
        self.elements = elements
        self.reactions = reactions
        self.delta_channel = delta_channel 

    def V_nernst(self, T: float, Ps_fuel: np.ndarray, Ps_air: np.ndarray) -> float:
        """
        Thermodynamic voltage (reversible limit) [V]

        Parameters
        ----------
        T : float
            Temperature [K]
        Ps_fuel : np.ndarray
            Partial pressure for fuel side [Pa]
        Ps_air : np.ndarray
            Partial pressure for air side [Pa]
        """
        return -(
            self.electrode_fuel.kinetic.V_nernst_half(T, Ps_fuel)
            + self.electrode_air.kinetic.V_nernst_half(T, Ps_air)
        )
    def Vt_nernst(self, T: float) -> float:
        """
        Thermodynamic voltage (reversible limit) [V]

        Parameters
        ----------
        T : float
            Temperature [K]
        Ps_fuel : np.ndarray
            Partial pressure for fuel side [Pa]
        Ps_air : np.ndarray
            Partial pressure for air side [Pa]
        """
        return -(
            self.electrode_fuel.kinetic.Vt_nernst_half(T)
            + self.electrode_air.kinetic.Vt_nernst_half(T)
        )

    def V(
        self, j: float, T: float, Ps_fuel: np.ndarray, Ps_air: np.ndarray, **kwargs
    ) -> float:
        """
        Cell voltage [V]

        Parameters
        ----------
        j : float
            Current density [A/m^2]
        T : float
            Temperature [K]
        Ps_fuel : numpy.ndarray
            Partial pressures fuel side [Pa]
        Ps_air : numpy.ndarray
            Partial pressures air side [Pa]
        """
        Ps_fuel_star = self.electrode_fuel.Ps_star(j, T, Ps_fuel)
        Ps_air_star = self.electrode_air.Ps_star(j, T, Ps_air)

        V_th = self.V_nernst(T, Ps_fuel_star, Ps_air_star)

        kwargs_fuel = {key: value[0] for key, value in kwargs.items()}
        V_fuel = self.electrode_fuel.V(j, T, Ps=Ps_fuel_star, **kwargs_fuel)

        kwargs_air = {key: value[1] for key, value in kwargs.items()}
        V_air = self.electrode_air.V(j, T, Ps=Ps_air_star, **kwargs_air)

        V_el = 0
        for i, layer in enumerate(self.electrolyte):
            kwargs_el = {key: value[2 + i] for key, value in kwargs.items()}
            V_el += layer.V(j, T, **kwargs_el)

        return V_th - V_el - V_fuel - V_air

    def dn_fuel(self, j: float, boundary: BoundaryData) -> np.ndarray:
        """
        Net molar flow per element [mol/s] - fuel side

        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            temperature [K]
        Ps : np.ndarray
            partial pressures [Pa]
        """
        # Get intermediary composition after electrochemistry
        dn_fuel_int = (
            boundary.n_fuel
            + self.electrode_fuel.kinetic.mol_flux(j) * self.area / self.elements
        )
        Ps = boundary.P * dn_fuel_int / sum(dn_fuel_int)
        # Get kinetic rate
        T = boundary.T
        source = np.zeros(len(Ps))
        for r in self.reactions:
            dn = r.dn(T, Ps)
            for i, dn_i in enumerate(dn):
                source[i] += dn_i
        # Limit problems with sparse discretization (negative flow rates)
        ratio = np.zeros(len(dn_fuel_int))
        for i, r in enumerate(ratio):
            r = (source[i] * self.area / self.elements) / dn_fuel_int[i]
        if max(ratio) > 1.0:
            source = source / (max(ratio) + 1e-3)
        return (
            (self.electrode_fuel.kinetic.mol_flux(j) + source)
            * self.area
            / self.elements
        )

    def dn_air(self, j: float) -> np.ndarray:
        """
        Net molar flow per element [mol/s] - air side

        Parameters
        ----------
        j : float
            current density [A/m^2]
        """
        return self.electrode_air.kinetic.mol_flux(j) * self.area / self.elements

    def Ps_fuel(self, boundary: BoundaryData, j: float) -> np.ndarray:
        """
        Partial pressure for fuel channel [Pa]
        assuming the central difference

        Parameters
        ----------
        boundary : BoundaryData
            boundary state variables
        j : float
            current density [A/cm^2]
        """
        return (
            boundary.Ps_fuel()
            + (boundary.n_fuel + self.dn_fuel(j, boundary))
            / sum(boundary.n_fuel + self.dn_fuel(j, boundary))
            * boundary.P
        ) / 2

    def Ps_air(self, boundary: BoundaryData, j: float) -> np.ndarray:
        """
        Partial pressure for air channel [Pa]
        assuming the central difference

        Parameters
        ----------
        boundary : BoundaryData
            boundary state variables
        j : float
            current density [A/cm^2]
        """
        return (
            boundary.Ps_air()
            + (boundary.n_air + self.dn_air(j))
            / sum(boundary.n_air + self.dn_air(j))
            * boundary.P
        ) / 2

    def advance_step_area(self, boundary: BoundaryData, **kwargs) -> BoundaryData:
        """
        Solves current density for element area based on boundary conditions

        Parameters
        ----------
        boundary : BoundaryData
            Initial value conditions
        cell : Cell
            Model for electrochemical cell

        Notes
        -----
        * For now it neglects energy balance and pressure drops.
        * Problem uses center difference method to improve convergence and
        reduce number of area elements
        * JAX autodiff is not compatible, because scipy.optimize.newton is not compatible
        * auto-diff package is not compatible, because it has not been updated to new numpy versions
        """

        def find_current(x):
            return boundary.V - self.V(
                x,
                boundary.T,
                self.Ps_fuel(boundary, x),
                self.Ps_air(boundary, x),
                **kwargs,
            )

        j = newton(
            find_current, boundary.j, tol=10
        )  # Note: it uses previous step current density or user-input guess

        # Record solution
        n_out_fuel = boundary.n_fuel + self.dn_fuel(j, boundary)
        n_out_air = boundary.n_air + self.dn_air(j)
        T_out = boundary.T  # Simplified for now - Isothermic
        P_out = boundary.P  # Simplified for now - Isobaric
        return BoundaryData(
            V=boundary.V, j=j, n_fuel=n_out_fuel, n_air=n_out_air, T=T_out, P=P_out
        )

    def solve_for_voltage(self, boundary: BoundaryData, **kwargs) -> np.ndarray:
        """
        Wrapper to solve finite-element description of cell area
        solving for the voltage value in boundary condition

        Parameters
        ----------
        boundary : BoundaryData
            Initial value conditions
        cell : Cell
            Model for electrochemical cell
        """
        n_layers = 2 + len(self.electrolyte)
        n_variables = 4 + len(boundary.n_fuel) + len(boundary.n_air) + n_layers
        solutions = np.zeros((n_variables, self.elements))
        for e in range(self.elements):
            # converting arrays of keywords into a single value kewywords
            kwargs_el = {key: value[e] for key, value in kwargs.items()}
            boundary = self.advance_step_area(boundary, **kwargs_el)
            solutions[0][e] = boundary.V
            solutions[1][e] = boundary.j
            solutions[2][e] = boundary.T
            solutions[3][e] = boundary.P
            for i, n in enumerate(boundary.n_fuel):
                solutions[i + 4][e] = n
            for i, n in enumerate(boundary.n_air):
                solutions[i + 4 + len(boundary.n_fuel)][e] = n
            aux = 4 + len(boundary.n_fuel) + len(boundary.n_air)
            # Recording overpotentials (for analysis and degradation modeling)
            kwargs_fuel = {key: value[0] for key, value in kwargs_el.items()}
            solutions[aux][e] = self.electrode_fuel.V(
                boundary.j,
                boundary.T,
                self.Ps_fuel(boundary, boundary.j),
                **kwargs_fuel,
            )
            kwargs_air = {key: value[1] for key, value in kwargs_el.items()}
            solutions[aux + 1][e] = self.electrode_air.V(
                boundary.j, boundary.T, self.Ps_air(boundary, boundary.j), **kwargs_air
            )
            for i, layer in enumerate(self.electrolyte):
                kwargs_electro = {key: value[2 + i] for key, value in kwargs_el.items()}
                solutions[aux + 2 + i][e] = layer.V_ohm(
                    boundary.j, boundary.T, **kwargs_electro
                )

        return solutions

    def solve_for_current(self, boundary: BoundaryData, **kwargs) -> np.ndarray:
        """
        Wrapper to solve finite-element description of cell area
        solving for the current value in boundary condition

        Parameters
        ----------
        boundary : BoundaryData
            Initial value conditions
        cell : Cell
            Model for electrochemical cell
        """

        def find_voltage(V, **kwargs):
            boundary.V = V
            solution = self.solve_for_voltage(boundary, **kwargs)
            return boundary.j - np.sum(solution[1]) / self.elements

        boundary.V = newton(lambda V: find_voltage(V, **kwargs), boundary.V, tol=5e-4)
        return self.solve_for_voltage(boundary, **kwargs)

    def count_pol_deg(self) -> int:
        """
        Returns number of degration models for polarization resistance
        """
        counter = 0
        if self.electrode_fuel.degradation.pol_active:
            counter += 1
        if self.electrode_air.degradation.pol_active:
            counter += 1
        return counter

    def count_ohm_deg(self) -> int:
        """
        Returns number of degradation models for ohmic resistance
        """
        counter = 0
        for layer in self.electrolyte:
            if layer.degradation.ohm_active:
                counter += 1
        return counter

    def solve_time_step(
        self, y: np.ndarray, boundary: BoundaryData, mode: str
    ) -> np.ndarray:
        """
        Returns the steady-state solution matrix for boundary conditions

        Parameters
        ----------
        y : np.ndarray
            array of material properties from the degradation ODE
        boundary : BoundaryData
            state variable conditions of the cell
        mode : str
            operational mode: "current" or "voltage" constant

        Notes
        -----
        * Missing the "current+voltage" mode of operation (varying temperature)

        """
        n_pol_deg = self.count_pol_deg()
        n_ohm_deg = self.count_ohm_deg()
        n_layers = 2 + len(self.electrolyte)
        aux_counter = n_pol_deg * self.elements
        pol_deg = np.zeros((self.elements, n_layers))
        ohm_deg = np.zeros((self.elements, n_layers))

        for i in range(self.elements):
            aux = 0
            if self.electrode_fuel.degradation.pol_active:
                pol_deg[i][0] = y[i * n_pol_deg]
                aux += 1
            if self.electrode_air.degradation.pol_active:
                pol_deg[i][1] = y[i * n_pol_deg + aux]

            aux = 0
            for j, layer in enumerate(self.electrolyte):
                if layer.degradation.ohm_active:
                    ohm_deg[i][j] = y[aux + i * n_ohm_deg + aux_counter]
                    aux += 1
        kwargs = {"pol_deg": pol_deg, "ohm_deg": ohm_deg}

        if mode == "voltage":
            steady_solution = self.solve_for_voltage(boundary, **kwargs)
            boundary.j = np.sum(steady_solution[1]) / self.elements
        elif mode == "current":
            steady_solution = self.solve_for_current(boundary, **kwargs)
            boundary.V = np.sum(steady_solution[0]) / self.elements
        else:
            raise ValueError("Undefined mode of operation")

        return steady_solution

    def advance_time_step(
        self, t: np.ndarray, y: np.ndarray, boundary: BoundaryData, mode: str
    ) -> np.ndarray:
        """
        Returns the material change rates from degradation
        units varies depending on the degradation model

        Parameters
        ----------
        t : np.ndarray
            time array required for scipy.integrate.ivp_solve
        y : np.ndarray
            material variables (pol_deg and ohm_deg) that change during degration
        boundary : BoundaryData
            state variables for the cell related to the boundary conditions of the ODE
        mode : str
            operational mode for the degradation simulation (i.e., "current" or "voltage")
        """
        # 1. Create the placeholders
        n_pol_deg = self.count_pol_deg()
        n_ohm_deg = self.count_ohm_deg()
        pol_deg_dt = np.zeros(self.elements * n_pol_deg)
        ohm_deg_dt = np.zeros(self.elements * n_ohm_deg)

        steady_solution = self.solve_time_step(y, boundary, mode)
        # 3. Calculate the rates for each segment and layer
        index = 4 + len(boundary.n_fuel) + len(boundary.n_air)
        for j in range(self.elements):
            i = 0
            # 3.1 Create the state for layer conditions
            n_fuel = [steady_solution[4 + i][j] for i in range(len(boundary.n_fuel))]
            n_air = [
                steady_solution[4 + len(n_fuel) + i][j]
                for i in range(len(boundary.n_air))
            ]
            in_b = BoundaryData(
                boundary.V, steady_solution[1][j], n_fuel, n_air, boundary.T, boundary.P
            )
            P_fuel_tpb = self.electrode_fuel.Ps_star(
                steady_solution[1][j],
                steady_solution[2][j],
                self.Ps_fuel(in_b, steady_solution[1][j]),
            )
            P_air_tpb = self.electrode_air.Ps_star(
                steady_solution[1][j],
                steady_solution[2][j],
                self.Ps_air(in_b, steady_solution[1][j]),
            )
            # OBS: boundary.P may not be equal to sum(P_fuel_tpb) or sum(P_air_tpb)
            in_b = BoundaryData(
                boundary.V, boundary.j, P_fuel_tpb, P_air_tpb, boundary.T, boundary.P
            )
            # 3.2 Calculate rates
            if self.electrode_fuel.degradation.pol_active:
                in_b.V = steady_solution[index][j]
                pol_deg_dt[i + j * n_pol_deg] = (
                    self.electrode_fuel.degradation.material_dt(
                        y[i + j * n_pol_deg], in_b, t
                    )
                )
                i += 1
            if self.electrode_air.degradation.pol_active:
                in_b.V = steady_solution[index + 1][j]
                pol_deg_dt[i + j * n_pol_deg] = (
                    self.electrode_air.degradation.material_dt(
                        y[i + j * n_pol_deg], in_b, t
                    )
                )
                i += 1

        for j in range(self.elements):
            i = 0
            for k, layer in enumerate(self.electrolyte):
                if layer.degradation.ohm_active:
                    in_b.V = steady_solution[index + 2 + k][j]
                    ohm_deg_dt[i + j * n_ohm_deg] = layer.degradation.material_dt(
                        y[i + j * n_ohm_deg + n_pol_deg * self.elements], in_b, t
                    )
                    i += 1
        power_dt = [boundary.j * boundary.V]
        # 4. Return it
        return np.concatenate((pol_deg_dt, ohm_deg_dt, power_dt))

    def solve_for_time(self, boundary, mode, t_max):
        """
        Wrapper to solve the differential equations for degradation

        Parameters
        ----------
        boundary : BoundaryData
            state variables for the cell related to the boundary conditions of the ODE
        mode : str
            operational mode for the degradation simulation (i.e., "current" or "voltage")
        t_max : int
            Final time for simulation in hours [h]
        """
        # Initial value boundary
        n_pol_deg = self.count_pol_deg()
        n_ohm_deg = self.count_ohm_deg()
        pol_deg_0 = np.zeros(self.elements * n_pol_deg)
        ohm_deg_0 = np.zeros(self.elements * n_ohm_deg)
        power_0 = np.zeros(1)

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
                    ohm_deg_0[k] = layer.degradation.m0
                    k += 1

        y0 = np.concatenate((pol_deg_0, ohm_deg_0, power_0))
        t_span = (0, t_max)
        return solve_ivp(
            self.advance_time_step, t_span, y0, args=(boundary, mode), method="RK23"
        )
