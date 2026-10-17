"""
Provide data object for solving problems and transfering information for degradation
"""

import numpy as np


class BoundaryData:
    """
    Boundary data for solving initial value ODE

    Parameters
    ----------
    V : float
        voltage (fix or guess) [V]
    j : float
        current density (fix or guess) [A/m^2]
    n_fuel : numpy.ndarray
        molar flow rate - fuel [mol/s]
    n_air : numpy.ndarray
        molar flow rate - air [mol/s]
    T : float
        temperature [K]
    P : float
        pressure [Pa]
    fuel_pressure : float, optional
        Fuel-side pressure [Pa]; defaults to P at initialization.
    air_pressure : float, optional
        Air-side pressure [Pa]; defaults to P at initialization.
    """

    def __init__(
        self,
        V: float = 1,
        j: float = 0,
        n_fuel: np.ndarray = np.zeros(1),
        n_air: np.ndarray = np.zeros(1),
        T: float = 25 + 273.15,
        P: float = 1e5,
        fuel_pressure: float | None = None,
        air_pressure: float | None = None,
    ):
        fuel_pressure = P if fuel_pressure is None else fuel_pressure
        air_pressure = P if air_pressure is None else air_pressure
        self.V = V
        self.j = j
        self.n_fuel = np.array(n_fuel)
        self.n_air = np.array(n_air)
        self.T = T
        self.P = P
        self.fuel_pressure = fuel_pressure
        self.air_pressure = air_pressure

    def Ps_fuel(self) -> np.ndarray:
        """
        Partial pressure for fuel side [Pa]
        """
        return np.array([n / sum(self.n_fuel) * self.P for n in self.n_fuel])

    def Ps_air(self) -> np.ndarray:
        """
        Partial pressure for air side [Pa]
        """
        return np.array([n / sum(self.n_air) * self.P for n in self.n_air])
