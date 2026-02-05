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
        current density (fix or guess) [A/cm^2]
    n_fuel : numpy.ndarray
        molar flow rate - fuel [mol/s]
    n_air : numpy.ndarray
        molar flow rate - air [mol/s]
    T : float
        temperature [K]
    P : float
        pressure [Pa]
    """

    def __init__(
        self,
        V: float = 0,
        j: float = 0,
        n_fuel: np.ndarray = np.zeros(1),
        n_air: np.ndarray = np.zeros(1),
        T: float = 0,
        P: float = 0,
    ):
        self.V = V
        self.j = j
        self.n_fuel = n_fuel
        self.n_air = n_air
        self.T = T
        self.P = P

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
