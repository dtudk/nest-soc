"""
Module providing classes and functions for creating the differential equation problem
"""
import numpy as np
from scipy.optimize import newton

class BoundaryData:
    """ Boundary data for solving initial value ODE

    Parameters:
    V (float) : voltage [V]
    j (float) : current density guess [A/cm^2]
    n_fuel (numpy.ndarray) : molar flow rate - fuel [mol/s]
    n_air (numpy.ndarray) : molar flow rate - air [mol/s]
    T (float) : temperature [K]
    P (float) : pressure [Pa]

    Returns:
    BoundaryData: data for solving initial value ODE
    """
    def __init__(self,
                 V=0,
                 j=0,
                 n_fuel=(0,),
                 n_air=(0,),
                 T=0,
                 P=0):
        self.V = V
        self.j = j
        self.n_fuel = n_fuel
        self.n_air = n_air
        self.T = T
        self.P = P
    def Ps_fuel(self):
        """ Partial pressure for fuel side [Pa]

        Returns:
        numpy.ndarray: Partial pressures for fuel side [Pa]
        """
        return np.array([n/sum(self.n_fuel)*self.P for n in self.n_fuel])
    def Ps_air(self):
        """ Partial pressure for air side [Pa]

        Returns:
        numpy.ndarray: Partial pressures for air side [Pa]
        """
        return np.array([n/sum(self.n_air)*self.P for n in self.n_air])

def advance_step_area(boundary,cell):
    """ Solves current density for element area based on boundary conditions

    Parameters:
    boundary (BoundaryData) : Initial value conditions
    cell (Cell) : Model for electrochemical cell

    Returns:
    BoundaryData : Inivial value condition for next element

    Comments:
    * For now it neglects energy balance and pressure drops.
    * Problem uses center difference method to improve convergence and 
      reduce number of area elements
    * JAX autodiff is not compatible, because scipy.optimize.newton is not compatible
    * auto-diff package is not compatible, because it has not been updated to new numpy versions
    """
    # Use center difference instead of forward difference to reduce the amount of elements required
    def Ps_fuel(x):
        return (boundary.Ps_fuel() + (boundary.n_fuel+cell.dn_fuel(x))/
                sum(boundary.n_fuel+cell.dn_fuel(x))*boundary.P)/2

    def Ps_air(x):
        return (boundary.Ps_fuel() + (boundary.n_air+cell.dn_air(x))/
                sum(boundary.n_air+cell.dn_air(x))*boundary.P)/2

    def function(x):
        return boundary.V - cell.V(x,boundary.T,Ps_fuel(x),Ps_air(x))
    j = newton(function, boundary.j) # use previous step current density or user-input guess

    # Record solution
    n_out_fuel = boundary.n_fuel+cell.dn_fuel(j)
    n_out_air = boundary.n_air+cell.dn_air(j)
    T_out = boundary.T # Simplified for now - Isothermic
    P_out = boundary.P # Simplified for now - Isobaric
    return BoundaryData(V=boundary.V,j=j,n_fuel=n_out_fuel,n_air=n_out_air,T=T_out,P=P_out)

def solve_area(boundary,cell):
    """Wrapper to solve finite-element description of cell area

    Parameters:
    boundary (BoundaryData) : Initial value conditions
    cell (Cell) : Model for electrochemical cell

    Returns:
    numpy.ndarray : Matrix with n_elements x [V,j,T,P,n_fuel,n_air] values 
    """
    solutions = np.zeros((cell.elements,4+len(boundary.n_fuel)+len(boundary.n_air)))
    for e in range(cell.elements):
        boundary = advance_step_area(boundary,cell)
        solutions[e][0] = boundary.V
        solutions[e][1] = boundary.j
        solutions[e][2] = boundary.T
        solutions[e][3] = boundary.P
        for i,n in enumerate(boundary.n_fuel):
            solutions[e][i+4] = n
        for i,n in enumerate(boundary.n_air):
            solutions[e][i+4+len(boundary.n_fuel)] = n
    return solutions
