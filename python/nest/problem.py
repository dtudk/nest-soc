import numpy as np
from scipy.optimize import newton

class BoundaryData:
    """ Boundary data for solving initial value ODE

    Parameters:
    V (float) : voltage [V]
    j (float) : current density guess [A/cm^2]
    nFuel (numpy.ndarray) : molar flow rate - fuel [mol/s]
    nAir (numpy.ndarray) : molar flow rate - air [mol/s]
    T (float) : temperature [K]
    P (float) : pressure [Pa]

    Returns:
    BoundaryData: data for solving initial value ODE
    """
    def __init__(self,V=0,j=0,nFuel=(0,),nAir=(0,),T=0,P=0):
        self.V = V
        self.j = j
        self.nFuel = nFuel
        self.nAir = nAir
        self.T = T
        self.P = P
    def PsFuel(self):
        """ Partial pressure for fuel side [Pa]

        Returns:
        numpy.ndarray: Partial pressures for fuel side [Pa]
        """
        return np.array([n/sum(self.nFuel)*self.P for n in self.nFuel])
    def PsAir(self):
        """ Partial pressure for air side [Pa]

        Returns:
        numpy.ndarray: Partial pressures for air side [Pa]
        """
        return np.array([n/sum(self.nAir)*self.P for n in self.nAir])

def advanceStepArea(boundary,cell):
    """ Solves current density for element area based on boundary conditions

    Parameters:
    boundary (BoundaryData) : Initial value conditions
    cell (Cell) : Model for electrochemical cell

    Returns:
    BoundaryData : Inivial value condition for next element

    Comments:
    * For now it neglects energy balance and pressure drops.
    * Problem uses center difference method to improve convergence and redunce number of area elements
    * JAX autodiff is not compatible, because scipy.optimize.newton is not compatible
    * auto-diff package is not compatible, because it has not been updated to new numpy versions
    """
    # Use center difference instead of forward difference to reduce the amount of elements required
    PsFuel = lambda x : (boundary.PsFuel() + (boundary.nFuel+cell.dn_fuel(x))/sum(boundary.nFuel+cell.dn_fuel(x))*boundary.P)/2
    PsAir = lambda x : (boundary.PsAir() + (boundary.nAir+cell.dn_air(x))/sum(boundary.nAir+cell.dn_air(x))*boundary.P)/2
    function = lambda x : boundary.V - cell.V(x,boundary.T,PsFuel(x),PsAir(x))
    j = newton(function, boundary.j) # use previous step current density or user-input guess

    # Record solution
    n_out_fuel = boundary.nFuel+cell.dn_fuel(j)
    n_out_air = boundary.nAir+cell.dn_air(j)
    T_out = boundary.T # Simplified for now - Isothermic
    P_out = boundary.P # Simplified for now - Isobaric
    return BoundaryData(V=boundary.V,j=j,nFuel=n_out_fuel,nAir=n_out_air,T=T_out,P=P_out)

def solveArea(boundary,cell):
    """Wrapper to solve finite-element description of cell area

    Parameters:
    boundary (BoundaryData) : Initial value conditions
    cell (Cell) : Model for electrochemical cell

    Returns:
    numpy.ndarray : Matrix with n_elements x [V,j,T,P,nFuel,nAir] values 
    """
    solutions = np.zeros((cell.elements,4+len(boundary.nFuel)+len(boundary.nAir)))
    for e in range(cell.elements):
        boundary = advanceStepArea(boundary,cell)
        solutions[e][0] = boundary.V
        solutions[e][1] = boundary.j
        solutions[e][2] = boundary.T
        solutions[e][3] = boundary.P
        for i,n in enumerate(boundary.nFuel):
            solutions[e][i+4] = n
        for i,n in enumerate(boundary.nAir):
            solutions[e][i+4+len(boundary.nFuel)] = n
    return solutions