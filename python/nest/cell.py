"""
Module providing the classes and functions representing the electrochemical cell.
"""

class Cell:
    """Electrochemical cell model
    
    Parameters:
    area (float) : cell active area [m^2]
    electrode_fuel (Layer) : fuel-side electrode layer
    electrolyte (tuple[Layer]) : tuple of electrolyte-like layers
    electrode_air (Layer) : air-side electrode layer
    elements (int) : number of elements used in finite-element method

    Returns:
    Cell : electrochemical cell model
    """
    def __init__(self,area,electrode_fuel,electrolyte,electrode_air,elements=10):
        self.area = area
        self.electrode_fuel = electrode_fuel
        self.electrolyte = electrolyte
        self.electrode_air = electrode_air
        self.elements = elements
    def V_nerst(self,T,Ps_fuel,Ps_air):
        """Thermodynamic voltage (reversible limit) [V]

        Parameters:
        T (float) : Temperature [K]
        PsFuel (np.ndarray) : Partial pressure for fuel side [Pa]
        PsAir (np.ndarray) : Partial pressure for air side [Pa]

        Returns:
        float : Thermodynamic voltage limit [V]
        """
        return -(self.electrode_fuel.kinetic.V_nerst_half(T,Ps_fuel)+
                 self.electrode_air.kinetic.V_nerst_half(T,Ps_air))
    def V(self,j,T,Ps_fuel,Ps_air,**kwargs):
        """Cell voltage [V]

        Parameters:
        j (float) : Current density [A/m^2]
        T (float) : Temperature [K]
        PsFuel (numpy.ndarray) : Partial pressures fuel side [Pa]
        PsAir (numpy.ndarray) : Partial pressures air side [Pa]

        Returns:
        float : Cell voltage [V]
        """
        Ps_fuel_star = self.electrode_fuel.Ps_star(j,T,Ps_fuel)
        Ps_air_star = self.electrode_air.Ps_star(j,T,Ps_air)
        return (self.V_nerst(T,Ps_fuel_star,Ps_air_star)-
                sum(layer.V(j,T,**kwargs) for layer in self.electrolyte)-
                self.electrode_fuel.V(j,T,Ps=Ps_fuel_star,**kwargs)-
                self.electrode_air.V(j,T,Ps=Ps_air_star,**kwargs))
    def dn_fuel(self,j):
        """Net molar flow per element [mol/s] - fuel side

        Parameters:
        j (float) : current density [A/m^2]
        
        Returns:
        float : net molar flow at element [mol/s]
        """
        return self.electrode_fuel.kinetic.mol_flux(j)*self.area/self.elements
    def dn_air(self,j):
        """Net molar flow per element [mol/s] - air side

        Parameters:
        j (float) : current density [A/m^2]
        
        Returns:
        float : net molar flow at element [mol/s]
        """
        return self.electrode_air.kinetic.mol_flux(j)*self.area/self.elements
