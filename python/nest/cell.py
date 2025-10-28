class Cell:
    """Electrochemical cell model
    
    Parameters:
    area (float) : cell active area [m^2]
    electrodeFuel (Layer) : fuel-side electrode layer
    electrolyte (tuple[Layer]) : tuple of electrolyte-like layers
    electrodeAir (Layer) : air-side electrode layer
    elements (int) : number of elements used in finite-element method

    Returns:
    Cell : electrochemical cell model
    """
    def __init__(self,area,electrodeFuel,electrolyte,electrodeAir,elements=10):
        self.area = area
        self.electrodeFuel = electrodeFuel
        self.electrolyte = electrolyte
        self.electrodeAir = electrodeAir
        self.elements = elements
    def V_nerst(self,T,PsFuel,PsAir):
        """Thermodynamic voltage (reversible limit) [V]

        Parameters:
        T (float) : Temperature [K]
        PsFuel (np.ndarray) : Partial pressure for fuel side [Pa]
        PsAir (np.ndarray) : Partial pressure for air side [Pa]

        Returns:
        float : Thermodynamic voltage limit [V]
        """
        return -(self.electrodeFuel.kinetic.V_nerst_half(T,PsFuel)+self.electrodeAir.kinetic.V_nerst_half(T,PsAir))
    def V(self,j,T,PsFuel,PsAir,**kwargs):
        """Cell voltage [V]

        Parameters:
        j (float) : Current density [A/m^2]
        T (float) : Temperature [K]
        PsFuel (numpy.ndarray) : Partial pressures fuel side [Pa]
        PsAir (numpy.ndarray) : Partial pressures air side [Pa]

        Returns:
        float : Cell voltage [V]
        """
        PsFuel_star = self.electrodeFuel.Ps_star(j,T,PsFuel)
        PsAir_star = self.electrodeAir.Ps_star(j,T,PsAir)
        return self.V_nerst(T,PsFuel_star,PsAir_star)-sum(layer.V(j,T) for layer in self.electrolyte)-self.electrodeFuel.V(j,T,Ps=PsFuel_star,**kwargs)-self.electrodeAir.V(j,T,Ps=PsAir_star,**kwargs)
    def dn_fuel(self,j):
        """Net molar flow per element [mol/s] - fuel side

        Parameters:
        j (float) : current density [A/m^2]
        
        Returns:
        float : net molar flow at element [mol/s]
        """
        return self.electrodeFuel.kinetic.molFlux(j)*self.area/self.elements
    def dn_air(self,j):
        """Net molar flow per element [mol/s] - air side

        Parameters:
        j (float) : current density [A/m^2]
        
        Returns:
        float : net molar flow at element [mol/s]
        """
        return self.electrodeAir.kinetic.molFlux(j)*self.area/self.elements