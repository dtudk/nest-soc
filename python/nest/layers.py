"""
Module providing the classes and functions for the material layers in the single repeating unit
"""

import numpy as np
from scipy.optimize import newton
from scipy.constants import pi

from nest.constants import R,F,P_atm
from nest.properties import Specie,Mixture,BasicSpecies

class Kinetic():
    """
    Kinetic model for eletrochemical reaction
    
    Parameters
    ----------
    gas : Mixture or Specie
        gas produced/consumed in reaction
    nu : numpy.ndarray
        reaction stoichoimetric coefficients
    n_e : float
        electron moles transfered in half-reaction 
    alpha : float
        charge transfer coefficient - forward reaction
    beta : float
        charge transfer coefficient - reverse reaction
    gamma : float
        pre-exponential kinetic factor [A]
    p : numpy.ndarray
        exponential factors for partial pressure dependency
    theta : float
        exponential factor for temperature dependency
    E_act : float
        activation energy for temperature dependency [J/mol.K]
    
    Notes
    -----
    * General exchange current density equation from Ref. [1]
    * The half-reaction assumed as an heteregenous reaction (see Ref. [1])
    * No current losses

    References
    ----------
    1. https://doi.org/10.1016/j.pecs.2020.100902
    """
    def __init__(self,
                 gas=BasicSpecies.H2,
                 nu=np.array([0]),
                 n_e=2,
                 alpha=0,
                 beta=0,
                 gamma=0,
                 p=np.array([0]),
                 theta=1,
                 E_act=0):
        # Related to reaction balance
        self.gas = gas
        self.nu = nu
        self.n_e = n_e
        # Related to exchange current density
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.p = p
        self.theta = theta
        self.E_act = E_act
    def j_0(self,
            T:float,
            Ps:np.ndarray)->float:
        """
        Exchange current density [A/m^2]

        Parameters
        ----------
        T : float
            Temperature [K]
        Ps : numpy.ndarray
            Partial pressures [Pa]
        """
        return self.gamma*T**self.theta*np.exp(-self.E_act/(R*T))*np.prod((Ps/P_atm)**self.p)
    def mol_flux(self,
                 j:float)->np.ndarray:
        """
        Molar flux [mol/m^2]

        Parameters
        ----------
        j : float
            Current density [A/m^2]
        """
        return self.nu*j/(self.n_e*F)
    def V_nerst_half(self,
                     T,
                     Ps):
        """
        Voltage for electrode side reaction [V]

        Parameters
        ----------
        T : float
            Temperature [K]
        Ps : numpy.ndarray
            Partial pressures
        """
        if isinstance(self.gas.species, Specie):
            return self.nu[0]*self.gas.species.g(T,Ps[0])/(self.n_e*F)
        else:
            return sum([self.nu[i]*gas.g(T,Ps[i]) for i,gas
                        in enumerate(self.gas.species)])/(self.n_e*F)
    def V(self,
          j:float,
          T:float,
          j0:float)->float:
        """
        Voltage [V]

        Parameters
        ----------
        j : float
            Activation overpotential [V]
        T : float
            Temperature [K]
        j0 : float
            Exchange current density [A/m^2]

        Notes
        -----
        * It works as a pseudo-abstract method
        """
        return 0

class ButlerVolmer(Kinetic):
    """
    Butler-Volmer half-reaction kinetics

    Parameters
    ----------
    gas : Mixture
        species that are produced/consumed in reaction (no ions)
    nu : numpy.ndarray
        stoichoimetric coefficients of reaction
    n_e : float
        number of electron moles transfered in half-reaction 
    alpha : float
        charge transfer coefficient - forward reaction
    beta : float
        charge transfer coefficient - reverse reaction
    gamma : float
        pre-exponential kinetic factor [A]
    p : numpy.ndarray
        exponential factors for partial pressure dependency
    theta : float
        exponential factor for Arrhenius equation
    E_act : float
        activation energy for Arrhenius equation [J/mol.K]    
    """
    def j(self,
          V:float,
          T:float,
          j0:float)->float:
        """
        Current density [A/m^2]

        Parameters
        ----------
        V : float
            Activation overpotential [V]
        T : float
            Temperature [K]
        j0 : float
            Exchange current density [A/m^2]
        """
        return j0*(np.exp(self.alpha*self.n_e*F*V/(R*T))-np.exp(-self.beta*self.n_e*F*V/(R*T)))
    def dj_dV(self,
              V:float,
              T:float,
              j0:float)->float:
        """
        First derivative of current density by voltage [A/m^2.V]

        Parameters
        ----------
        V (float) : Activation overpotential [V]
        T (float) : Temperature [K]
        j0 (float) : Exchange current density [A/m^2]
        """
        return j0*(self.alpha*self.n_e*F/(R*T)*np.exp(self.alpha*self.n_e*F*V/(R*T))+
                   self.beta*self.n_e*F/(R*T)*np.exp(-self.beta*self.n_e*F*V/(R*T)))
    def dj_dV2(self,
               V:float,
               T:float,
               j0:float)->float:
        """
        Second derivative of current density by voltage [A/m^2.V^2]

        Parameters
        ----------
        V : float
            Activation overpotential [V]
        T : float
            Temperature [K]
        j0 : float
            Exchange current density [A/m^2]
        """
        return j0*((self.alpha*self.n_e*F/(R*T))**2*np.exp(self.alpha*self.n_e*F*V/(R*T))-
                   (self.beta*self.n_e*F/(R*T))**2*np.exp(-self.beta*self.n_e*F*V/(R*T)))
    def V(self,
          j:float,
          T:float,
          j0:float)->float:
        """
        Activation overpotential [V]  

        Parameters
        ----------
        j : float
            Current density [A/m^2]
        T : float
            Temperature [K]
        j0 : float
            Exchange current density [A/m^2]

        Notes
        -----
        * Linear, Sinh and Tafel approximations come from the mathematical approximations of 
          BV under restriced conditions [1]. 
        * This function is not compatible with JAX package, because of the use of 
          scipy.optimize.newton function
        * Note that j0 is passed as an argument to reduce the computational work

        References
        ----------
        1. https://doi.org/10.1016/j.pecs.2020.100902
        """
        # Finding V guess
        if abs(j) <= j0:
            # Linear approximation
            guess = R*T/(self.n_e*F)*(j/j0)
        elif (0.5 <= self.alpha/self.beta <= 1.5) and abs(j/j0) < 3:
            # Sinh approximation: sinh(x) = (exp(x)-exp(-x))/2
            # Comment: only better at limited conditions where alpha/beta approx. 1
            coef = self.alpha if j > 0 else -self.beta
            guess = (R*T/(coef*self.n_e*F)*np.arcsinh(abs(j)/j0/2))
        else:
            # Taffel approximation
            coef = self.alpha if j > 0 else -self.beta
            guess = R*T/(coef*self.n_e*F)*np.log(abs(j)/j0)
        # Root finding
        return newton(lambda x:self.j(x,T,j0)-j,guess,fprime=lambda x:self.dj_dV(x,T,j0),
                      fprime2=lambda x:self.dj_dV2(x,T,j0))

class Conductivity:
    """
    Generic conductivity model
    
    Parameters
    ----------
    sigma0 : float
        Conductivity at reference state [S/m]
    E_act : float
        Activation energy [J/mol.K]
    theta : float
        Temperature exponential coefficient

    Notes
    -----
    * Default : no resistance
    """
    def __init__(self,
                 sigma0=np.inf,
                 E_act=0.0,
                 theta=-1):
        self.sigma0 = sigma0
        self.E_act = E_act
        self.theta = theta
    def sigma(self,
              T:float)->float:
        """
        Material conductivity at given temperature (T) [S/m]

        Parameters
        ----------
        T : float
            Temperature [K]
        """
        return self.sigma0*T**self.theta*np.exp(-self.E_act/(R*T))

class PorousTransport:
    """
    Generic diffusion over porous media
    
    Parameters
    ----------
    dp : float
        average porous diameter [m]
    epsilon : float
        porosity
    tau : float
        tortuosity
    """
    def __init__(self,
                 dp=0.0,
                 epsilon=0.0,
                 tau=0.0):
        self.dp = dp
        self.epsilon = epsilon
        self.tau = tau
    def D_knudsen(self,
                  T:float,
                  specie:Specie)->float:
        """
        Knudsen diffusion coefficient [m^2/s^2]

        Parameters
        ----------
        T : float
            Temperature [K]
        specie : Specie
            Thermodynamic specie model
        
        References
        1. https://doi.org/10.1016/j.pecs.2020.100902
        """
        return self.dp/3*(8*R*T/(pi*specie.M/1000))**(0.5)
    def dP_dl(self,
              mol_flux:float,
              T:float,
              P:float,
              gas:Mixture)->float:
        """
        Pressure drop along layer thickness [Pa/m]
        
        Parameters
        ----------
        mol_flux : float
            molar flux [mol/m^2]
        T : float
            Temperature [K]
        P : float
            Pressure [Pa]
        gas : Mixture
            Mixture of species

        Notes
        -----
        * Works as a pseudo-abstract method
        """
        return 0

class BinaryFick(PorousTransport):
    """
    Fick diffusion model for binary mixture

    Parameters
    ----------
    dp : float
        average porous diameter [m]
    epsilon : float
        porosity
    tau : float
        tortuosity
    """
    def D_eff(self,
              T:float,
              P:float,
              gas:Mixture)->np.ndarray:
        """
        Effective binary diffusivity [m^2/s^2]
        
        Parameters
        ----------
        T : float
            Temperature [K]
        P : float
            Pressure [Pa]
        gas : Mixture
            Misture of species

        Notes
        -----
        * Bosanquet formula assumes: (i) binary system, (ii) equimolar counter transport,
        (iii) constant pressure [1]

        References
        ----------
        1. https://doi.org/10.1016/j.jpowsour.2016.01.099
        """
        if isinstance(gas.species, Specie):
            return np.array([ self.epsilon/self.tau*self.D_knudsen(T,gas.species) ])
        elif len(gas.species) == 1:
            return np.array([ self.epsilon/self.tau*self.D_knudsen(T,gas.species[0]) ])
        elif len(gas.species) == 2:
            D_ij = gas.D_ij(0,1,T,P)
            return np.array([ 1/((self.epsilon/self.tau*self.D_knudsen(T,gas.species[i]))**(-1)+
                                 (self.epsilon/self.tau*D_ij)**(-1)) for i in range(2)] )
        else:
            raise ValueError("Binary Fick is only valid for pure substance or binary mixture")
    def dP_dl(self,
              mol_flux:float,
              T:float,
              P:float,
              gas:Mixture)->np.ndarray:
        """
        Partial pressure ratio [Pa/m]
        
        Parameters
        ----------
        mol_flux : float : 
            molar flux [mol/m^2]
        T : float
            Temperature [K]
        P : float
            Pressure [Pa]
        gas : Mixture
            Mixture of species

        Notes
        -----
        * This function assumes: (i) no transiency (ii) binary mixture (iii) ideal gas law
        * Function also assumes -mol_flus as boundary condition for molar flux at x = x_max
        """
        return mol_flux/self.D_eff(T,P,gas)*R*T

class Layer:
    """
    Cell layer (i.e., support, electrode, electrolyte)

    Parameters
    ----------
    delta : float
        (initial) layer thickness [m]
    kinetic : Kinetic
        kinetic model
    conductivity : Conductivity
        conductivity model
    transport : PorousTransport
        diffusion mass transfer model
    """
    def __init__(self,
                 delta=0.0,
                 kinetic=Kinetic(),
                 conductivity=Conductivity(),
                 transport=PorousTransport()):
        self.delta = delta
        self.kinetic = kinetic
        self.conductivity = conductivity
        self.transport=transport
    def V_ohm(self,
              j:float,
              T:float,
              **kwargs)->float:
        """
        Ohmic voltage [V]
        
        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            temperature [K]
        delta_extra : float, optional
            increase of layer thickness for ohmic resistance [m]
        """
        delta = self.delta
        if "delta_extra" in kwargs:
            delta += kwargs["delta_extra"]
        return j*delta/self.conductivity.sigma(T)
    def V_act(self,
              j:float,
              T:float,
              Ps=np.array([0]),
              **kwargs)->float:
        """
        Activation voltage [V]
        
        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            temperature [K]
        Ps : numpy.ndarray
            partial pressures [Pa]
        ratio : float, optional
            ratio of active reaction area compared to begining of life [m]
        """
        j_0 = self.kinetic.j_0(T,Ps)
        if "ratio" in kwargs:
            j_0 *= kwargs["ratio"]
        return self.kinetic.V(j,T,j_0)
    def Ps_star(self,
                j,
                T,
                Ps):
        """
        Partial pressures at the reaction site [Pa]

        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            temperature [K]
        Ps : numpy.ndarray
            partial pressure [Pa]
        """
        return Ps + self.transport.dP_dl(self.kinetic.mol_flux(j),
                                        T,sum(Ps),self.kinetic.gas)*self.delta
    def V(self,
          j:float,
          T:float,
          Ps=np.array([0]),
          **kwargs)->float:
        """
        Total layer voltage [V]

        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            Temperature [K]
        Ps : numpy.ndarray
            partial pressures [Pa]
        """
        return self.V_ohm(j,T,**kwargs)+self.V_act(j,T,Ps,**kwargs)

class BiMaterial(Layer):
    """
    Percolation theory for binary mixture materials (i.e., Ni/YSZ)
    
    Parameters
    ----------
    delta : float
        layer thickness (support+functional) [m]
    psi_ed : float
        solid volume fraction of electrode [-]
    psi_el : float
        solid volume fraction of electrolyte [-]
    r0_ed : float
        mean electrode particle radius [m]
    r0_el : float
        mean electrolyte particle radius [m]
    epsilon : float
        porosity of functional layer [-]
    z_bar : float, optional
        average coordination number. Defalt value source: [1]
    theta : float, optional
        particle contact angle [rads]. Default value source: [1]

    Notes
    -----
    * Contact angle needs to be assumed in the order of 15 degrees or less
    in order for percolation theory to be valid [2]
    * Impact on electronic conduction resistance is not modeled

    References
    ----------
    1. https://doi.org/10.1016/j.pecs.2020.100902
    2. https://doi.org/10.1016/j.jpowsour.2011.06.087
    """

    def __init__(self,
                 delta:float,
                 psi_ed:float,
                 psi_el:float,
                 r0_ed:float,
                 r0_el:float,
                 epsilon:float,
                 z_bar=6,
                 theta=15*pi/180,
                 kinetic=Kinetic(),
                 conductivity=Conductivity(),
                 transport=PorousTransport()):
        self.delta = delta
        self.psi_ed = psi_ed
        self.psi_el = psi_el
        self.r0_ed = r0_ed
        self.r0_el = r0_el
        self.epsilon = epsilon
        self.kinetic = kinetic
        self.conductivity = conductivity
        self.transport = transport
        self.z_bar = z_bar
        self.theta = theta
    def z_ii(self,
             psi_i:float,
             psi_j:float,
             r_i:float,
             r_j:float)->float:
        """
        Coordination number between i,i

        Parameters
        ----------
        psi_i : float
            solid volume fraction of i [-]
        psi_j : float
            solid volume fraction of j [-]
        r_i : float
            mean particle radius of i [m]
        r_j : float
            mean particle radius of j [m]

        Reference
        ---------
        1. https://doi.org/10.1016/j.jpowsour.2009.02.051
        """
        return self.z_bar*psi_i/r_i/(psi_i/r_i+psi_j/r_j)
    def z_ij(self,
             psi_i:float,
             psi_j:float,
             r_i:float,
             r_j:float)->float:
        """
        Coordination number between i,j

        Parameters
        ----------
        psi_i : float
            solid volume fraction of i [-]
        psi_j : float
            solid volume fraction of j [-]
        r_i : float
            mean particle radius of i [m]
        r_j : float
            mean particle radius of j [m]

        Reference
        ---------
        1. https://doi.org/10.1016/j.jpowsour.2009.02.051
        """
        return 0.5*(1+r_i**2/r_j**2)*self.z_bar*psi_j/r_j/(psi_i/r_i+psi_j/r_j)
    def p_i(self,
            psi_i:float,
            psi_j:float,
            r_i:float,
            r_j:float)->float:
        """
        Percolation probability of i-particle

        Parameters
        ----------
        psi_i : float
            solid volume fraction of i [-]
        psi_j : float
            solid volume fraction of j [-]
        r_i : float
            mean particle radius of i [m]
        r_j : float
            mean particle radius of j [m]

        Notes
        -----
        * There are multiple equations for this parameter, 
        we are using the one proposed by Ref. [1]

        Reference
        ---------
        1. https://doi.org/10.1016/j.powtec.2011.07.011
        """
        z_ii = self.z_ii(psi_i,psi_j,r_i,r_j)
        p_i = 0
        if z_ii > 4.236:
            p_i = 1
        elif z_ii > 1.764:
            p_i = 1-((4.236-z_ii)/2.472)**3.7
        return p_i
    def l_tpb(self,
              r_ed:float)->float:
        """
        Triple phase boundary length [1/m^2]

        Parameters
        ----------
        r_ed : float
            mean electrode particle radius [m]

        Reference
        ---------
        [1] https://doi.org/10.1016/j.jpowsour.2009.02.051
        """
        z_ed_el = self.z_ij(self.psi_ed,self.psi_el,r_ed,self.r0_el)
        r_c = min(r_ed,self.r0_el)*np.sin(self.theta)
        return (3/2*r_c/r_ed**3*(1-self.epsilon))*self.psi_ed*z_ed_el
    def l_tpb_eff(self,
                  r_ed:float)->float:
        """
        Effective triple phase boundary length [1/m^2]

        Parameters
        ----------
        r_ed : float
            mean electrode particle radius [m]

        Notes
        -----
        * For numerical safety, we are limiting the percolation probability to a min. 1%

        References
        ----------
        1. https://doi.org/10.1016/j.jpowsour.2009.02.051
        """
        p_ed = self.p_i(self.psi_ed,self.psi_el,r_ed,self.r0_el)
        p_el = self.p_i(self.psi_el,self.psi_ed,self.r0_el,r_ed)
        return self.l_tpb(r_ed)*max(p_ed*p_el,0.01)
    def xi_i(self,
             psi_i:float,
             psi_j:float,
             r_i:float,
             r_j:float)->float:
        """
        Number fractions of i

        Parameters
        ----------
        psi_i : float
            solid volume fraction of i [-]
        psi_j : float
            solid volume fraction of j [-]
        r_i : float
            mean particle radius of i [m]
        r_j : float
            mean particle radius of j [m]

        References
        ----------
        1. https://doi.org/10.1016/j.jpowsour.2009.02.051
        """
        return psi_i/r_i**3/(psi_i/r_i**3+psi_j/r_j**3)
    def V_act(self,j:float,T:float,Ps=np.array([0]),**kwargs)->float:
        """ 
        Activation voltage [V]

        Parameters
        ----------
        j : float
            current density [A/m^2]
        T : float
            Temperature [K]
        Ps : numpy.ndarray
            partial pressures [Pa]
        r_ed : float, optional
            mean radius of the electrode particles [m]
        ratio : float, optional
            ratio of active reaction area compared to begining of life [m]
        
        Notes
        -----
        * Mixed ionic conductor electrodes use the contact area instead of TPB
        (feature not implemented)
        """
        j_0 = self.kinetic.j_0(T,Ps)
        if "r_ed" in kwargs:
            j_0 *= self.l_tpb_eff(kwargs["r_ed"])
        if "ratio" in kwargs:
            j_0 *= kwargs["ratio"]
        return self.kinetic.V(j,T,j_0)

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
    def __init__(self,area,electrode_fuel,electrolyte,electrode_air,elements=10):
        self.area = area
        self.electrode_fuel = electrode_fuel
        self.electrolyte = electrolyte
        self.electrode_air = electrode_air
        self.elements = elements
    def V_nerst(self,T,Ps_fuel,Ps_air):
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
        return -(self.electrode_fuel.kinetic.V_nerst_half(T,Ps_fuel)+
                 self.electrode_air.kinetic.V_nerst_half(T,Ps_air))
    def V(self,j,T,Ps_fuel,Ps_air,**kwargs):
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
        Ps_fuel_star = self.electrode_fuel.Ps_star(j,T,Ps_fuel)
        Ps_air_star = self.electrode_air.Ps_star(j,T,Ps_air)
        return (self.V_nerst(T,Ps_fuel_star,Ps_air_star)-
                sum(layer.V(j,T,**kwargs) for layer in self.electrolyte)-
                self.electrode_fuel.V(j,T,Ps=Ps_fuel_star,**kwargs)-
                self.electrode_air.V(j,T,Ps=Ps_air_star,**kwargs))
    def dn_fuel(self,j):
        """
        Net molar flow per element [mol/s] - fuel side

        Parameters
        ----------
        j : float
            current density [A/m^2]
        """
        return self.electrode_fuel.kinetic.mol_flux(j)*self.area/self.elements
    def dn_air(self,j):
        """
        Net molar flow per element [mol/s] - air side

        Parameters
        ----------
        j : float
            current density [A/m^2]
        """
        return self.electrode_air.kinetic.mol_flux(j)*self.area/self.elements