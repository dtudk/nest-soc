"""
Module providing the classes and functions for the material layers in the single repeating unit
"""

import numpy as np
from scipy.optimize import newton
from scipy.constants import pi

from nest.ideal_gas import Specie,Mixture,BasicSpecies

class Kinetic:
    """Electrochemical half-reaction - Default : No reaction
    
    Parameters:
    mix (Mixture) : species that are produced/consumed in reaction (no ions)
    coeff (numpy.ndarray) : stoichoimetric coefficients of reaction
    n_e (float) : number of electron moles transfered in half-reaction 
    alpha (float) : charge transfer coefficient - forward reaction
    beta (float) : charge transfer coefficient - reverse reaction
    gamma (float) : pre-exponential kinetic factor [A]
    p (numpy.ndarray) : exponential factors for partial pressure dependency
    theta : exponential factor for Arrhenius equation
    E_act : activation energy for Arrhenius equation
    
    Returns:
    Kinetic : representation of electrochemical half-reaction

    Comments:
    * General exchange current density equation from Ref. [1]
    * The half-reaction is implicitly approximated as an heteregenous reaction happening in 
      the interface of electrode/electrolyte by taking the area specific exchange current density 
      formulation [A/m2] a discussion on the differences between heterogenous and homogeneous 
      treatment is given in Ref. [1]
    * Molar flux assumes faraday law without current losses.
    * Nernst equation for half-reaction voltages.

    References:
    [1] https://doi.org/10.1016/j.pecs.2020.100902
    
    """
    def __init__(self,
                 mix=Mixture((BasicSpecies.H2,BasicSpecies.O2)),
                 coeff=np.array([0,0]),
                 n_e=2,
                 alpha=0.5,
                 beta=0.5,
                 gamma=0,
                 p=np.array([0,0]),
                 theta=1,
                 E_act=0):
        # Related to reaction balance
        self.mix = mix
        self.coeff = coeff
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
        """ Exchange current density [A/m^2]

        Parameters:
        T (float) : Temperature [K]
        Ps (numpy.ndarray) : Partial pressures [Pa]
        
        Returns:
        float : Exchange current density [A/m^2]
        """
        R = 8.314462618    # J/mol.K - Source : https://physics.nist.gov/cuu/Constants/
        P_0 = 101325    # Reference pressure [Pa] - Source: [1]
        return self.gamma*T**self.theta*np.exp(-self.E_act/(R*T))*np.prod((Ps/P_0)**self.p)
    def mol_flux(self,
                 j:float)->float:
        """ Molar flux [mol/m^2]

        Parameters:
        j (float) : Current density [A/m^2]

        Returns:
        float : Molar flux [mol/s]
        """
        F = 96485.33212    # s.A/mol - Source: https://physics.nist.gov/cuu/Constants/
        return self.coeff*j/(self.n_e*F)
    def V_nerst_half(self,
                     T,
                     Ps):
        """ Electric potential difference for half reaction [V]

        Parameters:
        T (float) : Temperature [K]
        Ps (numpy.ndarray) : Partial pressures

        Returns:
        float : Electric potential difference for half reaction [V]
        """
        F = 96485.33212    # s.A/mol - Source: https://physics.nist.gov/cuu/Constants/
        if isinstance(self.mix.species, Specie):
            return self.coeff[0]*self.mix.species.g(T,Ps[0])/(self.n_e*F)
        else:
            return sum([self.coeff[i]*gas.g(T,Ps[i]) for i,gas
                        in enumerate(self.mix.species)])/(self.n_e*F)
    def V(self,
          *args)->float:
        """ Defalt voltage function

        Returns:
        float : Zero potential difference as defalt [V]
        """
        return 0
class ButlerVolmer(Kinetic):
    """ Butler-Volmer half-reaction kinetics

    Parameters:
    mix (Mixture) : species that are produced/consumed in reaction (no ions)
    coeff (numpy.ndarray) : stoichoimetric coefficients of reaction
    n_e (float) : number of electron moles transfered in half-reaction 
    alpha (float) : charge transfer coefficient - forward reaction
    beta (float) : charge transfer coefficient - reverse reaction
    gamma (float) : pre-exponential kinetic factor [A]
    p (numpy.ndarray) : exponential factors for partial pressure dependency
    theta : exponential factor for Arrhenius equation
    E_act : activation energy for Arrhenius equation [J/mol.K]
    
    Returns:
    Kinetic : representation of electrochemical half-reaction

    Comments:
    Buttler-Volmer is not a theoretically rigourous model in most cases, 
    a brief explanation is given in Ref. [1]

    References:
    [1] https://doi.org/10.1016/j.pecs.2020.100902
    """
    def j(self,
          V:float,
          T:float,
          j0:float)->float:
        """ Butler-Volmer equation

        Parameters:
        V (float) : Activation overpotential [V]
        T (float) : Temperature [K]
        j0 (float) : Exchange current density [A/m^2]

        Returns:
        float : Current density [A/m^2]
        """
        F = 96485.33212    # s.A/mol - Source: https://physics.nist.gov/cuu/Constants/
        R = 8.314462618    # J/mol.K - Source: https://physics.nist.gov/cuu/Constants/
        return j0*(np.exp(self.alpha*self.n_e*F*V/(R*T))-np.exp(-self.beta*self.n_e*F*V/(R*T)))
    def dj_dV(self,
              V:float,
              T:float,
              j0:float)->float:
        """ First derivative of Butler-Volmer equation

        Parameters:
        V (float) : Activation overpotential [V]
        T (float) : Temperature [K]
        j0 (float) : Exchange current density [A/m^2]

        Returns:
        float : dj/dV [A/m^2.V]
        """
        F = 96485.33212    # s.A/mol - Source: https://physics.nist.gov/cuu/Constants/
        R = 8.314462618    # J/mol.K - Source: https://physics.nist.gov/cuu/Constants/
        return j0*(self.alpha*self.n_e*F/(R*T)*np.exp(self.alpha*self.n_e*F*V/(R*T))+
                   self.beta*self.n_e*F/(R*T)*np.exp(-self.beta*self.n_e*F*V/(R*T)))
    def dj_dV2(self,
               V:float,
               T:float,
               j0:float)->float:
        """ Second derivative of Butler-Volmer equation

        Parameters:
        V (float) : Activation overpotential [V]
        T (float) : Temperature [K]
        j0 (float) : Exchange current density [A/m^2]

        Returns:
        float : d2j/d2V [A/m^2.V^2]
        """
        F = 96485.33212    # s.A/mol - Source: https://physics.nist.gov/cuu/Constants/
        R = 8.314462618    # J/mol.K - Source: https://physics.nist.gov/cuu/Constants/
        return j0*((self.alpha*self.n_e*F/(R*T))**2*np.exp(self.alpha*self.n_e*F*V/(R*T))-
                   (self.beta*self.n_e*F/(R*T))**2*np.exp(-self.beta*self.n_e*F*V/(R*T)))
    def V(self,
          j:float,
          T:float,
          j0:float)->float:
        """ Activation voltage 

        Parameters:
        j (float) : Activation overpotential [V]
        T (float) : Temperature [K]
        j0 (float) : Exchange current density [A/m^2]

        Returns:
        float : Activation voltage [V]

        Comments:
        * Linear, Sinh and Tafel approximations come from the mathematical approximations of 
          BV under restriced conditions. 
        * This function is not compatible with JAX package, because of the use of 
          scipy.optimize.newton function
        """
        F = 96485.33212    # s.A/mol - Source: https://physics.nist.gov/cuu/Constants/
        R = 8.314462618    # J/mol.K - Source: https://physics.nist.gov/cuu/Constants/
        # Finding V guess
        if abs(j) <= j0:
            # Linear approximation: https://doi.org/10.1016/j.pecs.2020.100902
            guess = R*T/(self.n_e*F)*(j/j0)
        elif (0.5 <= self.alpha/self.beta <= 1.5) and abs(j/j0) < 3:
            # Sinh approximation: sinh(x) = (exp(x)-exp(-x))/2
            # Comment: only better at limited conditions where alpha/beta approx. 1
            coef = self.alpha if j > 0 else -self.beta
            guess = (R*T/(coef*self.n_e*F)*np.arcsinh(abs(j)/j0/2))
        else:
            # Taffel approximation: https://doi.org/10.1016/j.pecs.2020.100902
            coef = self.alpha if j > 0 else -self.beta
            guess = R*T/(coef*self.n_e*F)*np.log(abs(j)/j0)
        # Root finding
        return newton(lambda x:self.j(x,T,j0)-j,guess,fprime=lambda x:self.dj_dV(x,T,j0),
                      fprime2=lambda x:self.dj_dV2(x,T,j0))
    # To-do: dC/dt which adds an extra current because of the charge state and overpotential

class Conductivity:
    """Generic conductivity model - Default : no resistance.
    
    Parameters:
    sigma0 (float) : Conductivity at reference state [S/m]
    E_act (float) : Activation energy [J/mol.K]
    theta (float) : Temperature exponential coefficient

    Returns:
    (float) : Conductivity model
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
        """ Material conductivity at given temperature

        Parameters:
        T (float) : Temperature [K]

        Returns:
        float : Conductivity [S/m]
        """
        R = 8.314462618    # J/mol.K - Source: https://physics.nist.gov/cuu/Constants/
        return self.sigma0*T**self.theta*np.exp(-self.E_act/(R*T))
class PorousTransport:
    """Generic diffusion mass transport over porous media
    
    Parameters:
    dp (float) : average porous diameter [m]
    epsilon (float) : porosity
    tau (float) : tortuosity

    Returns:
    PorousTransport: generic diffusion mass transport
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
        """ Knudsen diffusion coefficient [m^2/s^2]

        Parameters:
        T : Temperature [K]
        specie : Thermodynamic specie model

        Returns:
        float : Knudsen diffusion coefficient [m^2/s^2]
        
        References:
        [1] https://doi.org/10.1016/j.pecs.2020.100902
        """
        R = 8.314462618    # J/mol.K - Source: https://physics.nist.gov/cuu/Constants/
        return self.dp/3*(8*R*T/(pi*specie.M/1000))**(0.5)
    def dP_dl(self,
              *args)->float:
        """
        Defalt pressure drop along layer thickness

        Returns:
        float : Zero pressure drop [Pa]
        """
        return 0

class BinaryFick(PorousTransport):
    """
    Special class for binary Fick diffusion
    """
    def D_eff(self,
              T:float,
              P:float,
              mix:Mixture)->np.ndarray:
        """ Effective binary diffusivity [m^2/s^2]
        
        Parameters:
        T (float) : Temperature [K]
        P (float) : Pressure [Pa]
        mix (Mixture) : Misture of species

        Comments:
        Bosanquet formula assumes: (i) binary system, (ii) equimolar counter transport,
        (iii) constant pressure [1]

        References:
        [1] https://doi.org/10.1016/j.jpowsour.2016.01.099
        """
        if isinstance(mix.species, Specie):
            return np.array([ self.epsilon/self.tau*self.D_knudsen(T,mix.species) ])
        elif len(mix.species) == 1:
            return np.array([ self.epsilon/self.tau*self.D_knudsen(T,mix.species[0]) ])
        elif len(mix.species) == 2:
            D_ij = mix.D_ij(0,1,T,P)
            return np.array([ 1/((self.epsilon/self.tau*self.D_knudsen(T,mix.species[i]))**(-1)+
                                 (self.epsilon/self.tau*D_ij)**(-1)) for i in range(2)] )
        else:
            raise ValueError("Binary Fick is only valid for pure substance or binary mixture")
    def dP_dl(self,
              mol_flus:float,
              T:float,
              P:float,
              mix:Mixture)->np.ndarray:
        """ Fick's first law of diffusion - Pressure
        
        Parameters:
        mol_flus (float) : molar flux [mol/m^2]
        T (float) : Temperature [K]
        P (float) : Pressure [Pa]
        mix (Mixture) : Mixture of species

        Returns:
        numpy.ndarray : pressure variation rates

        Comments:
        * This function assumes: (i) no transiency (ii) binary mixture (iii) ideal gas law
        * Function also assumes -mol_flus as boundary condition for molar flux at x = x_max
        """
        R = 8.314462618    # J/mol.K - Source: https://physics.nist.gov/cuu/Constants/
        return mol_flus/self.D_eff(T,P,mix)*R*T
    # To-do: dP/dt which takes P_t, and two boundary conditions (P_bulk, mol_flus)
    #  to calculate P_t+dt : mass-transport transiency
class Layer:
    """ Cell layer representation

    Parameters:
    delta (float) : layer thickness [m]
    kinetic (Kinetic) : kinetic model
    conductivity (Conductivity) : conductivity model
    transport (PorousTransport) : diffusion mass transfer model

    Returns:
    Layer: cell layer representation

    Comments:
    * delta reflects the initial thickness of the layer, as this may change because of degradation.
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
              T:float)->float:
        """ Ohmic voltage [V]
        
        Parameters:
        j (float) : current density [A/m^2]
        T (float) : temperature [K]

        Returns:
        float : Ohmic voltage contribution [V]
        """
        return j*self.delta/self.conductivity.sigma(T)
    def V_act(self,
              j:float,
              T:float,
              Ps=np.array([0]),
              **kwargs)->float:
        """ Activation voltage [V]
        
        Parameters:
        j (float) : current density [A/m^2]
        T (float) : temperature [K]
        Ps (numpy.ndarray) : partial pressures [Pa]

        Returns:
        float : Activation voltage contribution [V]
        """
        return self.kinetic.V(j,T,self.kinetic.j_0(T,Ps))
    def Ps_star(self,
                j,
                T,
                Ps):
        """Partial pressures at the reaction site [Pa]

        Parameters:
        j (float) : current density [A/m^2]
        T (float) : Temperature [K]
        Ps (numpy.ndarray) : partial pressure [Pa]

        Returns:
        float : Partial pressures at the reaction site [Pa]
        """
        return Ps + self.transport.dP_dl(self.kinetic.mol_flux(j),
                                        T,sum(Ps),self.kinetic.mix)*self.delta
    def V(self,
          j:float,
          T:float,
          Ps=np.array([0]),**kwargs)->float:
        """ Total layer voltage [V]

        Parameters:
        j (float) : current density [A/m^2]
        T (float) : Temperature [K]
        Ps (numpy.ndarray) : partial pressures [Pa]
        
        Returns:
        float : Total layer voltage (ohmic+activation)
        """
        return self.V_ohm(j,T)+self.V_act(j,T,Ps,**kwargs)
