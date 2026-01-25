"""
Thermodynamic and transport properties for ideal gases
"""
import numpy as np
from nest.constants import R, P_0

class Specie:
    """
    Ideal gas specie properties

    Parameters
    ----------
    cs : tuple[numpy.ndarray]
        NASA coefficients and temperature limits from Ref. [1]
    M : float
        molar mass [g/mol]
    V : float 
        special atomic diffusion volumes [cm^3]

    Notes
    -----
    * Molar mass and thermodynamic properties from Ref. [1]
    * Gas constant value from Ref. [1], different from NIST
    * Reference pressure (1 bar) from Ref. [1]

    References
    ----------
    1. https://ntrs.nasa.gov/citations/20020085330
    """
    def __init__(self,
                 cs:tuple[np.ndarray],
                 M:float,
                 V:float):
        self.cs = cs
        self.M = M
        self.V = V
    def a(self,
          T:float)->np.ndarray:
        """ 
        NASA coefficients for given temperature (T)

        Parameters
        ----------
        T : float
            Temperature [K]
        """
        for c in self.cs:
            if c[9] <= T <= c[10]:
                return c
        raise ValueError("Undefined coefficients for specified temperature")
    def cp(self,
           T:float)->float:
        """ 
        Molar specific heat at constant pressure [J/mol.K]

        Parameters
        ----------
        T : float
            Temperature [K]
        """
        a = self.a(T)
        return R*(a[0]*T**(-2)+a[1]*T**(-1)+a[2]+a[3]*T+a[4]*T**2+a[5]*T**3+a[6]*T**4)
    def h(self,
          T:float)->float:
        """
        Molar specific enthalpy [J/mol]

        Parameters
        ----------
        T : float
            Temperature [K]
        """
        a = self.a(T)
        return R*(-a[0]*T**(-1)+a[1]*np.log(T)+a[2]*T+a[3]*T**2/2+a[4]*T**3/3+
                       a[5]*T**4/4+a[6]*T**5/5+a[7])
    def s(self,
          T:float,
          P:float)->float:
        """ 
        Molar specific entropy [J/mol.K]

        Parameters:
        -----------
        T : float
            Temperature [K]
        P : float
            Absolute pressure [Pa]
        """
        if P == 0:
            return 0
        else:
            a = self.a(T)
            return R*(-a[0]*T**(-2)/2-a[1]*T**(-1)+a[2]*np.log(T)+a[3]*T+a[4]*T**2/2+
                      a[5]*T**3/3+a[6]*T**4/4+a[8])-R*np.log(P/P_0)
    def g(self,
          T:float,
          P:float)->float:
        """ 
        Molar specific gibbs free energy [J/mol.K]

        Parameters
        ----------
        T : float
            Temperature [K]
        P : float
            Absolute pressure [Pa]
        """
        if P == 0:
            return 0
        else:
            return self.h(T)-T*self.s(T,P)

class Mixture:
    """
    Ideal gas mixture properties

    Parameters
    ----------
    Species : tuple[Specie]
        species in mixture
    
    Notes
    -----
    * Binary diffusivities from Ref. [1]

    References
    ----------
    1. https://doi.org/10.1021/j100845a020    
    """
    def __init__(self,
                 species:tuple[Specie]):
        self.species = species
        if isinstance(species, Specie):
            self.M_ij = 0
        elif len(species) == 1:
            self.M_ij = 0
        else:
            self.M_ij = np.array([[1/(s_i.M**(-1)+s_j.M**(-1)) 
                                   for s_i in species] 
                                   for s_j in species])
    def D_ij(self,
             i:int,
             j:int,
             T:float,
             P:float)->float:
        """
        Binary diffusivities [m^2/s^2]

        Parameters
        ----------
        i : int
            index of first specie in mixture
        j : int
            index of second specie in mixture
        T : float
            Temperature [K]
        P : float
            Absolute pressure [Pa]
        """
        return 1.01325E-2*T**1.75/(P*self.M_ij[i,j]**0.5*(self.species[i].V**(1/3)+
                                                          self.species[j].V**(1/3))**2)
    def cp(self,
           T:float,
           xs:np.ndarray)->float:
        """
        Molar specific heat at constant pressure [J/mol.K]

        Parameters
        ----------
        T : float
            Temperature [K]
        xs : numpy.ndarray
            Molar fractions
        """
        return sum(np.array([xs[i]*specie.cp(T) for i,specie in enumerate(self.species)]))
    def h(self,
          T:float,
          xs:np.ndarray)->float:
        """
        Molar specific enthalpy [J/mol]

        Parameters
        ----------
        T : float
            Temperature [K]
        xs : numpy.ndarray
            Molar fractions
        """
        return sum(np.array([xs[i]*specie.h(T) for i,specie in enumerate(self.species)]))
    def s(self,
          T:float,
          P:float,
          xs:np.ndarray)->float:
        """
        Molar specific entropy [J/mol.K]

        Parameters
        ----------
        T : float
            Temperature [K]
        P : float
            Absolute pressure [Pa]
        xs : numpy.ndarray
            Molar fractions
        """
        return sum(np.array([xs[i]*specie.s(T,P*xs[i]) for i,specie in enumerate(self.species)]))
    def g(self,
          T:float,
          P:float,
          xs:np.ndarray)->float:
        """
        Molar specific free gibbs energy [J/mol.K]
        
        Parameters
        ----------
        T : float
            Temperature [K]
        P : float
            Absolute pressure [Pa]
        xs : numpy.ndarray
            Molar fractions [-]
        """
        return sum(np.array([xs[i]*specie.g(T,P*xs[i]) for i,specie in enumerate(self.species)]))

class BasicSpecies:
    """
    Gas species for SOC modelling (i.e., H2, H2O, O2, N2)

    Notes
    -----
    * Coefficients from Ref. [1]
    
    References
    ----------
    1. https://ntrs.nasa.gov/citations/20020085330
    """
    H2 = Specie((np.array([4.078323210E+04,-8.009186040E+02,8.214702010E+00,
                           -1.269714457E-02,1.753605076E-05,-1.202860270E-08,
                           3.368093490E-12,2.682484665E+03,-3.043788844E+01,200,1e3]),
                 np.array([5.608128010E+05,-8.371504740E+02,2.975364532E+00,
                           1.252249124E-03,-3.740716190E-07,5.936625200E-11,
                           -3.606994100E-15,5.339824410E+03,-2.202774769E+00,1e3,6e3])),
                           2.01588,6.12)
    H2O = Specie((np.array([-3.947960830E+04,5.755731020E+02,9.317826530E-01,
                            7.222712860E-03,-7.342557370E-06,4.955043490E-09,
                            -1.336933246E-12,-3.303974310E+04,1.724205775E+01,200,1e3]),
                  np.array([1.034972096E+06,-2.412698562E+03,4.646110780E+00,
                            2.291998307E-03,-6.836830480E-07,9.426468930E-11,
                            -4.822380530E-15,-1.384286509E+04,-7.978148510E+00,1e3,6e3])),
                            18.01528,13.1)
    O2 = Specie((np.array([-3.425563420E+04,4.847000970E+02,1.119010961E+00,
                           4.293889240E-03,-6.836300520E-07,-2.023372700E-09,
                           1.039040018E-12,-3.391454870E+03,1.849699470E+01,200,1e3]),
                 np.array([-1.037939022E+06,2.344830282E+03,1.819732036E+00,
                           1.267847582E-03,-2.188067988E-07,2.053719572E-11,
                           -8.193467050E-16,-1.689010929E+04,1.738716506E+01,1e3,6e3])),
                           31.99880,16.3)
    N2 = Specie((np.array([2.210371497E+04,-3.818461820E+02,6.082738360E+00,
                           -8.530914410E-03,1.384646189E-05,-9.625793620E-09,
                           2.519705809E-12,7.108460860E+02,-1.076003744E+01,200,1e3]),
                 np.array([5.877124060E+05,-2.239249073E+03,6.066949220E+00,
                           -6.139685500E-04,1.491806679E-07,-1.923105485E-11,
                           1.061954386E-15,1.283210415E+04,-1.586640027E+01,1e3,6e3])),
                           28.01340,18.5)
