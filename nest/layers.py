"""
Module providing the classes and functions for the material layers in the single repeating unit
"""

import numpy as np
from scipy.optimize import newton
from scipy.integrate import solve_ivp
from scipy.constants import pi

from nest.constants import R, F, P_atm
from nest.properties import Specie, Mixture, BasicSpecies
from nest.degradation import NoDegradation


class Kinetic:
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

    def __init__(
        self,
        gas=BasicSpecies.H2,
        nu=np.array([0]),
        n_e=2,
        alpha=0,
        beta=0,
        gamma=0,
        p=np.array([0]),
        theta=1,
        E_act=0,
    ):
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

    def j_0(self, T: float, Ps: np.ndarray) -> float:
        """
        Exchange current density [A/m^2]

        Parameters
        ----------
        T : float
            Temperature [K]
        Ps : numpy.ndarray
            Partial pressures [Pa]
        """
        return (
            self.gamma
            * T**self.theta
            * np.exp(-self.E_act / (R * T))
            * np.prod((Ps / P_atm) ** self.p)
        )

    def mol_flux(self, j: float) -> np.ndarray:
        """
        Molar flux [mol/m^2]

        Parameters
        ----------
        j : float
            Current density [A/m^2]
        """
        return self.nu * j / (self.n_e * F)

    def V_nerst_half(self, T, Ps):
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
            return self.nu[0] * self.gas.species.g(T, Ps[0]) / (self.n_e * F)
        else:
            return sum(
                [self.nu[i] * gas.g(T, Ps[i]) for i, gas in enumerate(self.gas.species)]
            ) / (self.n_e * F)

    def V(self, j: float, T: float, j0: float) -> float:
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

    def j(self, V: float, T: float, j0: float) -> float:
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
        return j0 * (
            np.exp(self.alpha * self.n_e * F * V / (R * T))
            - np.exp(-self.beta * self.n_e * F * V / (R * T))
        )

    def dj_dV(self, V: float, T: float, j0: float) -> float:
        """
        First derivative of current density by voltage [A/m^2.V]

        Parameters
        ----------
        V (float) : Activation overpotential [V]
        T (float) : Temperature [K]
        j0 (float) : Exchange current density [A/m^2]
        """
        return j0 * (
            self.alpha
            * self.n_e
            * F
            / (R * T)
            * np.exp(self.alpha * self.n_e * F * V / (R * T))
            + self.beta
            * self.n_e
            * F
            / (R * T)
            * np.exp(-self.beta * self.n_e * F * V / (R * T))
        )

    def dj_dV2(self, V: float, T: float, j0: float) -> float:
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
        return j0 * (
            (self.alpha * self.n_e * F / (R * T)) ** 2
            * np.exp(self.alpha * self.n_e * F * V / (R * T))
            - (self.beta * self.n_e * F / (R * T)) ** 2
            * np.exp(-self.beta * self.n_e * F * V / (R * T))
        )

    def V(self, j: float, T: float, j0: float) -> float:
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
            guess = R * T / (self.n_e * F) * (j / j0)
        elif (0.5 <= self.alpha / self.beta <= 1.5) and abs(j / j0) < 3:
            # Sinh approximation: sinh(x) = (exp(x)-exp(-x))/2
            # Comment: only better at limited conditions where alpha/beta approx. 1
            coef = self.alpha if j > 0 else -self.beta
            guess = R * T / (coef * self.n_e * F) * np.arcsinh(abs(j) / j0 / 2)
        else:
            # Taffel approximation
            coef = self.alpha if j > 0 else -self.beta
            guess = R * T / (coef * self.n_e * F) * np.log(abs(j) / j0)
        # Root finding
        return newton(
            lambda x: self.j(x, T, j0) - j,
            guess,
            fprime=lambda x: self.dj_dV(x, T, j0),
            fprime2=lambda x: self.dj_dV2(x, T, j0),
        )


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

    def __init__(self, sigma0=np.inf, E_act=0.0, theta=-1):
        self.sigma0 = sigma0
        self.E_act = E_act
        self.theta = theta

    def sigma(self, T: float) -> float:
        """
        Material conductivity at given temperature (T) [S/m]

        Parameters
        ----------
        T : float
            Temperature [K]
        """
        return self.sigma0 * T**self.theta * np.exp(-self.E_act / (R * T))


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

    def __init__(self, dp=0.0, epsilon=0.0, tau=0.0):
        self.dp = dp
        self.epsilon = epsilon
        self.tau = tau

    def D_knudsen(self, T: float, specie: Specie) -> float:
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
        return self.dp / 3 * (8 * R * T / (pi * specie.M / 1000)) ** (0.5)

    def dP_dl(
        self, mol_flux: np.ndarray, T: float, P: np.ndarray, gas: Mixture, delta: float
    ) -> float:
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
        delta : float
            Thickness of the porous layer [m]

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

    def D_eff(self, T: float, P: float, gas: Mixture) -> np.ndarray:
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
            return np.array([self.epsilon / self.tau * self.D_knudsen(T, gas.species)])
        elif len(gas.species) == 1:
            return np.array(
                [self.epsilon / self.tau * self.D_knudsen(T, gas.species[0])]
            )
        elif len(gas.species) == 2:
            D_ij = gas.D_ij(0, 1, T, P)
            return np.array(
                [
                    1
                    / (
                        (self.epsilon / self.tau * self.D_knudsen(T, gas.species[i]))
                        ** (-1)
                        + (self.epsilon / self.tau * D_ij) ** (-1)
                    )
                    for i in range(2)
                ]
            )
        else:
            raise ValueError(
                "Binary Fick is only valid for pure substance or binary mixture"
            )

    def dP_dl(
        self, mol_flux: np.ndarray, T: float, P: np.ndarray, gas: Mixture, delta: float
    ) -> np.ndarray:
        """
        Partial pressures at the reaction site [Pa]

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
        return P + mol_flux / self.D_eff(T, sum(P), gas) * R * T * delta


class StefanMaxwell(PorousTransport):
    """
    Stefan-Maxwell gas model for multi-component mixture

    Parameters
    ----------
    dp : float
        average porous diameter [m]
    epsilon : float
        porosity
    tau : float
        tortuosity
    """

    def D_eff(self, T: float, P: float, gas: Mixture) -> np.ndarray:
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
        * Equilvalent binary diffusivities are calculated based on the extended Stefan Maxwell model proposed by [1]

        References
        ----------
        1. https://doi.org/10.1016/j.jpowsour.2016.01.099
        """
        if isinstance(gas.species, Specie):
            return np.array([self.epsilon / self.tau * self.D_knudsen(T, gas.species)])
        elif len(gas.species) == 1:
            return np.array(
                [self.epsilon / self.tau * self.D_knudsen(T, gas.species[0])]
            )
        else:
            n_species = len(gas.species)
            return (
                self.epsilon
                / self.tau
                / 2
                * np.array(
                    [
                        [
                            1
                            / (
                                1 / gas.D_ij(i, j, T, P)
                                + 1 / self.D_knudsen(T, gas.species[i])
                            )
                            + 1
                            / (
                                1 / gas.D_ij(i, j, T, P)
                                + 1 / self.D_knudsen(T, gas.species[j])
                            )
                            for j in range(n_species)
                        ]
                        for i in range(n_species)
                    ]
                )
            )

    def dP_dl(
        self, mol_flux: float, T: float, P: np.ndarray, gas: Mixture, delta: float
    ) -> np.ndarray:
        """
        Partial pressures at the reaction site [Pa]

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
        * This function assumes: (i) no transiency (ii) binary mixture (iii) ideal gas law
        * Stefan-Maxwell model for diffusion
        * Note that the molar flux in the diffusion equations is equal to -mol_flux
        """
        P_gas = sum(P)
        D = self.D_eff(T, P_gas, gas)

        def molar_fractions(y, x):
            n_species = len(gas.species)
            dx_dy = np.zeros(n_species)
            for i in range(n_species):
                sum_1 = 0
                sum_2 = 0
                for j in range(n_species):
                    if j != i:
                        sum_1 += x[j] / D[i][j]
                        sum_2 += mol_flux[j] / D[i][j]
                dx_dy[i] = R * T / P_gas * (mol_flux[i] * sum_1 - x[i] * sum_2)
            return dx_dy

        solution = solve_ivp(molar_fractions, (0, delta), P)
        return solution.y[:, -1]


class Layer:
    """
    Cell layer (i.e., support, electrode, electrolyte)

    Parameters
    ----------
    delta : float
        (initial) layer thickness [m]
    kinetic : Kinetic, optional
        kinetic model
    conductivity : Conductivity, optional
        conductivity model
    transport : PorousTransport, optional
        diffusion mass transfer model
    pol_deg : bool, optional
        flags if degradation for polarization resistance is active
    ohm_deg : bool, optional
        flags if degradation for ohmic resistance is active
    """

    def __init__(
        self,
        delta=0.0,
        kinetic=Kinetic(),
        conductivity=Conductivity(),
        transport=PorousTransport(),
        degradation=NoDegradation(),
    ):
        self.delta = delta
        self.kinetic = kinetic
        self.conductivity = conductivity
        self.transport = transport
        self.degradation = degradation

    def V_ohm(self, j: float, T: float, **kwargs) -> float:
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
        if self.degradation.ohm_active:
            delta += self.degradation.ohm_deg(kwargs["ohm_deg"])
        return j * delta / self.conductivity.sigma(T)

    def V_act(self, j: float, T: float, Ps=np.array([0]), **kwargs) -> float:
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
        pol_ratio : float, optional
            ratio of active reaction area compared to begining of life [m]
        """
        j_0 = self.kinetic.j_0(T, Ps)
        if self.degradation.pol_active:
            j_0 *= self.degradation.pol_deg(kwargs["pol_deg"])
        return self.kinetic.V(j, T, j_0)

    def Ps_star(self, j, T, Ps):
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
        return self.transport.dP_dl(
            self.kinetic.mol_flux(j), T, Ps, self.kinetic.gas, self.delta
        )

    def V(self, j: float, T: float, Ps=np.array([0]), **kwargs) -> float:
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
        return self.V_ohm(j, T, **kwargs) + self.V_act(j, T, Ps, **kwargs)
