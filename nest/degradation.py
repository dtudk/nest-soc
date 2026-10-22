import numpy as np
from scipy.constants import pi
from abc import ABC, abstractmethod

from nest.constants import R, F
from nest.boundary import BoundaryData


class Degradation(ABC):
    """
    Abstract class for degradation modeling.
    """

    def __init__(self):
        """
        Here pol_active and ohm_active flag if
        polarization and ohmic degradation are considered, respectively.
        """
        self.pol_active = False
        self.ohm_active = False
        self.m0 = 0

    @abstractmethod
    def material_dt(self, m, state, time, **kwargs):
        """
        Default material degradation ODE.

        Parameters
        ----------
        m : float
            material variable (e.g., Ni radius or oxide scale thickness)
        state : BoundaryData
            state variables
        time : float
            time (h) related to degradation
        kwargs:
            Not used (prevent overflow)

        Returns
        -------
        float
            material degradation rate; 0 by default

        """
        return 0

    @abstractmethod
    def pol_deg(self, m, **kwargs):
        """
        Wrapper function from material to polarization degradation ratio.

        Parameters
        ----------
        m : float
            material variable (e.g., Ni radius or oxide scale thickness)
        kwargs:
            Not used (prevent overflow)

        Returns
        -------
        float
            polarization degradation ratio; 1 (no degradation) by default

        """
        return 1

    @abstractmethod
    def ohm_deg(self, m, **kwargs):
        """
        Wrapper function from material to polarization degradation ratio.

        Parameters
        ----------
        m : float
            material variable (e.g., Ni radius or oxide scale thickness)
        kwargs:
            Not used (prevent overflow)

        Returns
        -------
        float
            ohmic degradation quantity; 0 (no degradation) by default

        """
        return 0


class NoDegradation(Degradation):
    """
    Default degradation class = No degradation.
    """

    def __init__(self):
        self.pol_active = False
        self.ohm_active = False
        self.m0 = 0

    def material_dt(self, m, state, time, **kwargs):
        """
        Default material degradation ODE => returns 0.

        Parameters
        ----------
        m : float
            material variable (e.g., Ni radius or oxide scale thickness)
        state : BoundaryData
            state variables
        time : float
            time (h) related to degradation
        kwargs:
            Not used (prevent overflow)

        Returns
        -------
        int
            material degradation rate; always 0

        """
        return 0

    def pol_deg(self, m, **kwargs):
        """
        Wrapper function from material to polarization degradation ratio
        returns 1 (no degradation).

        Parameters
        ----------
        m : float
            material variable (e.g., Ni radius or oxide scale thickness)
        kwargs:
            Not used (prevent overflow)

        Returns
        -------
        int
            polarization degradation ratio; always 1 (no degradation)

        """
        return 1

    def ohm_deg(self, m, **kwargs):
        """
        Wrapper function from material to ohmic degradation ratio
        returns 0 (no degradation).

        Parameters
        ----------
        m : float
            material variable (e.g., Ni radius or oxide scale thickness)
        kwargs:
            Not used (prevent overflow)

        Returns
        -------
        int
            ohmic degradation ratio; always 0 (no degradation)

        """
        return 0


class NickelAgglomeration(Degradation):
    """
    Degradation model for nickel agglomeration considering
    percolation theory for binary mixture materials (i.e., Ni/YSZ).

    Parameters
    ----------
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
    alpha : float
        pre-exponential factor for Ni agglomeration rate [?]
    r_ed_max : float
        maximum Ni particle radius [m]
    E_act : float
        activation energy for Ni agglomeration [J/mol]
    z_bar : float, optional
        average coordination number. Defalt value source: [1]
    theta : float, optional
        particle contact angle [rads]. Default value source: [1]
    kinetic : Kinetic, optional
        kinetic model
    conductivity : Conductivity, optional
        conductivity model
    transport : PorousTransport, optional
        diffusion mass transfer model
    pol_ratio : bool, optional
        flags if degradation for polarization resistance is active

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

    def __init__(
        self,
        psi_ed: float,
        psi_el: float,
        r0_ed: float,
        r0_el: float,
        epsilon: float,
        alpha: float,
        r_ed_max: float,
        E_act=242e3,
        z_bar=6,
        theta=15 * pi / 180,
        pol_active=True,
    ):
        self.m0 = r0_ed * 1e6  # converting units
        self.psi_ed = psi_ed
        self.psi_el = psi_el
        self.r0_ed = r0_ed
        self.r0_el = r0_el
        self.epsilon = epsilon
        self.alpha = alpha
        self.r_ed_max = r_ed_max
        self.E_act = E_act
        self.z_bar = z_bar
        self.theta = theta
        self.pol_active = pol_active
        self.ohm_active = False

    def z_ii(self, psi_i: float, psi_j: float, r_i: float, r_j: float) -> float:
        """
        Coordination number between i,i.

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

        Returns
        -------
        float
            coordination number between i,i [-]

        Reference
        ---------
        1. https://doi.org/10.1016/j.jpowsour.2009.02.051

        """
        return self.z_bar * psi_i / r_i / (psi_i / r_i + psi_j / r_j)

    def z_ij(self, psi_i: float, psi_j: float, r_i: float, r_j: float) -> float:
        """
        Coordination number between i,j.

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

        Returns
        -------
        float
            coordination number between i,j [-]

        Reference
        ---------
        1. https://doi.org/10.1016/j.jpowsour.2009.02.051

        """
        return (
            0.5
            * (1 + r_i**2 / r_j**2)
            * self.z_bar
            * psi_j
            / r_j
            / (psi_i / r_i + psi_j / r_j)
        )

    def p_i(self, psi_i: float, psi_j: float, r_i: float, r_j: float) -> float:
        """
        Percolation probability of i-particle.

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

        Returns
        -------
        float
            percolation probability of i-particle [-]

        Notes
        -----
        * There are multiple equations for this parameter,
          we are using the one proposed by Ref. [1]

        Reference
        ---------
        1. https://doi.org/10.1016/j.powtec.2011.07.011

        """
        z_ii = self.z_ii(psi_i, psi_j, r_i, r_j)
        p_i = 0
        if z_ii > 4.236:
            p_i = 1
        elif z_ii > 1.764:
            p_i = 1 - ((4.236 - z_ii) / 2.472) ** 3.7
        return p_i

    def l_tpb(self, r_ed: float) -> float:
        """
        Triple phase boundary length [1/m^2].

        Parameters
        ----------
        r_ed : float
            mean electrode particle radius [m]

        Returns
        -------
        float
            triple phase boundary length [1/m^2]

        Notes
        -----
        The neck radius equation here is different from the Ref. [1] which only basis in the "smaller particle".
        The original source, Ref. [2], actually uses the "inclusion particle" as the radius of reference, which is the Ni in this case.
        This is important in the cases that Ni may grow lager than the initial size of YSZ, making the degradation model suddenly change from the "min" function.

        Reference
        ---------
        [1] https://doi.org/10.1016/j.jpowsour.2009.02.051
        [2] https://doi.org/10.1016/S0013-4686(97)00063-7

        """
        z_ed_el = self.z_ij(self.psi_ed, self.psi_el, r_ed, self.r0_el)
        r_c = r_ed * np.sin(self.theta)
        return (3 / 2 * r_c / r_ed**3 * (1 - self.epsilon)) * self.psi_ed * z_ed_el

    def l_tpb_eff(self, r_ed: float) -> float:
        """
        Effective triple phase boundary length [1/m^2].

        Parameters
        ----------
        r_ed : float
            mean electrode particle radius [m]

        Returns
        -------
        float
            effective triple phase boundary length [1/m^2]

        Notes
        -----
        * For numerical safety, we are limiting the percolation probability to a min. 1%

        References
        ----------
        1. https://doi.org/10.1016/j.jpowsour.2009.02.051

        """
        p_ed = self.p_i(self.psi_ed, self.psi_el, r_ed, self.r0_el)
        p_el = self.p_i(self.psi_el, self.psi_ed, self.r0_el, r_ed)
        return self.l_tpb(r_ed) * max(p_ed * p_el, 0.01)

    def xi_i(self, psi_i: float, psi_j: float, r_i: float, r_j: float) -> float:
        """
        Number fractions of i.

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

        Returns
        -------
        float
            number fraction of i [-]

        References
        ----------
        1. https://doi.org/10.1016/j.jpowsour.2009.02.051

        """
        return psi_i / r_i**3 / (psi_i / r_i**3 + psi_j / r_j**3)

    def material_dt(
        self, m: float, state: BoundaryData, time: float, **kwargs
    ) -> float:
        """
        Ni radius rate of change [μm/h].

        Parameters
        ----------
        m : float
            mean Ni radius [μm]
        state : BoundaryData
            state variables
        time : float
            time [h]
        kwargs:
            Not used (prevent overflow)

        Returns
        -------
        float
            Ni radius rate of change [μm/h]

        Notes
        -----
        * It assumes that that Ps_fuel = [P_H2, P_H2O] in this order
        * It assumes that the contact angle YSZ_YSZ is equal to Ni_YSZ.
        * In theory, both the support layer and the functional layer suffers Ni aglomeration. Here the agglomeration in the functional layer is modeled
        * The model here is based on Ref. [1]

        References
        ----------
        1. https://doi.org/10.1016/j.enconman.2021.113902

        """
        r = m
        # Partial pressures at TPB
        Ps_fuel_star = state.Ps_fuel()
        # Percolation theory varibles
        xi_ed = (
            self.psi_ed
            / (r / 1e6) ** 3
            / (self.psi_el / self.r0_el**3 + self.psi_ed / (r / 1e6) ** 3)
        )
        xi_el = (
            self.psi_el
            / self.r0_el**3
            / (self.psi_el / self.r0_el**3 + self.psi_ed / (r / 1e6) ** 3)
        )
        Z_el_el = self.z_ii(self.psi_el, self.psi_el, self.r0_el, self.r0_el)
        Z_el_ed = self.z_ij(self.psi_el, self.psi_ed, self.r0_el, r / 1e6)
        n_tot = (1 - self.epsilon) / (
            4 / 3 * pi * self.r0_el**3 * xi_el + 4 / 3 * pi * (r / 1e6) ** 3 * xi_ed
        )
        s_el = (
            2
            * pi
            * self.r0_el**2
            * (2 - (1 - np.cos(self.theta)) * (Z_el_el + Z_el_ed))
        )
        A_el = s_el * n_tot * xi_el / 1e6  # 1/m * m/1E6 microM
        # Rate equation
        k = (
            self.alpha
            / 16
            * self.psi_ed
            / (1 - self.psi_ed)
            / A_el
            * (Ps_fuel_star[1] / 1e5)
            / (Ps_fuel_star[0] / 1e5) ** 0.5
        )
        if r > self.r_ed_max * 1e6:
            return 0
        else:
            return (
                k * np.exp(-self.E_act / (R * state.T)) / (r * 2) ** 7
            )  # unit conversion

    def pol_deg(self, m, **kwargs):
        """
        Returns the polarization degradation ratio
        In this case, actually the effective TPB length [1/m^2].

        Parameters
        ----------
        m : float
            Nickel radius in micro meter [μm]
        kwargs:
            Not used (prevent overflow)

        Returns
        -------
        float
            effective triple phase boundary length [1/m^2]

        """
        return self.l_tpb_eff(m / 1e6)

    def ohm_deg(self, *args, **kwargs):
        """
        Returns null, as no ohmic degradation is considered.

        Returns
        -------
        int
            ohmic degradation ratio; always 0

        """
        return 0


class NickelMigration(Degradation):
    """
    Degradation model for Ni migration in the fuel electrode

    Ni particles migrate away from the fuel electrode/electrolyte interface,
    leaving a Ni-depleted zone that conducts only ions and adds ohmic
    resistance (like a thicker electrolyte). The Ni migration rate can depend
    on temperature, current density, and fuel electrode overpotential, 
    or on only some of them, depending on the case. 
    Once a threshold is reached, the migration can follow a different rate.
    The threshold can also depend on T, j, eta.

    Parameters
    ----------
    rate : callable
        rate(T, j, eta) -> growth rate of the Ni-depleted zone thickness [m/h],
        with T [K], j [A/m^2], and eta [V].
        All three are always passed; a rate that does not depend on one of them is ignored.
    threshold : callable, optional
        threshold(T, j, eta) -> True when the threshold is reached,
        All three are always passed, as for `rate`
    rate_after_threshold : callable, optional
        same as `rate`; used once the threshold is reached
    ohm_active : bool, optional
        activates the ohmic degradation modeling in simulation
    """

    def __init__(self, rate, threshold=None, rate_after_threshold=None, ohm_active=True):
        self.m0 = 0
        self.rate = rate
        self.threshold = threshold
        self.rate_after_threshold = rate_after_threshold
        self.pol_active = False
        self.ohm_active = ohm_active

    def threshold_reached(self, T: float, j: float, eta: float) -> bool:
        """
        Flags if the threshold is reached

        Parameters
        ----------
        T : float
            temperature [K]
        j : float
            current density [A/m^2]
        eta : float
            fuel electrode overpotential [V]

        Returns
        -------
        bool
            True if the threshold is reached; always False if no threshold is set
        """
        if self.threshold is None or self.rate_after_threshold is None:
            return False
        return bool(self.threshold(T, j, eta))

    def material_dt(self, m, state, time, **kwargs):
        """
        Ni-depleted zone growth rate [m/h]

        Parameters
        ----------
        m : float
            Ni-depleted zone thickness [m]
        state : BoundaryData
            state variables
        time : float
            time (h) related to degradation
        **kwargs
            must contain ``eta_fuel``, the fuel electrode overpotential [V],
            calculated by the cell model

        Returns
        -------
        float
            Ni-depleted zone growth rate [m/h]
        """
        eta_fuel = kwargs["eta_fuel"]
        if self.threshold_reached(state.T, state.j, eta_fuel):
            return self.rate_after_threshold(state.T, state.j, eta_fuel)
        return self.rate(state.T, state.j, eta_fuel)

    def pol_deg(self, *args, **kwargs):
        """
        Returns null, as no polarization degradation is considered

        Returns
        -------
        int
            polarization degradation ratio; always 0
        """
        return 0

    def ohm_deg(self, m, **kwargs):
        """
        Returns the Ni-depleted zone thickness [m]

        Parameters
        ----------
        m : float
            Ni-depleted zone thickness [m]
        **kwargs
            unused

        Returns
        -------
        float
            Ni-depleted zone thickness [m]
        """
        return m


class InterconnectOxidation(Degradation):
    """
    Degradation model for interconnect corrosion.

    Parameters
    ----------
    k_mg : float
        oxidation rate [m^2/h]
    E_ox : float
        activation energy of the oxidation rate [J/mol.K]
    ohm_active : bool, optional
        activates the ohmic degradation modeling in simulation

    """

    def __init__(self, k_mg, E_ox, ohm_active=True):
        self.m0 = 0
        self.k_mg = k_mg
        self.E_ox = E_ox
        self.pol_active = False
        self.ohm_active = ohm_active

    def material_dt(self, m, state, time, **kwargs):
        """
        Interconnect oxidation rate [Wagner's theory].

        Parameters
        ----------
        m : float
            interconnect oxidation layer thickness squared [m^2]
        state : BoundaryData
            state variables
        time : float
            time (h) related to degradation
        kwargs:
            Not used (prevent overflow)

        Returns
        -------
        float
            rate of change of oxidation layer thickness squared [m^2/h]

        """
        return self.k_mg * np.exp(-self.E_ox / (R * state.T))

    def pol_deg(self, *args, **kwargs):
        """
        Returns null, as no polarization degradation is considered.

        Parameters
        ----------
        args:
            TODO
        kwargs:
            Not used (prevent overflow)

        Returns
        -------
        int
            polarization degradation ratio; always 0

        """
        return 0

    def ohm_deg(self, m, **kwargs):
        """
        Returns the interconnect oxidation layer thickness [m].

        Parameters
        ----------
        m : float
            interconnect oxidation layer thickness squared [m^2]
        kwargs:
            Not used (prevent overflow)

        Returns
        -------
        float
            interconnect oxidation layer thickness [m]

        """
        return m**0.5


class ChromiumPoison(Degradation):
    """
    Degradation model for chromium poisoning of air electrodes.

    Parameters
    ----------
    x_H2O : float
        water molar content in air stream
    j0 : float
        exchange current density for chromium deposition [A]
    pol_active : bool
        flags if degradation for polarization resistance is active

    """

    def __init__(self, x_H2O: float, j0: float, E_act: float, pol_active=True):
        self.m0 = 1  # polarization degradation ratio [1 = begining of life]
        self.x_H2O = x_H2O
        self.j0 = j0
        self.E_act = E_act
        self.pol_active = pol_active
        self.ohm_active = False

    def material_dt(self, m, state, time, **kwargs):
        """
        Chromium poisoning rate.

        Parameters
        ----------
        m : float
            polarization degradation ratio [1 = begining of life]
        state : BoundaryData
            layer state conditions (voltage = activation voltage)
        time : float
            time [h]
        kwargs:
            Not used (prevent overflow)


        Returns
        -------
        float
            polarization degradation ratio rate of change [1/h]

        """
        P_CrO2 = (
            2.26e-2 * (state.P * self.x_H2O) ** 0.992 * np.exp(-6.7e4 / (R * state.T))
        )
        h_TPB = 35e-9  # m
        M_Cr2O3 = 151.99  # A/m^2
        rho_Cr2O3 = 5.22 / (1e-2) ** 3  # g/m^3
        j = (
            self.j0
            * np.exp(-self.E_act / (R * state.T))
            * (P_CrO2 / state.P) ** 0.5
            * self.x_H2O**0.5
            * 2
            * np.sinh(F / (2 * R * state.T) * state.V)
        )
        return (
            -M_Cr2O3 / (6 * F * rho_Cr2O3 * h_TPB) * j * m * 3600
        )  # g/mol * mom/(A*s)

    def ohm_deg(self, m, **kwargs):
        """
        Returns null, as no ohmic degradation is considered.

        Returns
        -------
        int
            ohmic degradation ratio; always 0

        """
        return 0

    def pol_deg(self, m, **kwargs):
        """
        Returns the polarization degradation ratio
        In this case is just identity function.

        Parameters
        ----------
        m : float
            polarization degradation ratio [1 = begining of life]
        kwargs:
            Not used (prevent overflow)

        Returns
        -------
        float
            polarization degradation ratio (equal to `m`) [-]

        """
        return m
