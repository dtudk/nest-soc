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

    def __init__(self, cs: tuple[np.ndarray], M: float, V: float):
        self.cs = cs
        self.M = M
        self.V = V

    def a(self, T: float) -> np.ndarray:
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

    def cp(self, T: float) -> float:
        """
        Molar specific heat at constant pressure [J/mol.K]

        Parameters
        ----------
        T : float
            Temperature [K]
        """
        a = self.a(T)
        return R * (
            a[0] * T ** (-2)
            + a[1] * T ** (-1)
            + a[2]
            + a[3] * T
            + a[4] * T**2
            + a[5] * T**3
            + a[6] * T**4
        )

    def h(self, T: float) -> float:
        """
        Molar specific enthalpy [J/mol]

        Parameters
        ----------
        T : float
            Temperature [K]
        """
        a = self.a(T)
        return R * (
            -a[0] * T ** (-1)
            + a[1] * np.log(T)
            + a[2] * T
            + a[3] * T**2 / 2
            + a[4] * T**3 / 3
            + a[5] * T**4 / 4
            + a[6] * T**5 / 5
            + a[7]
        )

    def s(self, T: float, P: float) -> float:
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
            return R * (
                -a[0] * T ** (-2) / 2
                - a[1] * T ** (-1)
                + a[2] * np.log(T)
                + a[3] * T
                + a[4] * T**2 / 2
                + a[5] * T**3 / 3
                + a[6] * T**4 / 4
                + a[8]
            ) - R * np.log(P / P_0)

    def g(self, T: float, P: float) -> float:
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
            return self.h(T) - T * self.s(T, P)

    def gT(self, T: float) -> float:
        """
        Molar specific gibbs free energy at reference pressure [J/mol.K]

        Parameters
        ----------
        T : float
            Temperature [K]
        P : float
            Absolute pressure [Pa]
        """
        return self.h(T) - T * self.s(T, P_0)


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

    def __init__(self, species: tuple[Specie]):
        self.species = species
        if isinstance(species, Specie):
            self.M_ij = 0
            self.species = (species,)  # convert input to tuple
        elif len(species) == 1:
            self.M_ij = 0
        else:
            self.M_ij = np.array(
                [
                    [1 / (s_i.M ** (-1) + s_j.M ** (-1)) for s_i in species]
                    for s_j in species
                ]
            )

    def D_ij(self, i: int, j: int, T: float, P: float) -> float:
        """
        Binary diffusivities [m^2/s]

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
        return (
            1.01325e-2
            * T**1.75
            / (
                P
                * self.M_ij[i, j] ** 0.5
                * (self.species[i].V ** (1 / 3) + self.species[j].V ** (1 / 3)) ** 2
            )
        )

    def cp(self, T: float, xs: np.ndarray) -> float:
        """
        Molar specific heat at constant pressure [J/mol.K]

        Parameters
        ----------
        T : float
            Temperature [K]
        xs : numpy.ndarray
            Molar fractions
        """
        return sum(
            np.array([xs[i] * specie.cp(T) for i, specie in enumerate(self.species)])
        )

    def h(self, T: float, xs: np.ndarray) -> float:
        """
        Molar specific enthalpy [J/mol]

        Parameters
        ----------
        T : float
            Temperature [K]
        xs : numpy.ndarray
            Molar fractions
        """
        return sum(
            np.array([xs[i] * specie.h(T) for i, specie in enumerate(self.species)])
        )

    def s(self, T: float, P: float, xs: np.ndarray) -> float:
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
        return sum(
            np.array(
                [
                    xs[i] * specie.s(T, P * xs[i])
                    for i, specie in enumerate(self.species)
                ]
            )
        )

    def g(self, T: float, P: float, xs: np.ndarray) -> float:
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
        return sum(
            np.array(
                [
                    xs[i] * specie.g(T, P * xs[i])
                    for i, specie in enumerate(self.species)
                ]
            )
        )


class BasicSpecies:
    """
    Gas species for SOC modelling (i.e., H2, H2O, O2, N2, CO, CO2, CH4)

    Notes
    -----
    * Coefficients from Ref. [1]

    References
    ----------
    1. https://ntrs.nasa.gov/citations/20020085330
    """

    H2 = Specie(
        (
            np.array(
                [
                    4.078323210e04,
                    -8.009186040e02,
                    8.214702010e00,
                    -1.269714457e-02,
                    1.753605076e-05,
                    -1.202860270e-08,
                    3.368093490e-12,
                    2.682484665e03,
                    -3.043788844e01,
                    200,
                    1e3,
                ]
            ),
            np.array(
                [
                    5.608128010e05,
                    -8.371504740e02,
                    2.975364532e00,
                    1.252249124e-03,
                    -3.740716190e-07,
                    5.936625200e-11,
                    -3.606994100e-15,
                    5.339824410e03,
                    -2.202774769e00,
                    1e3,
                    6e3,
                ]
            ),
        ),
        2.01588,
        6.12,
    )
    H2O = Specie(
        (
            np.array(
                [
                    -3.947960830e04,
                    5.755731020e02,
                    9.317826530e-01,
                    7.222712860e-03,
                    -7.342557370e-06,
                    4.955043490e-09,
                    -1.336933246e-12,
                    -3.303974310e04,
                    1.724205775e01,
                    200,
                    1e3,
                ]
            ),
            np.array(
                [
                    1.034972096e06,
                    -2.412698562e03,
                    4.646110780e00,
                    2.291998307e-03,
                    -6.836830480e-07,
                    9.426468930e-11,
                    -4.822380530e-15,
                    -1.384286509e04,
                    -7.978148510e00,
                    1e3,
                    6e3,
                ]
            ),
        ),
        18.01528,
        13.1,
    )
    O2 = Specie(
        (
            np.array(
                [
                    -3.425563420e04,
                    4.847000970e02,
                    1.119010961e00,
                    4.293889240e-03,
                    -6.836300520e-07,
                    -2.023372700e-09,
                    1.039040018e-12,
                    -3.391454870e03,
                    1.849699470e01,
                    200,
                    1e3,
                ]
            ),
            np.array(
                [
                    -1.037939022e06,
                    2.344830282e03,
                    1.819732036e00,
                    1.267847582e-03,
                    -2.188067988e-07,
                    2.053719572e-11,
                    -8.193467050e-16,
                    -1.689010929e04,
                    1.738716506e01,
                    1e3,
                    6e3,
                ]
            ),
        ),
        31.99880,
        16.3,
    )
    N2 = Specie(
        (
            np.array(
                [
                    2.210371497e04,
                    -3.818461820e02,
                    6.082738360e00,
                    -8.530914410e-03,
                    1.384646189e-05,
                    -9.625793620e-09,
                    2.519705809e-12,
                    7.108460860e02,
                    -1.076003744e01,
                    200,
                    1e3,
                ]
            ),
            np.array(
                [
                    5.877124060e05,
                    -2.239249073e03,
                    6.066949220e00,
                    -6.139685500e-04,
                    1.491806679e-07,
                    -1.923105485e-11,
                    1.061954386e-15,
                    1.283210415e04,
                    -1.586640027e01,
                    1e3,
                    6e3,
                ]
            ),
        ),
        28.01340,
        18.5,
    )
    CO2 = Specie(
        (
            np.array(
                [
                    4.943650540e04,
                    -6.264116010e02,
                    5.301725240e00,
                    2.503813816e-03,
                    -2.127308728e-07,
                    -7.689988780e-10,
                    2.849677801e-13,
                    -4.528198460e04,
                    -7.048279440e00,
                    200,
                    1e3,
                ]
            ),
            np.array(
                [
                    1.176962419e05,
                    -1.788791477e03,
                    8.291523190e00,
                    -9.223156780e-05,
                    4.863676880e-09,
                    -1.891053312e-12,
                    6.330036590e-16,
                    -3.908350590e04,
                    -2.652669281e01,
                    1e3,
                    6e3,
                ]
            ),
        ),
        44.00950,
        26.7,
    )
    CO = Specie(
        (
            np.array(
                [
                    1.489045326e04,
                    -2.922285939e02,
                    5.724527170e00,
                    -8.176235030e-03,
                    1.456903469e-05,
                    -1.087746302e-08,
                    3.027941827e-12,
                    -1.303131878e04,
                    -7.859241350e00,
                    200,
                    1e3,
                ]
            ),
            np.array(
                [
                    4.619197250e05,
                    -1.944704863e03,
                    5.916714180e00,
                    -5.664282830e-04,
                    1.398814540e-07,
                    -1.787680361e-11,
                    9.620935570e-16,
                    -2.466261084e03,
                    -1.387413108e01,
                    1e3,
                    6e3,
                ]
            ),
        ),
        28.01010,
        18.0,
    )
    CH4 = Specie(
        (
            np.array(
                [
                    -1.766850998e05,
                    2.786181020e03,
                    -1.202577850e01,
                    3.917619290e-02,
                    -3.619054430e-05,
                    2.026853043e-08,
                    -4.976705490e-12,
                    -2.331314360e04,
                    8.904322750e01,
                    200,
                    1e3,
                ]
            ),
            np.array(
                [
                    3.730042760e06,
                    -1.383501485e04,
                    2.049107091e01,
                    -1.961974759e-03,
                    4.727313040e-07,
                    -3.728814690e-11,
                    1.623737207e-15,
                    7.532066910e04,
                    -1.219124889e02,
                    1e3,
                    6e3,
                ]
            ),
        ),
        16.04246,
        15.9 + 2.31 * 4,
    )
