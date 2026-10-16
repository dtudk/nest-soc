"""Gas transport and electrochemistry at constant temperature, with co-flow channels."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.chebyshev import chebval

from nest.boundary import BoundaryData
from nest.cell import Cell
from nest.constants import F, P_0, P_atm, R
from nest.layers import Kinetic, PorousTransport
from nest.properties import BasicSpecies, Mixture, Specie

from .cellEIS import CellEIS, ElectrodeEIS

__all__ = ["Model"]


def _pure_viscosity_fit(species, temperature, *, pressure=101325.0):
    """Calculate pure gas viscosity.

    Parameters
    ----------
    species : {'H2', 'H2O', 'O2', 'N2'}
        Gas species.
    temperature : float or array_like
        Temperature [K]; fit range is 500-1500 K.
    pressure : float or array_like, optional
        Pressure [Pa], default 101325. Only affects output broadcasting.

    Returns
    -------
    viscosity : float or ndarray
        Viscosity [Pa s], with temperature and pressure broadcast to a common shape.

    Notes
    -----
    Using coolprop slows the code.
    Fiting used CoolProp 8.0.0 at 101325 Pa and 500-1500 K.
    Pressure dependence is omitted. 
    """
    temperature_min = 500.0
    temperature_max = 1500.0
    coefficients = {'H2': (2.0457311737779576e-05,
            7.411319275883211e-06,
            -2.72877930223889e-07,
            2.8745333230981723e-08,
            -4.839574109394787e-09,
            9.403878027299988e-10,
            -1.9562107874165754e-10,
            4.2442872115098416e-11,
            -9.480105690058353e-12,
            2.1631362226946564e-12,
            -5.016924253264926e-13,
            1.178622717907969e-13,
            -2.7978179397977276e-14,
            6.698452691014889e-15,
            -1.6152004945171984e-15,
            3.912813496327565e-16,
            -1.0515611703288789e-16),
     'H2O': (3.706585416422283e-05,
             1.9315055422547798e-05,
             -5.287654467177458e-07,
             -4.869670119128122e-08,
             2.157386244937981e-08,
             -5.140425279600837e-09,
             1.041324032555767e-09,
             -1.9527539216156197e-10,
             3.4771744677687445e-11,
             -5.830762157890382e-12,
             8.524519623457102e-13,
             -6.762473641027829e-14,
             -2.486950917775971e-14,
             2.0139883015569898e-14,
             -1.007457630699695e-14,
             4.350145315983998e-15,
             -2.321127684539595e-15),
     'O2': (4.8194746589094807e-05,
            1.6596206487430763e-05,
            -9.428377287290597e-07,
            1.4317802422863624e-07,
            -2.2052522335917142e-08,
            3.5932726351045405e-09,
            -6.058361107140393e-10,
            1.0426494878107493e-10,
            -1.811662755862605e-11,
            3.144587439234155e-12,
            -5.384989460260643e-13,
            8.937079654675707e-14,
            -1.392952966215937e-14,
            1.8968317693753245e-15,
            -1.7229843130201527e-16,
            -1.468130057641022e-17,
            1.7090376205123297e-17),
     'N2': (4.082329489576643e-05,
            1.3877500783794923e-05,
            -7.387096474138558e-07,
            1.207564609613202e-07,
            -1.9395250457326346e-08,
            3.295483625158618e-09,
            -5.795726068961285e-10,
            1.0421447462529067e-10,
            -1.8980128611162316e-11,
            3.4717919387226885e-12,
            -6.32092158260298e-13,
            1.1324344384682467e-13,
            -1.9621714406584537e-14,
            3.185631136073377e-15,
            -4.4966580164201105e-16,
            4.130402101966881e-17,
            6.3059881444877864e-18)}

    T = np.asarray(temperature, dtype=float)
    P = np.asarray(pressure, dtype=float)
    T, P = np.broadcast_arrays(T, P)
    theta = (2*T - temperature_min - temperature_max) / (temperature_max - temperature_min)
    value = chebval(theta, coefficients[species])
    return float(value) if value.ndim == 0 else value


_VISCOSITY_SPECIES: dict[Specie, str] = {
    BasicSpecies.H2: "H2",
    BasicSpecies.H2O: "H2O",
    BasicSpecies.O2: "O2",
    BasicSpecies.N2: "N2",
}


def _half_potential(temperature, kinetic, concentrations):
    """Calculate the equilibrium half-reaction potential from gas concentrations.

    Parameters
    ----------
    temperature : float
        Temperature [K].
    kinetic : Kinetic
        Reaction stoichiometry and thermodynamic properties.
    concentrations : ndarray
        Species concentrations [mol/m^3], shaped (n_species, ...).

    Returns
    -------
    potential : float or ndarray
        Half-reaction equilibrium potential [V].

    Notes
    -----
    E = E_ref(T) + RT/(n_e*F) * sum(nu_i * ln(p_i/P_0)).
    """
    potential = kinetic.Vt_nernst_half(temperature)

    for nu, concentration in zip(kinetic.nu, concentrations):
        if nu != 0:
            partial_pressure = R * temperature * concentration
            potential += (
                R * temperature / (kinetic.n_e * F)
                * nu * np.log(partial_pressure / P_0)
            )

    return potential


def _exchange_current_per_tpb(
    temperature: float,
    partial_pressures: np.ndarray,
    kinetic: Kinetic,
    electrode: ElectrodeEIS,
) -> float | np.ndarray:
    """Calculate exchange current per TPB length.

    Parameters
    ----------
    temperature : float
        Temperature [K].
    partial_pressures : array_like
        Species partial pressures [Pa], shaped (n_species, ...).
    kinetic : Kinetic
        Temperature and partial-pressure dependencies.
    electrode : ElectrodeEIS
        Exchange-current prefactor.

    Returns
    -------
    current : float or ndarray
        Exchange current per TPB length [A/m].

    Notes
    -----
    i0 = A_r*T^theta*exp(-E_act/(R*T))*prod((p_i/P_atm)^a_i).
    """
    T = float(temperature)
    Ps = np.atleast_1d(np.asarray(partial_pressures, dtype=float))

    exponents = np.asarray(kinetic.p, dtype=float).reshape(
        (-1,) + (1,) * (Ps.ndim - 1)
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        current = (
            electrode.exchange_current_prefactor
            * T ** kinetic.theta
            * np.exp(-kinetic.E_act / (R * T))
            * np.prod((Ps / P_atm) ** exponents, axis=0)
        )

    return float(current) if np.ndim(current) == 0 else current

def _butler_volmer_current(
    overpotential: float | np.ndarray,
    temperature: float,
    exchange_current: float | np.ndarray,
    kinetic: Kinetic,
) -> float | np.ndarray:
    """Calculate Butler-Volmer current per TPB length.

    Parameters
    ----------
    overpotential : float or array_like
        Activation overpotential [V].
    temperature : float
        Temperature [K].
    exchange_current : float or array_like
        Exchange current per TPB length [A/m].
    kinetic : Kinetic
        Charge-transfer coefficients and electron count.

    Returns
    -------
    current : float or ndarray
        Faradaic current per TPB length [A/m].

    Notes
    -----
    i = i0 * [exp(alpha*n_e*F*eta/(R*T)) - exp(-beta*n_e*F*eta/(R*T))].
    """
    eta = np.asarray(overpotential, dtype=float)
    i0 = np.asarray(exchange_current, dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        current = i0 * (
            np.expm1(kinetic.alpha * kinetic.n_e * F * eta / (R * temperature))
            - np.expm1(-kinetic.beta * kinetic.n_e * F * eta / (R * temperature))
        )

    return float(current) if np.ndim(current) == 0 else current


def _faradaic_species_source(
    volumetric_current: float | np.ndarray,
    stoichiometry: np.ndarray,
    n_electrons: float,
) -> np.ndarray:
    """Convert reaction current to species production rates.

    Parameters
    ----------
    volumetric_current : float or array_like
        Reaction current per volume [A/m^3].
    stoichiometry : array_like
        Species coefficients; positive for production.
    n_electrons : float
        Number of electrons transferred per reaction.

    Returns
    -------
    source : ndarray
        Production rates [mol/(m^3 s)], with species on the first axis.

    Notes
    -----
    Uses S_i = nu_i*i_reaction/(n_e*F); positive means species production.
    """
    current = np.asarray(volumetric_current, dtype=float)
    nu = np.asarray(stoichiometry, dtype=float)

    return nu.reshape((-1,) + (1,) * current.ndim) * current / (n_electrons * F)


@lru_cache(maxsize=512)
def _pure_viscosity(temperature: float, specie: Specie) -> float:
    """Return cached pure-gas viscosity [Pa s]."""
    return _pure_viscosity_fit(_VISCOSITY_SPECIES[specie], temperature)


def _gas_viscosity(
    temperature: float,
    gas: Mixture,
    mole_fractions: np.ndarray,
) -> float | np.ndarray:
    """Calculate gas-mixture viscosity using Wilke's mixing rule.

    Parameters
    ----------
    temperature : float
        Temperature [K]; viscosity fit range is 500-1500 K.
    gas : Mixture
        Gas mixture containing H2, H2O, O2, or N2.
    mole_fractions : array_like
        Species mole fractions [-], shaped (n_species, ...).

    Returns
    -------
    viscosity : float or ndarray
        Mixture viscosity [Pa s], with the species axis removed.
    """
    x = np.asarray(mole_fractions, dtype=float)

    viscosities = [
        _pure_viscosity(float(temperature), specie)
        for specie in gas.species
    ]
    molar_masses = [specie.M for specie in gas.species]

    viscosity = np.zeros_like(x[0], dtype=float)

    for i, mu_i in enumerate(viscosities):
        denominator = np.zeros_like(viscosity)

        for j, mu_j in enumerate(viscosities):
            mass_ratio = molar_masses[j] / molar_masses[i]

            phi = (
                (1 + np.sqrt(mu_i / mu_j) * mass_ratio**0.25)**2
                / np.sqrt(8 * (1 + 1 / mass_ratio))
            )

            denominator += x[j] * phi

        viscosity += x[i] * mu_i / denominator

    return float(viscosity) if viscosity.ndim == 0 else viscosity



def _dusty_gas_flux(
    temperature: float,
    gas: Mixture,
    transport: PorousTransport,
    concentrations: np.ndarray,
    concentration_gradient: np.ndarray,
) -> np.ndarray:
    """Calculate binary molar fluxes using the Dusty-Gas Model.

    Parameters
    ----------
    temperature : float
        Temperature [K].
    gas : Mixture
        Two-species gas mixture.
    transport : PorousTransport
        Porosity, tortuosity, and pore diameter.
    concentrations : array_like
        Species concentrations [mol/m^3], shaped (2, ...).
    concentration_gradient : array_like
        Concentration gradients [mol/m^4], same shape as concentrations.

    Returns
    -------
    flux : ndarray
        Species molar fluxes [mol/(m^2 s)], shaped (2, ...).

    Notes
    -----
    Accounts for molecular diffusion, Knudsen diffusion, and pressure-driven transport.
    """
    T = float(temperature)
    c = np.asarray(concentrations, dtype=float)
    grad_c = np.asarray(concentration_gradient, dtype=float)

    # Local gas pressure and mole fractions.
    pressure = R * T * c.sum(axis=0)
    x = c / c.sum(axis=0)

    # Effective molecular and Knudsen diffusivities [m²/s].
    porosity_factor = transport.epsilon / transport.tau
    D12 = porosity_factor * gas.D_ij(0, 1, T, pressure)
    DK = porosity_factor * np.array([
        transport.D_knudsen(T, specie) for specie in gas.species
    ])

    # Gas viscosity [Pa s] and porous permeability [m²].
    viscosity = _gas_viscosity(T, gas, x)
    permeability = porosity_factor * transport.dp**2 / 32

    # Coefficient matrix of the binary Dusty-Gas equations.
    matrix = np.empty(pressure.shape + (2, 2))
    matrix[..., 0, 0] = 1 / DK[0] + x[1] / D12
    matrix[..., 0, 1] = -x[0] / D12
    matrix[..., 1, 0] = -x[1] / D12
    matrix[..., 1, 1] = 1 / DK[1] + x[0] / D12

    # Concentration-gradient and pressure-driven terms.
    driving = (
        -grad_c
        - permeability / viscosity
        * c / DK.reshape((2,) + (1,) * (c.ndim - 1))
        * (R * T * grad_c.sum(axis=0))
    )

    # Solve the coupled 2x2 system at each spatial location.
    rhs = np.moveaxis(driving, 0, -1)[..., None]
    flux = np.linalg.solve(matrix, rhs)[..., 0]

    return np.moveaxis(flux, -1, 0)



class _Grid:
    """Uniform finite-volume grids for the channels, electrodes, and support."""

    def __init__(self, length, fuel_thickness, air_thickness,
                 n_x, n_zf, n_za, support_thickness, n_zs):
        def _uniform_grid(length, n):
            """Return equally spaced faces, centers, and spacing [m]."""
            faces = np.linspace(0.0, length, n + 1)
            centers = (faces[:-1] + faces[1:]) / 2
            return faces, centers, length / n if n else 0.0

        self.x_faces, self.x, self.dx = _uniform_grid(length, n_x)
        self.z_fuel_faces, self.z_fuel, self.dz_fuel = _uniform_grid(fuel_thickness, n_zf)
        self.z_air_faces, self.z_air, self.dz_air = _uniform_grid(air_thickness, n_za)
        self.z_support_faces, self.z_support, self.dz_support = _uniform_grid(support_thickness, n_zs)


class Model:
    """Cell equations for steady and transient solvers.

    Voltage is air minus fuel; current is positive in fuel-cell operation.

    Parameters
    ----------
    cell : Cell
        Geometry and materials; electrode conductivities must be effective ionic values.
    eis : CellEIS
        TPB, capacitance, channel, and support inputs.
    boundary : BoundaryData
        Temperature [K], pressure [Pa], and inlet flows [mol/s].
    n_x : int
        Number of cells along each channel.
    n_zf, n_za : int
        Number of cells across the fuel and air electrodes.
    n_zs : int, optional
        Number of support cells. Default is 6; ignored for zero support thickness.

    Notes
    -----
    Fuel runs from channel to electrolyte; air runs from electrolyte to channel.
    """

    def __init__(self, cell: Cell, eis: CellEIS, boundary: BoundaryData,
                 n_x: int, n_zf: int, n_za: int, n_zs: int = 6):
        self.cell, self.eis, self.boundary = cell, eis, boundary
        self.n_x, self.n_zf, self.n_za = n_x, n_zf, n_za
        self.n_zs = n_zs if eis.support_thickness > 0 else 0
        self.T = float(boundary.T)

        fuel, air = cell.electrode_fuel, cell.electrode_air
        self.grid = _Grid(eis.fuel_channel.length, fuel.delta, air.delta,
                          n_x, n_zf, n_za, eis.support_thickness, self.n_zs)
        self.support_transport = (fuel.transport if eis.support_transport is None
                                  else eis.support_transport)
        self.width_active = cell.area / eis.fuel_channel.length
        self.fuel_pressure = boundary.P if eis.fuel_pressure is None else eis.fuel_pressure
        self.air_pressure = boundary.P if eis.air_pressure is None else eis.air_pressure

        self.sigma_f = float(fuel.conductivity.sigma(self.T))
        self.sigma_a = float(air.conductivity.sigma(self.T))
       
        self.resistance = (
            self.grid.dz_fuel / (2 * self.sigma_f)                  # half of one grid cell in fuel electrode 
            + sum(layer.delta / layer.conductivity.sigma(self.T)    # sum of all electrolyte layers
                  for layer in cell.electrolyte)    
            + self.grid.dz_air / (2 * self.sigma_a)                 # half of one grid cell in air electrode
        )

        self.c_in_f, self.u_in_f = self._inlet_conditions(
            boundary.n_fuel, self.fuel_pressure, eis.fuel_channel)
        self.c_in_a, self.u_in_a = self._inlet_conditions(
            boundary.n_air, self.air_pressure, eis.air_channel)
        self._shapes = (
            (n_zf, n_x), (n_za, n_x),                   # Electrode potentials
            (2, n_zf, n_x), (2, n_za, n_x),             # Electrode concentrations
            (2, n_x), (2, n_x), (2, self.n_zs, n_x),    # Channels and support
        )
        self._indices = np.cumsum([np.prod(shape) for shape in self._shapes])[:-1]

    def _inlet_conditions(self, molar_flows, pressure, geometry):
        """Convert species molar flows to inlet concentrations and velocity.

        Parameters
        ----------
        molar_flows : array_like
            Species molar flows [mol/s].
        pressure : float
            Channel pressure [Pa].
        geometry : ChannelGeometry
            Channel width and height [m].

        Returns
        -------
        concentration : ndarray
            Species concentrations [mol/m^3] from the ideal-gas law.
        velocity : float
            Gas velocity [m/s] through the channel cross-section.
        """
        molar_flows = np.asarray(molar_flows, dtype=float)
        total_flow = molar_flows.sum()
        total_concentration = pressure / (R * self.T)
        concentration = molar_flows / total_flow * total_concentration
        velocity = total_flow / (total_concentration * geometry.width * geometry.thickness)
        return concentration, velocity

    def pack_state(self, *fields: np.ndarray) -> np.ndarray:
        """Flatten the state fields into one vector.

        Parameters
        ----------
        *fields : ndarray
            psi_f, psi_a, c_pf, c_pa, c_cf, c_ca, c_ps, in that order.

        Returns
        -------
        y : ndarray
            One-dimensional state vector.
        """
        return np.concatenate([np.asarray(v, dtype=float).ravel() for v in fields])

    def unpack_state(self, y: np.ndarray) -> tuple[np.ndarray, ...]:
        """View the state vector as separate fields.

        Parameters
        ----------
        y : ndarray
            State vector in pack_state order.

        Returns
        -------
        psi_f, psi_a : ndarray
            Electronic-minus-ionic potential [V], shaped (n_zf, n_x) and (n_za, n_x).
        c_pf, c_pa : ndarray
            Electrode gas [mol/m^3], shaped (2, n_zf, n_x) and (2, n_za, n_x).
        c_cf, c_ca : ndarray
            Channel gas [mol/m^3], each shaped (2, n_x).
        c_ps : ndarray
            Support gas [mol/m^3], shaped (2, n_zs, n_x).

        """
        return tuple(v.reshape(shape) for v, shape in
                     zip(np.split(np.asarray(y), self._indices), self._shapes))

    def initial_state(self) -> np.ndarray:
        """Initialize with inlet gas conditions and zero activation overpotentials.

        Returns
        -------
        y : ndarray
            State vector in pack_state order.
        """
        c_pf = np.broadcast_to(self.c_in_f[:, None, None], self._shapes[2])
        c_pa = np.broadcast_to(self.c_in_a[:, None, None], self._shapes[3])
        c_cf = np.broadcast_to(self.c_in_f[:, None], self._shapes[4])
        c_ca = np.broadcast_to(self.c_in_a[:, None], self._shapes[5])
        c_ps = np.broadcast_to(self.c_in_f[:, None, None], self._shapes[6])
        psi_f = _half_potential(self.T, self.cell.electrode_fuel.kinetic, c_pf)
        psi_a = -_half_potential(self.T, self.cell.electrode_air.kinetic, c_pa)
        return self.pack_state(psi_f, psi_a, c_pf, c_pa, c_cf, c_ca, c_ps)

    def current_density(self, y: np.ndarray, voltage: float) -> np.ndarray:
        """Calculate local current density passing through the electrolyte.

        Parameters
        ----------
        y : ndarray
            State vector in pack_state order.
        voltage : float
            Voltage [V].

        Returns
        -------
        current : ndarray
            Current density [A/m^2], shaped (n_x,); positive in fuel-cell operation.
        """
        psi_f, psi_a = self.unpack_state(y)[:2]
        return (-voltage - psi_f[-1] + psi_a[0]) / self.resistance

    def overpotentials(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate activation overpotentials from local equilibrium potentials.

        Parameters
        ----------
        y : ndarray
            State vector in pack_state order.

        Returns
        -------
        eta_f, eta_a : ndarray
            Fuel and air overpotentials [V], shaped (n_zf, n_x) and (n_za, n_x).
            Electrolyte-adjacent values are eta_f[-1] and eta_a[0].
        """
        psi_f, psi_a, c_pf, c_pa, *_ = self.unpack_state(y)
        eta_f = psi_f - _half_potential(self.T, self.cell.electrode_fuel.kinetic, c_pf)
        eta_a = psi_a + _half_potential(self.T, self.cell.electrode_air.kinetic, c_pa)
        return eta_f, eta_a

    def voltage_for_current(self, y: np.ndarray, current_density: float = 0.0) -> float:
        """Calculate voltage for a given state and mean current.

        Parameters
        ----------
        y : ndarray
            State vector in pack_state order.
        current_density : float, optional
            Mean current density [A/m^2]. Default is 0 (open circuit).

        Returns
        -------
        voltage : float
            Voltage [V].
        """
        psi_f, psi_a = self.unpack_state(y)[:2]
        return float(np.mean(psi_a[0] - psi_f[-1]) - current_density * self.resistance)

    def _electrode_charge_balance(self, side, potential, concentrations, *,
                           start_current=0.0, end_current=0.0):
        """Calculate electrode potential derivatives and reaction sources.

        Parameters
        ----------
        side : str
            "fuel" or "air".
        potential : ndarray
            Electronic-minus-ionic potential [V], shaped (n_z, n_x).
        concentrations : ndarray
            Species concentrations [mol/m^3], shaped (2, n_z, n_x).
        start_current, end_current : float or ndarray, optional
            Ionic current densities at the first and last faces [A/m^2].
            Positive along increasing z; default is zero.

        Returns
        -------
        potential_rate : ndarray
            dpsi/dt [V/s], from C_vol*dpsi/dt = dj_ion/dz - i_reaction.
        species_source : ndarray
            nu*i_reaction/(n_e*F) [mol/(m^3 s)]. The caller reverses the air sign.
        """
        if side == "fuel":
            kinetic, eis = self.cell.electrode_fuel.kinetic, self.eis.fuel
            conductivity, spacing = self.sigma_f, self.grid.dz_fuel
            equilibrium = _half_potential(self.T, kinetic, concentrations)
        else:
            kinetic, eis = self.cell.electrode_air.kinetic, self.eis.air
            conductivity, spacing = self.sigma_a, self.grid.dz_air
            equilibrium = -_half_potential(self.T, kinetic, concentrations)

        overpotential = potential - equilibrium
        exchange_current = _exchange_current_per_tpb(
            self.T, R * self.T * concentrations, kinetic, eis)
        reaction_current = eis.tpb_density * _butler_volmer_current(
            overpotential, self.T, exchange_current, kinetic)
        ionic_current = np.empty((potential.shape[0] + 1, self.n_x))
        ionic_current[0], ionic_current[-1] = start_current, end_current
        ionic_current[1:-1] = conductivity * np.diff(potential, axis=0) / spacing
        potential_rate = (np.diff(ionic_current, axis=0) / spacing
                          - reaction_current) / eis.capacitance_volumetric
        species_source = _faradaic_species_source(reaction_current, kinetic.nu, kinetic.n_e)
        return potential_rate, species_source

    def _support_to_fuel_flux(self, c_ps, c_pf):
        """Calculate gas flux from the support into the fuel electrode.

        Parameters
        ----------
        c_ps, c_pf : ndarray
            Support and fuel-electrode concentrations [mol/m^3],
            shaped (2, n_zs, n_x) and (2, n_zf, n_x).

        Returns
        -------
        molar_flux : ndarray
            Species flux [mol/(m^2 s)], shaped (2, n_x).
            Positive from support to fuel electrode.

        """
        support = self.support_transport
        electrode = self.cell.electrode_fuel
        gas = electrode.kinetic.gas
        ds, df = self.grid.dz_support / 2, self.grid.dz_fuel / 2
        cs, cf = c_ps[:, -1], c_pf[:, 0]
        face = (df * cs + ds * cf) / (ds + df)
        if support is electrode.transport:
            return _dusty_gas_flux(self.T, gas, support, face, (cf - cs) / (ds + df))

        basis = np.zeros((2, self.n_x, 2))
        basis[0, :, 0], basis[1, :, 1] = 1.0, 1.0
        matrices = []
        for transport in (support, electrode.transport):
            response = _dusty_gas_flux(
                self.T, gas, transport, np.broadcast_to(face[..., None], basis.shape), basis)
            matrices.append(-np.moveaxis(response, 0, 1))
        resistance = (ds * np.linalg.inv(matrices[0])
                      + df * np.linalg.inv(matrices[1]))
        return -np.linalg.solve(resistance, (cf - cs).T[..., None])[..., 0].T

    def _porous_gas_balance(self, concentrations, species_source, gas, transport, spacing, *,
                        channel_concentrations=None, channel_at_start=True,
                        start_flux=0.0, end_flux=0.0):
        """Calculate gas concentration derivatives in an electrode or support.

        Parameters
        ----------
        concentrations : ndarray
            Species concentrations [mol/m^3], shaped (2, n_z, n_x).
        species_source : float or ndarray
            Reaction source per total material volume [mol/(m^3 s)]; zero for support.
        gas : Mixture
            Gas species and properties.
        transport : PorousTransport
            Porosity, tortuosity, and pore diameter.
        spacing : float
            Cell thickness [m].
        channel_concentrations : ndarray, optional
            Adjacent channel concentrations [mol/m^3], shaped (2, n_x).
        channel_at_start : bool, optional
            True places the channel at the first face; False at the last.
        start_flux, end_flux : float or ndarray, optional
            Boundary species fluxes [mol/(m^2 s)], positive along increasing z.
            Default is zero; a connected channel determines its own boundary flux.

        Returns
        -------
        dc_dt : ndarray
            Concentration derivatives [mol/(m^3 s)], from epsilon*dc/dt = -dN/dz + S.
        molar_flux : ndarray
            Species fluxes at all faces [mol/(m^2 s)], shaped (2, n_z + 1, n_x).
        """
        molar_flux = np.zeros((2, concentrations.shape[1] + 1, self.n_x))
        molar_flux[:, 0], molar_flux[:, -1] = start_flux, end_flux
        if concentrations.shape[1] > 1:
            molar_flux[:, 1:-1] = _dusty_gas_flux(
                self.T, gas, transport,
                (concentrations[:, :-1] + concentrations[:, 1:]) / 2,
                (concentrations[:, 1:] - concentrations[:, :-1]) / spacing,
            )
        if channel_concentrations is not None:
            edge, direction = (0, 1) if channel_at_start else (-1, -1)
            molar_flux[:, edge] = _dusty_gas_flux(
                self.T, gas, transport, (concentrations[:, edge] + channel_concentrations) / 2,
                direction * (concentrations[:, edge] - channel_concentrations) / (spacing / 2))
        return (-np.diff(molar_flux, axis=1) / spacing + species_source) / transport.epsilon, molar_flux


    def _channel_gas_balance(self, side, concentrations, electrode_flux):
        """Calculate channel concentration derivatives from flow and gas exchange.

        Parameters
        ----------
        side : str
            "fuel" or "air".
        concentrations : ndarray
            Channel concentrations [mol/m^3], shaped (2, n_x).
        electrode_flux : ndarray
            Species flux [mol/(m^2 s)], shaped (2, n_x).
            Positive from channel into electrode or support.

        Returns
        -------
        dc_dt : ndarray
            Channel concentration derivatives [mol/(m^3 s)].

        Notes
        -----
        dc/dt = -d(u*c)/dx - (w_active/A_channel)*N_out.
        """
        if side == "fuel":
            inlet_concentrations, inlet_velocity = self.c_in_f, self.u_in_f
            geometry, pressure = self.eis.fuel_channel, self.fuel_pressure
        else:
            inlet_concentrations, inlet_velocity = self.c_in_a, self.u_in_a
            geometry, pressure = self.eis.air_channel, self.air_pressure

        # Channel cross sectiona area
        channel_area = geometry.width * geometry.thickness

        # Inlet molar flow rate
        inlet_molar_flow = inlet_velocity * channel_area * pressure / (R * self.T)

        # Molar flow rate at each channel face [mol/s]
        molar_flow = np.concatenate((
            [inlet_molar_flow],
            inlet_molar_flow - np.cumsum(self.width_active * self.grid.dx * electrode_flux.sum(axis=0)),
        ))

        # Gas velocity at each channel face [m/s]
        velocity = molar_flow * R * self.T / (pressure * channel_area)

        # Convective species flux at channel faces.
        molar_flux = np.empty((2, self.n_x + 1))
        molar_flux[:, 0] = velocity[0] * inlet_concentrations
        molar_flux[:, 1:] = velocity[1:] * concentrations

        # Channel species conservation 
        return (
            -np.diff(molar_flux, axis=1) / self.grid.dx
            - self.width_active / channel_area * electrode_flux
        )



    def time_derivatives(self, t: float, y: np.ndarray, voltage: float) -> np.ndarray:
        """Evaluate the coupled electrochemical and gas conservation equations.

        Parameters
        ----------
        t : float
            Time [s]; required by the solver interface.
        y : ndarray
            State vector in pack_state order.
        voltage : float
            Applied cell voltage [V].

        Returns
        -------
        dy : ndarray
            State derivatives [V/s and mol/(m^3 s)].

        Notes
        -----
        Electrode charge: C_vol*dpsi/dt = dj_ion/dz - i_F.
        Porous gas: epsilon*dc/dt = -dN/dz + S.
        Channel gas: dc/dt = -d(u*c)/dx - (w_active/A_channel)*N.
        """
        psi_f, psi_a, c_pf, c_pa, c_cf, c_ca, c_ps = self.unpack_state(y)

        # Electrolyte current density [A/m²].
        current = self.current_density(y, voltage)

        # Electrode charge balances and Faradaic species sources.
        dpsi_f_dt, fuel_source = self._electrode_charge_balance(
            "fuel", psi_f, c_pf, end_current=current
        )
        dpsi_a_dt, air_source = self._electrode_charge_balance(
            "air", psi_a, c_pa, start_current=current
        )

        # Fuel-side gas transport, with or without a support layer.
        if self.n_zs:
            support_to_fuel_flux = self._support_to_fuel_flux(c_ps, c_pf)

            dc_ps_dt, flux_s = self._porous_gas_balance(
                c_ps, 0.0,
                self.cell.electrode_fuel.kinetic.gas,
                self.support_transport,
                self.grid.dz_support,
                channel_concentrations=c_cf,
                end_flux=support_to_fuel_flux,
            )

            dc_pf_dt, _ = self._porous_gas_balance(
                c_pf, fuel_source,
                self.cell.electrode_fuel.kinetic.gas,
                self.cell.electrode_fuel.transport,
                self.grid.dz_fuel,
                start_flux=support_to_fuel_flux,
            )

            fuel_channel_flux = flux_s[:, 0]

        else:
            dc_ps_dt = np.empty_like(c_ps)

            dc_pf_dt, flux_f_gas = self._porous_gas_balance(
                c_pf, fuel_source,
                self.cell.electrode_fuel.kinetic.gas,
                self.cell.electrode_fuel.transport,
                self.grid.dz_fuel,
                channel_concentrations=c_cf,
            )

            fuel_channel_flux = flux_f_gas[:, 0]

        # Air-side porous gas transport.
        dc_pa_dt, flux_a_gas = self._porous_gas_balance(
            c_pa, -air_source,
            self.cell.electrode_air.kinetic.gas,
            self.cell.electrode_air.transport,
            self.grid.dz_air,
            channel_concentrations=c_ca,
            channel_at_start=False,
        )

        air_channel_flux = -flux_a_gas[:, -1]

        # Channel gas conservation.
        dc_cf_dt = self._channel_gas_balance(
            "fuel", c_cf, fuel_channel_flux
        )
        dc_ca_dt = self._channel_gas_balance(
            "air", c_ca, air_channel_flux
        )

        return self.pack_state(
            dpsi_f_dt, dpsi_a_dt,
            dc_pf_dt, dc_pa_dt,
            dc_cf_dt, dc_ca_dt, dc_ps_dt,
        )