"""Gas transport and electrochemistry at constant temperature, with co-flow channels."""

from __future__ import annotations

import numpy as np

from nest.boundary import BoundaryData
from nest.cell import Cell
from nest.constants import R

__all__ = ["Model"]


class _Grid:
    """Uniform finite-volume grids for the channels, electrodes, and support."""

    def __init__(
        self,
        length,
        fuel_thickness,
        air_thickness,
        n_x,
        n_zf,
        n_za,
        support_thickness,
        n_zs,
        air_length,
    ):
        def _uniform_grid(length, n):
            """Return equally spaced faces, centers, and spacing [m].

            Returns
            -------
            faces : numpy.ndarray
                Face coordinates [m].
            centers : numpy.ndarray
                Cell-center coordinates [m].
            spacing : float
                Uniform cell width [m]; 0 when n is 0.

            """
            faces = np.linspace(0.0, length, n + 1)
            centers = (faces[:-1] + faces[1:]) / 2
            return faces, centers, length / n if n else 0.0

        self.x_faces, self.x, self.dx = _uniform_grid(length, n_x)
        self.x_air_faces, self.x_air, self.dx_air = _uniform_grid(air_length, n_x)
        self.z_fuel_faces, self.z_fuel, self.dz_fuel = _uniform_grid(
            fuel_thickness, n_zf
        )
        self.z_air_faces, self.z_air, self.dz_air = _uniform_grid(air_thickness, n_za)
        self.z_support_faces, self.z_support, self.dz_support = _uniform_grid(
            support_thickness, n_zs
        )


class Model:
    """Cell equations for steady and transient solvers.

    Voltage is air minus fuel; current is positive in fuel-cell operation.

    Parameters
    ----------
    cell : Cell
        Geometry and materials; electrode conductivities must be effective ionic values.
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

    def __init__(
        self,
        cell: Cell,
        boundary: BoundaryData,
        n_x: int,
        n_zf: int,
        n_za: int,
        n_zs: int = 6,
    ):
        self.cell, self.boundary = cell, boundary
        support = cell.fuel_support
        support_thickness = 0.0 if support is None else support.delta
        self.n_x, self.n_zf, self.n_za = n_x, n_zf, n_za
        self.n_zs = n_zs if support_thickness > 0 else 0
        self.T = float(boundary.T)

        if cell.fuel_channel is None or cell.air_channel is None:
            raise ValueError(
                "EIS model requires cell.fuel_channel and cell.air_channel"
            )
        self.fuel_channel, self.air_channel = cell.fuel_channel, cell.air_channel

        fuel, air = cell.electrode_fuel, cell.electrode_air
        self.grid = _Grid(
            self.fuel_channel.length,
            fuel.delta,
            air.delta,
            n_x,
            n_zf,
            n_za,
            support_thickness,
            self.n_zs,
            self.air_channel.length,
        )
        self.support_transport = (
            fuel.transport if support is None else support.transport
        )
        self.width_active = cell.area / self.fuel_channel.length
        self.fuel_pressure = boundary.fuel_pressure
        self.air_pressure = boundary.air_pressure

        self.sigma_f = float(fuel.conductivity.sigma(self.T))
        self.sigma_a = float(air.conductivity.sigma(self.T))

        self.resistance = (
            self.grid.dz_fuel
            / (2 * self.sigma_f)  # half of one grid cell in fuel electrode
            + sum(
                layer.delta
                / layer.conductivity.sigma(self.T)  # sum of all electrolyte layers
                for layer in cell.electrolyte
            )
            + self.grid.dz_air
            / (2 * self.sigma_a)  # half of one grid cell in air electrode
        )

        self.c_in_f, self.u_in_f = self._inlet_conditions(
            boundary.n_fuel, self.fuel_pressure, self.fuel_channel
        )
        self.c_in_a, self.u_in_a = self._inlet_conditions(
            boundary.n_air, self.air_pressure, self.air_channel
        )
        self._shapes = (
            (n_zf, n_x),
            (n_za, n_x),  # Electrode potentials
            (2, n_zf, n_x),
            (2, n_za, n_x),  # Electrode concentrations
            (2, n_x),
            (2, n_x),
            (2, self.n_zs, n_x),  # Channels and support
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
        velocity = total_flow / (total_concentration * geometry.width * geometry.height)
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
        return tuple(
            v.reshape(shape)
            for v, shape in zip(np.split(np.asarray(y), self._indices), self._shapes)
        )

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
        psi_f = self.cell.electrode_fuel.kinetic.V_nernst_half(
            self.T, R * self.T * c_pf
        )
        psi_a = -self.cell.electrode_air.kinetic.V_nernst_half(
            self.T, R * self.T * c_pa
        )
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
        eta_f = psi_f - self.cell.electrode_fuel.kinetic.V_nernst_half(
            self.T, R * self.T * c_pf
        )
        eta_a = psi_a + self.cell.electrode_air.kinetic.V_nernst_half(
            self.T, R * self.T * c_pa
        )
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

    def _electrode_charge_balance(
        self, side, potential, concentrations, *, start_current=0.0, end_current=0.0
    ):
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

        Returns
        -------
        potential_rate : ndarray
            dpsi/dt [V/s], from C_vol*dpsi/dt = dj_ion/dz - i_reaction.
        species_source : ndarray
            nu*i_reaction/(n_e*F) [mol/(m^3 s)]. The caller reverses the air sign.

        """
        if side == "fuel":
            electrode = self.cell.electrode_fuel
            kinetic = electrode.kinetic
            conductivity, spacing = self.sigma_f, self.grid.dz_fuel
            equilibrium = kinetic.V_nernst_half(self.T, R * self.T * concentrations)
        else:
            electrode = self.cell.electrode_air
            kinetic = electrode.kinetic
            conductivity, spacing = self.sigma_a, self.grid.dz_air
            equilibrium = -kinetic.V_nernst_half(self.T, R * self.T * concentrations)

        overpotential = potential - equilibrium
        reaction_current = electrode.i_F_vol(
            overpotential, self.T, R * self.T * concentrations
        )
        ionic_current = np.empty((potential.shape[0] + 1, self.n_x))
        ionic_current[0], ionic_current[-1] = start_current, end_current
        ionic_current[1:-1] = conductivity * np.diff(potential, axis=0) / spacing
        potential_rate = (
            np.diff(ionic_current, axis=0) / spacing - reaction_current
        ) / electrode.capacitance_volumetric
        species_source = np.moveaxis(
            kinetic.mol_flux(reaction_current[..., None]), -1, 0
        )
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
            return support.dusty_gas_flux(self.T, gas, face, (cf - cs) / (ds + df))

        basis = np.zeros((2, self.n_x, 2))
        basis[0, :, 0], basis[1, :, 1] = 1.0, 1.0
        matrices = []
        for transport in (support, electrode.transport):
            response = transport.dusty_gas_flux(
                self.T, gas, np.broadcast_to(face[..., None], basis.shape), basis
            )
            matrices.append(-np.moveaxis(response, 0, 1))
        resistance = ds * np.linalg.inv(matrices[0]) + df * np.linalg.inv(matrices[1])
        return -np.linalg.solve(resistance, (cf - cs).T[..., None])[..., 0].T

    def _porous_gas_balance(
        self,
        concentrations,
        species_source,
        gas,
        transport,
        spacing,
        *,
        channel_concentrations=None,
        channel_at_start=True,
        start_flux=0.0,
        end_flux=0.0,
    ):
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
            Boundary species fluxes [mol/(m^2 s)].

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
            molar_flux[:, 1:-1] = transport.dusty_gas_flux(
                self.T,
                gas,
                (concentrations[:, :-1] + concentrations[:, 1:]) / 2,
                (concentrations[:, 1:] - concentrations[:, :-1]) / spacing,
            )
        if channel_concentrations is not None:
            edge, direction = (0, 1) if channel_at_start else (-1, -1)
            molar_flux[:, edge] = transport.dusty_gas_flux(
                self.T,
                gas,
                (concentrations[:, edge] + channel_concentrations) / 2,
                direction
                * (concentrations[:, edge] - channel_concentrations)
                / (spacing / 2),
            )
        return (
            -np.diff(molar_flux, axis=1) / spacing + species_source
        ) / transport.epsilon, molar_flux

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
            geometry, pressure = self.fuel_channel, self.fuel_pressure
        else:
            inlet_concentrations, inlet_velocity = self.c_in_a, self.u_in_a
            geometry, pressure = self.air_channel, self.air_pressure

        spacing = geometry.length / self.n_x
        width_active = self.cell.area / geometry.length

        # Channel cross sectiona area
        channel_area = geometry.width * geometry.height

        # Inlet molar flow rate
        inlet_molar_flow = inlet_velocity * channel_area * pressure / (R * self.T)

        # Molar flow rate [mol/s]
        molar_flow = np.concatenate(
            (
                [inlet_molar_flow],
                inlet_molar_flow
                - np.cumsum(width_active * spacing * electrode_flux.sum(axis=0)),
            )
        )

        # Gas velocity [m/s]
        velocity = molar_flow * R * self.T / (pressure * channel_area)

        # Convective species flux.
        molar_flux = np.empty((2, self.n_x + 1))
        molar_flux[:, 0] = velocity[0] * inlet_concentrations
        molar_flux[:, 1:] = velocity[1:] * concentrations

        # Channel species conservation
        return (
            -np.diff(molar_flux, axis=1) / spacing
            - width_active / channel_area * electrode_flux
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
                c_ps,
                0.0,
                self.cell.electrode_fuel.kinetic.gas,
                self.support_transport,
                self.grid.dz_support,
                channel_concentrations=c_cf,
                end_flux=support_to_fuel_flux,
            )

            dc_pf_dt, _ = self._porous_gas_balance(
                c_pf,
                fuel_source,
                self.cell.electrode_fuel.kinetic.gas,
                self.cell.electrode_fuel.transport,
                self.grid.dz_fuel,
                start_flux=support_to_fuel_flux,
            )

            fuel_channel_flux = flux_s[:, 0]

        else:
            dc_ps_dt = np.empty_like(c_ps)

            dc_pf_dt, flux_f_gas = self._porous_gas_balance(
                c_pf,
                fuel_source,
                self.cell.electrode_fuel.kinetic.gas,
                self.cell.electrode_fuel.transport,
                self.grid.dz_fuel,
                channel_concentrations=c_cf,
            )

            fuel_channel_flux = flux_f_gas[:, 0]

        # Air-side porous gas transport.
        dc_pa_dt, flux_a_gas = self._porous_gas_balance(
            c_pa,
            -air_source,
            self.cell.electrode_air.kinetic.gas,
            self.cell.electrode_air.transport,
            self.grid.dz_air,
            channel_concentrations=c_ca,
            channel_at_start=False,
        )

        air_channel_flux = -flux_a_gas[:, -1]

        # Channel gas conservation.
        dc_cf_dt = self._channel_gas_balance("fuel", c_cf, fuel_channel_flux)
        dc_ca_dt = self._channel_gas_balance("air", c_ca, air_channel_flux)

        return self.pack_state(
            dpsi_f_dt,
            dpsi_a_dt,
            dc_pf_dt,
            dc_pa_dt,
            dc_cf_dt,
            dc_ca_dt,
            dc_ps_dt,
        )
