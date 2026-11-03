from nest import boundary, cell, properties, layers
import numpy as np

from nest.constants import F, R


# ----------------------------------------------------------------------------
# Boundary
# ----------------------------------------------------------------------------
class Boundary(boundary.BoundaryData):
    """``BoundaryData`` with per-side gas quantities."""

    @classmethod
    def from_data(cls, data: boundary.BoundaryData) -> "Boundary":
        """Wrap existing boundary data (shares its arrays)."""
        if isinstance(data, cls):
            return data
        new = cls.__new__(cls)
        new.__dict__.update(vars(data))
        return new

    def flows(self, side: str) -> np.ndarray:
        """Species molar flows [mol/s]."""
        return np.asarray(self.n_fuel if _check_side(side) == "fuel" else self.n_air, dtype=float)

    def pressure(self, side: str) -> float:
        """Side pressure [Pa]."""
        return self.fuel_pressure if _check_side(side) == "fuel" else self.air_pressure

    def total_flow(self, side: str) -> float:
        """Total molar flow [mol/s]."""
        return float(self.flows(side).sum())

    def mole_fractions(self, side: str) -> np.ndarray:
        """Inlet mole fractions [-]."""
        flows = self.flows(side)
        return flows / flows.sum()

    def Ps(self, side: str) -> np.ndarray:
        """Inlet partial pressures at the side pressure [Pa]."""
        return self.mole_fractions(side) * self.pressure(side)

    def total_concentration(self, side: str, T: float | None = None) -> float:
        """Total gas concentration [mol/m^3] (ideal gas)."""
        return self.pressure(side) / (R * (self.T if T is None else T))

    def concentrations(self, side: str, T: float | None = None) -> np.ndarray:
        """Inlet species concentrations [mol/m^3]."""
        return self.mole_fractions(side) * self.total_concentration(side, T)


# ----------------------------------------------------------------------------
# Cell
# ----------------------------------------------------------------------------
class Cell(cell.Cell):
    """``nest.cell.Cell`` with the computations shared by all solvers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.distributed and self.fuel_channel.length != self.air_channel.length:
            raise ValueError("Fuel and air channel lengths must be equal (co-flow columns).")

    # -- layout --
    @property
    def distributed(self) -> bool:
        """True when both channels are given (distributed model); else area sweep only."""
        return self.fuel_channel is not None and self.air_channel is not None

    def layers(self) -> tuple:
        """Layers in degradation order: fuel, air, electrolytes."""
        return (self.electrode_fuel, self.electrode_air, *self.electrolyte)

    def electrode(self, side: str) -> layers.Layer:
        return self.electrode_fuel if _check_side(side) == "fuel" else self.electrode_air

    def channel(self, side: str) -> cell.ChannelGeometry:
        return self.fuel_channel if _check_side(side) == "fuel" else self.air_channel

    @property
    def support_transport(self) -> layers.PorousTransport:
        """Gas transport of the fuel support (the fuel electrode's if there is none)."""
        support = self.fuel_support
        return self.electrode_fuel.transport if support is None else support.transport

    def width_active(self, side: str = "fuel") -> float:
        """Active width [m]: area per unit channel length."""
        return self.area / self.channel(side).length

    def channel_area(self, side: str) -> float:
        """Channel cross-section [m^2]."""
        channel = self.channel(side)
        return channel.width * channel.height

    def channel_depth(self, side: str) -> float:
        """Channel gas volume per active area [m]: cross-section over active width."""
        return self.channel_area(side) / self.width_active(side)

    # -- porous layers ("fuel", "air", or "support") --
    def gas(self, side: str) -> properties.Mixture:
        """Gas mixture of an electrode; the support carries the fuel gas."""
        return self.electrode("fuel" if side == "support" else side).kinetic.gas

    def transport(self, side: str) -> layers.PorousTransport:
        """Gas transport model of an electrode or the fuel support."""
        return self.support_transport if side == "support" else self.electrode(side).transport

    def porosity(self, side: str) -> float:
        """Porosity of an electrode or the fuel support [-]."""
        return self.transport(side).epsilon

    def conductivity(self, side: str, T) -> float:
        """Effective conductivity of the electrode [S/m]."""
        return float(self.electrode(side).conductivity.sigma(T))

    def capacitance(self, side: str) -> float:
        """Volumetric double-layer capacitance of the electrode [F/m^3]."""
        return self.electrode(side).capacitance_volumetric

    # -- electrochemistry --
    def equilibrium_potential(self, side: str, T, Ps):
        """Electrode equilibrium potential [V]; fuel ``V_half``, air ``-V_half``."""
        V_half = self.electrode(side).kinetic.V_nernst_half(T, Ps)
        return V_half if side == "fuel" else -V_half

    def faradaic_current(self, side: str, potential, T, Ps):
        """Volumetric Faradaic current [A/m^3] at the local overpotential."""
        overpotential = potential - self.equilibrium_potential(side, T, Ps)
        return self.electrode(side).i_F_vol(overpotential, T, Ps)

    def species_source(self, side: str, current) -> np.ndarray:
        """Species production [mol/(m^3 s)] of a Faradaic current, species on the first axis."""
        return np.moveaxis(self.electrode(side).kinetic.mol_flux(current[..., None]), -1, 0)

    def V_nernst(self, T, Ps_fuel, Ps_air):
        """Thermodynamic voltage [V], see :meth:`nest.cell.Cell.V_nernst`."""
        return self.equilibrium_potential("air", T, Ps_air) - self.equilibrium_potential(
            "fuel", T, Ps_fuel
        )

    def electrolyte_resistance(self, T, ohm_deg=None) -> float:
        """Area-specific electrolyte resistance [Ohm m^2], via ``Layer.V_ohm``.

        ``ohm_deg`` is indexed by layer (fuel, air, electrolytes); None means ``m0``.
        """
        total = 0
        for i, layer in enumerate(self.electrolyte):
            kwargs = {}
            if layer.degradation.ohm_active:
                kwargs["ohm_deg"] = layer.degradation.m0 if ohm_deg is None else ohm_deg[2 + i]
            total += layer.V_ohm(1.0, T, **kwargs)
        return total

    def resistance(self, T, dz_fuel: float, dz_air: float, ohm_deg=None) -> float:
        """Ohmic resistance between the electrode cells next to the electrolyte [Ohm m^2]."""
        return (
            dz_fuel / (2 * self.conductivity("fuel", T))
            + self.electrolyte_resistance(T, ohm_deg)
            + dz_air / (2 * self.conductivity("air", T))
        )

    # -- gas flows --
    def channel_molar_flow(self, side: str, inlet_flow: float, exchange_flux) -> np.ndarray:
        """Total molar flow [mol/s] at the channel faces.

        ``exchange_flux`` [mol/(m^2 s)] leaves the channel into the electrode,
        species on the first axis and one column per channel segment.
        """
        exchange_flux = np.asarray(exchange_flux, dtype=float)
        spacing = self.channel(side).length / exchange_flux.shape[-1]
        removed = np.cumsum(self.width_active(side) * spacing * exchange_flux.sum(axis=0))
        return np.concatenate(([inlet_flow], inlet_flow - removed))

    # -- degradation --
    def degradation_state(self, y=None) -> dict:
        """``pol_deg``/``ohm_deg`` matrices (elements x layers) for the solves.

        ``y`` is the degradation ODE state (per element: polarization, then ohmic, then power);
        None gives each active model's ``m0``.
        """
        layer_list = self.layers()
        pol_layers = [k for k in (0, 1) if layer_list[k].degradation.pol_active]
        ohm_layers = [
            k for k in range(2, len(layer_list)) if layer_list[k].degradation.ohm_active
        ]
        pol = np.zeros((self.elements, len(layer_list)))
        ohm = np.zeros((self.elements, len(layer_list)))
        if y is None:
            for k in pol_layers:
                pol[:, k] = layer_list[k].degradation.m0
            for k in ohm_layers:
                ohm[:, k] = layer_list[k].degradation.m0
        else:
            y = np.asarray(y, dtype=float)
            n_pol = self.elements * len(pol_layers)
            n_ohm = self.elements * len(ohm_layers)
            pol[:, pol_layers] = y[:n_pol].reshape(self.elements, len(pol_layers))
            ohm[:, ohm_layers] = y[n_pol : n_pol + n_ohm].reshape(self.elements, len(ohm_layers))
        return {"pol_deg": pol, "ohm_deg": ohm}

    def ohmic_resistance(self, T, ohm_deg=None) -> float:
        """Area-specific ohmic resistance of all layers [Ohm m^2], via ``Layer.V_ohm``.

        ``ohm_deg`` is indexed by layer (fuel, air, electrolytes); None means ``m0``.
        """
        total = 0
        for k, layer in enumerate(self.layers()):
            kwargs = {}
            if layer.degradation.ohm_active:
                kwargs["ohm_deg"] = layer.degradation.m0 if ohm_deg is None else ohm_deg[k]
            total += layer.V_ohm(1.0, T, **kwargs)
        return total

    def element_outlet(self, inlet: boundary.BoundaryData, j: float) -> boundary.BoundaryData:
        """Outlet state of one area element at current density ``j`` (no solving).

        Flows change by the electrochemistry and reactions; with the energy
        balance, the heat ``j A (V_tn - V)`` heats both flows (isobaric).
        """
        n_fuel = inlet.n_fuel + self.dn_fuel(j, inlet)
        n_air = inlet.n_air + self.dn_air(j)
        T = inlet.T
        if self.energy_balance:
            capacity = self.heat_capacity_flow(self.gas("fuel"), n_fuel, inlet.T) + (
                self.heat_capacity_flow(self.gas("air"), n_air, inlet.T)
            )
            heat = j * self.area / self.elements * (self.V_thermoneutral(inlet.T) - inlet.V)
            T = inlet.T + heat / capacity
        return boundary.BoundaryData(
            V=inlet.V, j=j, n_fuel=n_fuel, n_air=n_air, T=T, P=inlet.P,
            fuel_pressure=inlet.fuel_pressure, air_pressure=inlet.air_pressure,
        )

    def element_losses(self, inlet: boundary.BoundaryData, j: float, pol_deg=None, ohm_deg=None) -> dict:
        """Voltage breakdown [V] of one area element, see :meth:`nest.cell.Cell.V`.

        Channel pressures are central differences of the element's inlet and
        outlet; reaction-site pressures follow from the electrode transport.

        Returns
        -------
        dict
            ``Ps_fuel``, ``Ps_air`` (channel), ``Ps_fuel_tpb``, ``Ps_air_tpb``,
            ``V_nernst``, ``eta_fuel``, ``eta_air`` (electrode losses) and
            ``eta_electrolyte`` (one per electrolyte layer)
        """
        T = inlet.T

        def layer_kwargs(k):
            kwargs = {}
            if pol_deg is not None:
                kwargs["pol_deg"] = pol_deg[k]
            if ohm_deg is not None:
                kwargs["ohm_deg"] = ohm_deg[k]
            return kwargs

        losses = {}
        channel = {"fuel": self.Ps_fuel(inlet, j), "air": self.Ps_air(inlet, j)}
        for k, side in enumerate(SIDES):
            electrode = self.electrode(side)
            losses[f"Ps_{side}"] = channel[side]
            losses[f"Ps_{side}_tpb"] = electrode.Ps_star(j, T, channel[side])
            losses[f"eta_{side}"] = electrode.V(j, T, Ps=losses[f"Ps_{side}_tpb"], **layer_kwargs(k))
        losses["V_nernst"] = self.V_nernst(T, losses["Ps_fuel_tpb"], losses["Ps_air_tpb"])
        losses["eta_electrolyte"] = np.array(
            [layer.V(j, T, **layer_kwargs(2 + i)) for i, layer in enumerate(self.electrolyte)]
        )
        return losses

    def element_voltage(self, inlet: boundary.BoundaryData, j: float, pol_deg=None, ohm_deg=None) -> float:
        """Cell voltage [V] of one area element: Nernst voltage minus the losses."""
        losses = self.element_losses(inlet, j, pol_deg, ohm_deg)
        return (
            losses["V_nernst"]
            - sum(losses["eta_electrolyte"], 0)
            - losses["eta_fuel"]
            - losses["eta_air"]
        )


# ----------------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------------
def _uniform(length: float, n: int) -> tuple:
    """Faces, centers, and spacing [m] of a uniform grid (spacing 0 for n = 0)."""
    faces = np.linspace(0.0, length, n + 1)
    return faces, (faces[:-1] + faces[1:]) / 2, length / n if n else 0.0


class Grid:
    """Uniform finite-volume grid of a distributed cell, and the state layout.

    Counts default to ``cell.elements`` (along the channel) and each layer's
    ``elements`` (across it).
    """

    FIELDS = ("psi_f", "psi_a", "c_pf", "c_pa", "c_cf", "c_ca", "c_ps")

    def __init__(self, c: Cell, n_x=None, n_zf=None, n_za=None, n_zs=None):
        if not c.distributed:
            raise ValueError("The distributed grid needs cell.fuel_channel and cell.air_channel")
        fuel, air, support = c.electrode_fuel, c.electrode_air, c.fuel_support
        support_thickness = 0.0 if support is None else support.delta
        self.n_x = c.elements if n_x is None else n_x
        self.n_zf = fuel.elements if n_zf is None else n_zf
        self.n_za = air.elements if n_za is None else n_za
        if support_thickness <= 0:
            self.n_zs = 0
        else:
            self.n_zs = support.elements if n_zs is None else n_zs

        self.x_faces, self.x, self.dx = _uniform(c.fuel_channel.length, self.n_x)
        self.x_air_faces, self.x_air, self.dx_air = _uniform(c.air_channel.length, self.n_x)
        self.z_fuel_faces, self.z_fuel, self.dz_fuel = _uniform(fuel.delta, self.n_zf)
        self.z_air_faces, self.z_air, self.dz_air = _uniform(air.delta, self.n_za)
        self.z_support_faces, self.z_support, self.dz_support = _uniform(
            support_thickness, self.n_zs
        )

    def dz(self, side: str) -> float:
        """Cell thickness [m] of an electrode or the fuel support."""
        if side == "support":
            return self.dz_support
        return self.dz_fuel if _check_side(side) == "fuel" else self.dz_air

    @property
    def shapes(self) -> tuple:
        """Field shapes in :attr:`FIELDS` order."""
        return (
            (self.n_zf, self.n_x),
            (self.n_za, self.n_x),
            (2, self.n_zf, self.n_x),
            (2, self.n_za, self.n_x),
            (2, self.n_x),
            (2, self.n_x),
            (2, self.n_zs, self.n_x),
        )

    @property
    def sizes(self) -> list:
        return [int(np.prod(shape)) for shape in self.shapes]

    @property
    def size(self) -> int:
        return sum(self.sizes)

    def pack(self, *fields) -> np.ndarray:
        """Flatten fields (in :attr:`FIELDS` order) into one state vector."""
        return np.concatenate([np.asarray(v, dtype=float).ravel() for v in fields])

    def unpack(self, y) -> tuple:
        """View a state vector as the fields of :attr:`FIELDS`."""
        parts = np.split(np.asarray(y), np.cumsum(self.sizes)[:-1])
        return tuple(v.reshape(shape) for v, shape in zip(parts, self.shapes))


# ----------------------------------------------------------------------------
# Problem (replaces nest.eis.Model)
# ----------------------------------------------------------------------------
class Problem:
    """Cell, boundary conditions, and the state equations of either scenario.

    Voltage is air minus fuel; current is positive in fuel-cell operation.

    * Distributed (both channels given): potentials and gas fields on a
      :class:`Grid`; the steady state is where :meth:`time_derivatives` vanish.
    * Lumped (no channels): one current density per area element; the inlet of
      each element follows from the elements upstream, and each element's
      voltage must equal the applied one.

    In both, column (element) ``k`` only depends on columns ``<= k``.

    Parameters
    ----------
    cell : Cell
        geometry and materials
    boundary : BoundaryData
        temperature, pressures, and inlet flows; wrapped as :class:`Boundary`
    n_x, n_zf, n_za, n_zs : int, optional
        grid counts; default to the cell's and layers' ``elements`` (distributed only)
    material : numpy.ndarray, optional
        degradation ODE state (layout of ``Cell.degradation_state``); None is the fresh cell
        (lumped only)
    """

    def __init__(self, cell: Cell, boundary: boundary.BoundaryData,
                 n_x=None, n_zf=None, n_za=None, n_zs=None, material=None):
        self.cell = cell
        self.boundary = Boundary.from_data(boundary)
        self.grid = Grid(cell, n_x, n_zf, n_za, n_zs) if cell.distributed else None
        self.material = material

    @property
    def distributed(self) -> bool:
        return self.cell.distributed

    @property
    def T(self) -> float:
        return float(self.boundary.T)

    @property
    def n_columns(self) -> int:
        """Columns along the flow: channel cells or area elements."""
        return self.grid.n_x if self.distributed else self.cell.elements

    @property
    def resistance(self) -> float:
        """Ohmic resistance of the current path [Ohm m^2] (dV/dj of the ohmic losses)."""
        if self.distributed:
            return self.cell.resistance(self.T, self.grid.dz_fuel, self.grid.dz_air)
        return self.cell.ohmic_resistance(self.T, self._degradation()["ohm_deg"][0])

    # -- state --
    def pack_state(self, *fields) -> np.ndarray:
        return self.grid.pack(*fields)

    def unpack_state(self, y) -> tuple:
        return self.grid.unpack(y)

    @property
    def state_size(self) -> int:
        return self.grid.size if self.distributed else self.cell.elements

    def initial_state(self) -> np.ndarray:
        """Distributed: inlet gas everywhere, zero overpotentials. Lumped: ``boundary.j`` everywhere."""
        if not self.distributed:
            return np.full(self.cell.elements, float(self.boundary.j))
        shapes, T = self.grid.shapes, self.T
        c_in_f = self.boundary.concentrations("fuel", T)
        c_in_a = self.boundary.concentrations("air", T)
        c_pf = np.broadcast_to(c_in_f[:, None, None], shapes[2])
        c_pa = np.broadcast_to(c_in_a[:, None, None], shapes[3])
        c_cf = np.broadcast_to(c_in_f[:, None], shapes[4])
        c_ca = np.broadcast_to(c_in_a[:, None], shapes[5])
        c_ps = np.broadcast_to(c_in_f[:, None, None], shapes[6])
        psi_f = self.cell.equilibrium_potential("fuel", T, R * T * c_pf)
        psi_a = self.cell.equilibrium_potential("air", T, R * T * c_pa)
        return self.pack_state(psi_f, psi_a, c_pf, c_pa, c_cf, c_ca, c_ps)

    def typical_state(self) -> np.ndarray:
        """Typical magnitude of each state entry (physical reference values).

        Distributed: 0.1 V for potentials and the inlet total concentration for
        gas. Lumped: the boundary current density, at least 1e3 A/m^2.
        """
        if not self.distributed:
            return np.full(self.cell.elements, max(abs(float(self.boundary.j)), 1e3))
        c = [self.boundary.total_concentration(side, self.T) for side in SIDES]
        values = (0.1, 0.1, *c, *c, c[0])  # psi_f, psi_a, c_pf, c_pa, c_cf, c_ca, c_ps
        return np.repeat(values, self.grid.sizes)

    def storage(self) -> np.ndarray:
        """Diagonal of ``M`` in ``M dy/dt = f(y, V)``: storage per active area of each state entry.

        Distributed: charge (``C dz``) or gas (``F eps dz``, ``F`` times channel
        depth) stored per active area of one cell, so ``M dy/dt`` is in A/m^2.
        Lumped: zero, the equations are algebraic.
        """
        if not self.distributed:
            return np.zeros(self.cell.elements)
        c, g = self.cell, self.grid
        charge = [c.capacitance(side) * g.dz(side) for side in SIDES]  # psi_f, psi_a
        electrode_gas = [F * c.porosity(side) * g.dz(side) for side in SIDES]  # c_pf, c_pa
        channel_gas = [F * c.channel_depth(side) for side in SIDES]  # c_cf, c_ca
        support_gas = F * c.porosity("support") * g.dz_support  # c_ps
        return np.repeat((*charge, *electrode_gas, *channel_gas, support_gas), g.sizes)

    def fluxes(self, y, voltage: float, column: int | None = None) -> np.ndarray:
        """Right-hand side ``f`` of ``M dy/dt = f(y, V)`` [A/m^2]; zero at steady state.

        Distributed: net charge or gas flux into each cell per active area,
        ``storage() * time_derivatives``. Lumped: each element's current
        mismatch, ``(V - V_element) / R``.

        Parameters
        ----------
        y : numpy.ndarray
            state vector
        voltage : float
            applied voltage [V]
        column : int, optional
            only return the entries of this column; lumped problems then only
            evaluate that element

        Returns
        -------
        numpy.ndarray
            ``f`` for the whole state, or for one column
        """
        if self.distributed:
            f = self.storage() * self.time_derivatives(0.0, y, voltage)
            return f if column is None else f.reshape(-1, self.n_columns)[:, column]
        resistance = self.resistance or 1.0
        if column is None:
            return self.element_voltage_error(y, voltage) / resistance
        j = np.asarray(y, dtype=float)
        inlet = self.element_states(j[:column], voltage)[-1]
        error = voltage - self.cell.element_voltage(inlet, j[column], **self._degradation(column))
        return np.array([error]) / resistance

    @property
    def downstream_coupling(self) -> bool:
        """True: column ``k`` only depends on columns ``<= k`` (flow from inlet to outlet)."""
        return True

    def open_circuit_voltage(self) -> float:
        """Open-circuit voltage [V] at the inlet gas."""
        if not self.distributed:
            return float(self.cell.element_voltage(self._inlet(0.0), 0.0, **self._degradation(0)))
        return float(np.mean(self._interface_potential(self.initial_state())))

    # -- physics on the state --
    def _interface_potential(self, y) -> np.ndarray:
        """Air minus fuel potential [V] across the electrolyte, per column.

        Uses the electrode cells next to the electrolyte: the last fuel cell and
        the first air cell.
        """
        psi_f, psi_a = self.unpack_state(y)[:2]
        return psi_a[0] - psi_f[-1]

    def current_density(self, y, voltage: float) -> np.ndarray:
        """Local electrolyte current density [A/m^2], one per column."""
        if not self.distributed:
            return np.asarray(y, dtype=float)
        return (self._interface_potential(y) - voltage) / self.resistance

    def voltage_for_current(self, y, current_density: float = 0.0) -> float:
        """Voltage [V] estimate for a state and mean current density (ohmic path)."""
        if not self.distributed:
            return self.open_circuit_voltage() - current_density * self.resistance
        return float(np.mean(self._interface_potential(y)) - current_density * self.resistance)

    # -- full solution layout --
    def fields(self, y, voltage: float) -> dict:
        """Named solution fields; species on the first axis, columns along the flow last.

        Distributed: the state fields of :attr:`Grid.FIELDS` (``(n_z, n_x)`` or
        ``(2, n_z, n_x)``), the current density ``j`` ``(n_x,)``, and the
        overpotentials ``eta_f``/``eta_a`` ``(n_z, n_x)``.

        Lumped: ``j`` per element ``(n,)``; ``T``, ``P``, ``n_fuel``, ``n_air``
        at the element faces ``(..., n + 1)`` (inlet first, outlet last); and per
        element the channel and reaction-site pressures and the voltage
        breakdown (``V_nernst``, ``eta_fuel``, ``eta_air``, ``eta_electrolyte``
        ``(n_electrolyte, n)``), so ``V = V_nernst - sum of losses``.
        """
        if self.distributed:
            fields = dict(zip(Grid.FIELDS, self.unpack_state(y)))
            fields["j"] = self.current_density(y, voltage)
            fields["eta_f"], fields["eta_a"] = self.overpotentials(y)
            return fields
        j = np.asarray(y, dtype=float)
        states = self.element_states(j, voltage)
        losses = [
            self.cell.element_losses(states[e], j_e, **self._degradation(e)) for e, j_e in enumerate(j)
        ]
        fields = {
            "j": j,
            "T": np.array([s.T for s in states], dtype=float),
            "P": np.array([s.P for s in states], dtype=float),
            "n_fuel": np.stack([np.asarray(s.n_fuel, dtype=float) for s in states], axis=-1),
            "n_air": np.stack([np.asarray(s.n_air, dtype=float) for s in states], axis=-1),
        }
        for key in losses[0]:
            fields[key] = np.stack([np.asarray(loss[key], dtype=float) for loss in losses], axis=-1)
        return fields

    def coords(self) -> dict:
        """Coordinates [m] of the fields: cell centers and faces.

        Lumped elements have no geometry; ``x`` is then the normalized position
        along the flow [-].
        """
        if not self.distributed:
            faces = np.linspace(0.0, 1.0, self.cell.elements + 1)
            return {"x": (faces[:-1] + faces[1:]) / 2, "x_faces": faces}
        g = self.grid
        return {
            "x": g.x, "x_faces": g.x_faces,
            "z_fuel": g.z_fuel, "z_fuel_faces": g.z_fuel_faces,
            "z_air": g.z_air, "z_air_faces": g.z_air_faces,
            "z_support": g.z_support, "z_support_faces": g.z_support_faces,
        }

    # -- lumped (channel-less) equations --
    def _degradation(self, element=None) -> dict:
        """``pol_deg``/``ohm_deg`` of all elements, or of one element (per layer)."""
        state = self.cell.degradation_state(self.material)
        return state if element is None else {key: value[element] for key, value in state.items()}

    def _inlet(self, voltage: float) -> boundary.BoundaryData:
        """Inlet state of the first area element at the applied voltage."""
        inlet = copy.copy(self.boundary)
        inlet.V = voltage
        return inlet

    def element_states(self, j, voltage: float) -> list:
        """Inlet state of every area element, plus the cell outlet, for current densities ``j``."""
        states = [self._inlet(voltage)]
        for j_e in np.asarray(j, dtype=float):
            states.append(self.cell.element_outlet(states[-1], j_e))
        return states

    def element_voltage_error(self, j, voltage: float) -> np.ndarray:
        """Applied minus element voltage [V] of every area element."""
        states = self.element_states(j, voltage)
        return np.array([
            voltage - self.cell.element_voltage(states[e], j_e, **self._degradation(e))
            for e, j_e in enumerate(np.asarray(j, dtype=float))
        ])

    def overpotentials(self, y) -> tuple:
        """Activation overpotentials [V] of the fuel and air electrodes."""
        psi_f, psi_a, c_pf, c_pa, *_ = self.unpack_state(y)
        T = self.T
        return tuple(
            psi - self.cell.equilibrium_potential(side, T, R * T * c)
            for side, psi, c in zip(SIDES, (psi_f, psi_a), (c_pf, c_pa))
        )

    def time_derivatives(self, t: float, y, voltage: float) -> np.ndarray:
        """State derivatives [V/s and mol/(m^3 s)] of the coupled equations."""
        psi_f, psi_a, c_pf, c_pa, c_cf, c_ca, c_ps = self.unpack_state(y)
        current = self.current_density(y, voltage)

        dpsi_f_dt, fuel_source = self._electrode_charge_balance(
            "fuel", psi_f, c_pf, end_current=current
        )
        dpsi_a_dt, air_source = self._electrode_charge_balance(
            "air", psi_a, c_pa, start_current=current
        )

        if self.grid.n_zs:
            support_to_fuel_flux = self._support_to_fuel_flux(c_ps, c_pf)
            dc_ps_dt, flux_s = self._porous_gas_balance(
                "support", c_ps, 0.0, channel_concentrations=c_cf, end_flux=support_to_fuel_flux
            )
            dc_pf_dt, _ = self._porous_gas_balance(
                "fuel", c_pf, fuel_source, start_flux=support_to_fuel_flux
            )
            fuel_channel_flux = flux_s[:, 0]
        else:
            dc_ps_dt = np.empty_like(c_ps)  # zero-size field without support
            dc_pf_dt, flux_f = self._porous_gas_balance(
                "fuel", c_pf, fuel_source, channel_concentrations=c_cf
            )
            fuel_channel_flux = flux_f[:, 0]

        dc_pa_dt, flux_a = self._porous_gas_balance(
            "air", c_pa, -air_source, channel_concentrations=c_ca, channel_at_start=False
        )
        air_channel_flux = -flux_a[:, -1]

        dc_cf_dt = self._channel_gas_balance("fuel", c_cf, fuel_channel_flux)
        dc_ca_dt = self._channel_gas_balance("air", c_ca, air_channel_flux)
        return self.pack_state(
            dpsi_f_dt, dpsi_a_dt, dc_pf_dt, dc_pa_dt, dc_cf_dt, dc_ca_dt, dc_ps_dt
        )

    # -- equation kernels --
    def _electrode_charge_balance(self, side, potential, concentrations, *,
                                  start_current=0.0, end_current=0.0):
        """dpsi/dt [V/s] and Faradaic species sources [mol/(m^3 s)] of an electrode."""
        c, T = self.cell, self.T
        spacing = self.grid.dz(side)
        reaction_current = c.faradaic_current(side, potential, T, R * T * concentrations)
        ionic_current = np.empty((potential.shape[0] + 1, self.grid.n_x))
        ionic_current[0], ionic_current[-1] = start_current, end_current
        ionic_current[1:-1] = c.conductivity(side, T) * np.diff(potential, axis=0) / spacing
        potential_rate = (
            np.diff(ionic_current, axis=0) / spacing - reaction_current
        ) / c.capacitance(side)
        return potential_rate, c.species_source(side, reaction_current)

    def _support_to_fuel_flux(self, c_ps, c_pf):
        """Species flux [mol/(m^2 s)] from the support into the fuel electrode."""
        c, gas = self.cell, self.cell.gas("fuel")
        ds, df = self.grid.dz_support / 2, self.grid.dz_fuel / 2
        cs, cf = c_ps[:, -1], c_pf[:, 0]
        face = (df * cs + ds * cf) / (ds + df)
        if c.transport("support") is c.transport("fuel"):
            return c.transport("fuel").dusty_gas_flux(self.T, gas, face, (cf - cs) / (ds + df))
        basis = np.zeros((2, self.grid.n_x, 2))
        basis[0, :, 0], basis[1, :, 1] = 1.0, 1.0
        matrices = []
        for side in ("support", "fuel"):
            response = c.transport(side).dusty_gas_flux(
                self.T, gas, np.broadcast_to(face[..., None], basis.shape), basis
            )
            matrices.append(-np.moveaxis(response, 0, 1))
        resistance = ds * np.linalg.inv(matrices[0]) + df * np.linalg.inv(matrices[1])
        return -np.linalg.solve(resistance, (cf - cs).T[..., None])[..., 0].T

    def _porous_gas_balance(self, side, concentrations, species_source, *,
                            channel_concentrations=None, channel_at_start=True,
                            start_flux=0.0, end_flux=0.0):
        """dc/dt [mol/(m^3 s)] and face fluxes of an electrode or the support (``side``)."""
        gas, transport = self.cell.gas(side), self.cell.transport(side)
        spacing = self.grid.dz(side)
        molar_flux = np.zeros((2, concentrations.shape[1] + 1, self.grid.n_x))
        molar_flux[:, 0], molar_flux[:, -1] = start_flux, end_flux
        if concentrations.shape[1] > 1:
            molar_flux[:, 1:-1] = transport.dusty_gas_flux(
                self.T, gas,
                (concentrations[:, :-1] + concentrations[:, 1:]) / 2,
                (concentrations[:, 1:] - concentrations[:, :-1]) / spacing,
            )
        if channel_concentrations is not None:
            edge, direction = (0, 1) if channel_at_start else (-1, -1)
            molar_flux[:, edge] = transport.dusty_gas_flux(
                self.T, gas,
                (concentrations[:, edge] + channel_concentrations) / 2,
                direction * (concentrations[:, edge] - channel_concentrations) / (spacing / 2),
            )
        dc_dt = (-np.diff(molar_flux, axis=1) / spacing + species_source) / self.cell.porosity(side)
        return dc_dt, molar_flux

    def _channel_gas_balance(self, side, concentrations, electrode_flux):
        """dc/dt [mol/(m^3 s)] of a channel from convection and exchange with the electrode."""
        T = self.T
        pressure = self.boundary.pressure(side)
        spacing = self.cell.channel(side).length / self.grid.n_x
        channel_area = self.cell.channel_area(side)
        molar_flow = self.cell.channel_molar_flow(
            side, self.boundary.total_flow(side), electrode_flux
        )
        velocity = molar_flow * R * T / (pressure * channel_area)
        molar_flux = np.empty((2, self.grid.n_x + 1))
        molar_flux[:, 0] = velocity[0] * self.boundary.concentrations(side, T)
        molar_flux[:, 1:] = velocity[1:] * concentrations
        return (
            -np.diff(molar_flux, axis=1) / spacing
            - self.cell.width_active(side) / channel_area * electrode_flux
        )
