# Legend

- **[A]** Abstract class
- **[C]** Concrete class
- **[P]** Property / attribute
- **[M]** Method (function belonging to a class)
- **[F]** Standalone function
- **`←`** Inherits from

# possibilities for layer

```text
Layer [A]
├── Channel [C]
├── PorousLayer [C]
│   └── Electrode [C]
└── DenseLayer [C]
    ├── Electrolyte [C]
    ├── Barrier [C]
    └── Interconnect [C]
```

```text
Layer [A]
├── Channel [C]
├── PorousLayer [C]
│   └── Electrode [C]
├── Electrolyte [C]
├── Barrier [C]
└── Interconnect [C]
```

```text
Layer [A]
├── Channel [C]
├── Electrode [C]
├── FuelSupport [C]
├── Electrolyte [C]
├── Barrier [C]
└── Interconnect [C]
```

```text
Layer [C]
├── Channel (instance)
├── FuelSupport (instance)
├── Electrode (instance)
├── Electrolyte (instance)
├── Barrier (instance)
└── Interconnect (instance)
```


# problem structure

```text
Problem [C]
│
├── cell: Cell [C]
│   │
│   ├── fuel_channel: Channel [C] ← Layer [A]
│   │   ├── geometry: ChannelGeometry [C]
│   │   ├── transport: ChannelTransportLaw [A]
│   │   │   └── flux() [M]
│   │   ├── thickness [P] (from Channel; geometry.height)
│   │   └── grid: Grid [C] | None [P] (from Layer)
│   │
│   ├── fuel_support: PorousLayer [C] ← Layer [A]
│   │   ├── thickness [P] (from PorousLayer)
│   │   ├── porosity, tortuosity, pore_radius [P] (from PorousLayer)
│   │   ├── transport: TransportLaw [A] (from PorousLayer)
│   │   │   └── flux() [M]
│   │   └── grid: Grid [C] | None [P] (from Layer)
│   │
│   ├── fuel_electrode: Electrode [C] ← PorousLayer [C] ← Layer [A]
│   │   ├── thickness, porosity, tortuosity, pore_radius [P] (from PorousLayer)
│   │   ├── transport: TransportLaw [A] (from PorousLayer)
│   │   │   └── flux() [M]
│   │   ├── grid: Grid [C] | None [P] (from Layer)
│   │   ├── reaction: ElectrochemicalReaction [C]
│   │   │   ├── equilibrium_potential() [M]
│   │   │   └── species_source() [M]
│   │   ├── kinetics: KineticLaw [A]
│   │   │   └── current() [M]
│   │   ├── conductivity: ConductivityLaw [A]
│   │   │   └── sigma() [M]
│   │   ├── double_layer: DoubleLayer [C] | None
│   │   │   └── capacitive_current() [M]
│   │   └── degradation: DegradationLaw [A] | None
│   │       └── rate() [M]
│   │
│   ├── electrolyte: Electrolyte [C] ← DenseLayer [C] ← Layer [A]
│   │   ├── thickness [P] (from DenseLayer)
│   │   ├── conductivity: ConductivityLaw [A] (from DenseLayer)
│   │   │   └── sigma() [M]
│   │   ├── grid: Grid [C] | None [P] (from Layer)
│   │   └── resistance() [M]
│   │
│   ├── barrier: Barrier [C] ← DenseLayer [C] ← Layer [A]
│   │   ├── thickness [P] (from DenseLayer)
│   │   ├── conductivity: ConductivityLaw [A] (from DenseLayer)
│   │   │   └── sigma() [M]
│   │   └── grid: Grid [C] | None [P] (from Layer)
│   │
│   ├── air_electrode: Electrode [C] ← PorousLayer [C] ← Layer [A]
│   │   └── Same structure as fuel_electrode
│   │
│   ├── air_channel: Channel [C] ← Layer [A]
│   │   └── Same structure as fuel_channel
│   │
│   ├── interconnect: Interconnect [C] ← DenseLayer [C] ← Layer [A]
│   │   ├── thickness [P] (from DenseLayer)
│   │   ├── conductivity: ConductivityLaw [A] (from DenseLayer)
│   │   │   └── sigma() [M]
│   │   ├── grid: Grid [C] | None [P] (from Layer)
│   │   ├── degradation: DegradationLaw [A] | None
│   │   │   └── rate() [M]
│   │   └── resistance() [M]
│   │
│   └── additional_layers: list[Layer] [P] (optional)
│       └── User-defined concrete subclasses of Layer [A]
│
├── boundary: BoundaryConditions [C]
│   ├── fuel_inlet: Inlet [C]
│   │   ├── temperature [P]
│   │   ├── pressure [P]
│   │   ├── composition [P]
│   │   └── flow_rate [P]
│   ├── air_inlet: Inlet [C]
│   │   ├── temperature [P]
│   │   ├── pressure [P]
│   │   ├── composition [P]
│   │   └── flow_rate [P]
│   └── outlet_conditions: OutletConditions [C]
│
├── circuit: ExternalCircuit [C] | None
│   └── derivatives() [M]
│
├── steady: SteadyStateAnalysis [C] ← Analysis [A]
│   └── solve() [M]
│
├── transient: TransientAnalysis [C] ← Analysis [A]
│   ├── electrochemistry: ElectrochemicalModel [C]
│   ├── thermal: ThermalModel [C] | None
│   ├── degradation: DegradationModel [C] | None
│   ├── impedance_protocol: EISProtocol [A]
│   └── solve() [M]
│
└── eis: EISAnalysis [C] ← Analysis [A]
    ├── solve() [M]
    └── impedance() [M]
```

# extendable 

```text
KineticLaw [A]
├── ButlerVolmer [C]
└── ... (user-defined subclasses)

ConductivityLaw [A]
├── ArrheniusConductivity [C]
├── ConstantConductivity [C]
└── ... (user-defined subclasses)

TransportLaw [A]
├── BinaryFick [C]
├── StefanMaxwell [C]
├── DustyGas [C]
└── ... (user-defined subclasses)

ChannelTransportLaw [A]
├── PlugFlow [C]
├── AdvectionDiffusion [C]
└── ... (user-defined subclasses)

DegradationLaw [A]
├── NiMigration [C]
├── NiCoarsening [C]
├── InterconnectOxidation [C]
└── ... (user-defined subclasses)

Analysis [A]
├── SteadyStateAnalysis [C]
├── TransientAnalysis [C]
├── EISAnalysis [C]
└── ... (user-defined analyses)

Protocol [A]
├── VoltageStep [C]
├── CurrentStep [C]
└── ... (user-defined protocols)
```