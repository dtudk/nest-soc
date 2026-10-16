# nest-soc-eis

**Fast-transient and impedance modeling of solid oxide cells.**

`nest-soc-eis` extends [NEST-SOC](https://github.com/dtudk/nest-soc) material and
thermodynamic inputs with spatially resolved electrode charge, gas transport,
and gas storage. It supports steady-state operation, exponential voltage ramps,
current interruption, and impedance extraction from transient responses.

## Installation

Install from PyPI:

```console
pip install nest-soc-eis
```

Or install from the source folder containing `pyproject.toml`:

```console
python -m pip install -e .
```

Python 3.11 or later is required. Pip installs `nest-soc`, NumPy, SciPy,
Matplotlib, and openpyxl automatically. Import the package as `nesteis`.

Check the installed version and the packaged tutorial location:

```python
from importlib.metadata import version
from importlib.resources import files

version("nest-soc-eis"), str(files("nesteis").joinpath("examples/tutorials"))
```

## Model and numerical method

The model resolves both co-flow channels along their length and each electrode
and fuel support through its thickness. It couples Butler–Volmer reactions,
volumetric double-layer capacitance, ionic conduction, and dusty-gas transport.

The model assumes constant temperature, fixed channel pressures, uniform electronic
potential within each electrode, and binary H₂/H₂O and O₂/N₂ mixtures. 
Channel transport is upwind, with no axial diffusion or flow reversal.

Finite-volume balances provide the state time derivatives, `dy/dt = g(y, V)`.
At steady state, the solver finds `g(y, V) = 0`, solving coupled potentials and
concentrations one channel column at a time from inlet to outlet.

## Basic workflow

Define a `Model` using NEST-SOC's `Cell`, `Layer`, and `BoundaryData`, together
with `CellEIS`, `ElectrodeEIS`, and `ChannelGeometry`. The
[tutorial](src/nesteis/examples/tutorials/tutorial.ipynb)
provides a complete model definition.

```python
import numpy as np
from nesteis import solve_steady, solve_transient, voltage_ramp, save_results

ocv = model.voltage_for_current(model.initial_state())
voltage_before = ocv + 1e-3
steady = solve_steady(model, voltage_before)

protocol = voltage_ramp(initial=voltage_before, final=ocv, tau=2e-7)
times = np.r_[0, np.geomspace(1e-9, 30.0, 600)]
solution = solve_transient(model, protocol, (0, 30.0), steady.x, t_eval=times)

save_results(model, "voltage_ramp.xlsx", steady=(steady, voltage_before),
             solution=solution, protocol=protocol)
```

## Tutorial

The [step-by-step tutorial](src/nesteis/examples/tutorials/tutorial.ipynb) covers
model definition, steady-state operation, exponential voltage ramps, current
interruption, impedance extraction, plotting, and saving results.

Run its cells in order. The notebook is included in the installed package;
the installation check above gives its folder location. Results are saved in
`tutorial_results/` under the notebook's working directory.

## Results and conventions

All model inputs use SI units. Voltage is **air minus fuel** [V]; current density
is **positive in fuel-cell operation** [A/m²]. Impedance is area-specific,
`Z = -dV/dJ` [Ω·m²], with Nyquist coordinates `Z.real` and `-Z.imag`.

`plot_transient()` saves voltage, current density, overpotential, or species
mole-fraction responses. Select the channel mean, inlet, center, or outlet;
species can be sampled in the channel or the electrode cell nearest the electrolyte.
`plot_impedance()` saves Nyquist and frequency plots and can overlay an experimental
file with columns frequency [Hz], Re(Z), and −Im(Z) [Ω·m²]. Experimental points
can be plotted if supplied.

`save_results()` exports model inputs, grid coordinates, steady and transient
spatial fields, terminal values, and supplied impedance results to Excel.

Solver tolerance measures convergence of the discrete balances; it does not
establish mesh accuracy. Check spatial resolution, time resolution, tolerances,
and relaxation duration for the response of interest. Local impedance requires
small perturbations; a large interruption can include nonlinear effects.

Developed by Javid Beyrami and Rafael Nogueira Nakashima.


## License


