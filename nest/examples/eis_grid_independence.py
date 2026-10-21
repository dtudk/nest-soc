"""grid independence study for the EIS model, at 750 °C and thermoneutral voltage."""

import matplotlib.pyplot as plt
import numpy as np

from nest.boundary import BoundaryData
from nest.cell import Cell, ChannelGeometry
from nest.layers import ButlerVolmer, Conductivity, Layer, PorousTransport
from nest.properties import BasicSpecies, Mixture
from nest.eis import Model, solve_steady
from nest.constants import R


BASE_GRID = dict(n_x=30, n_zf=9, n_za=9, n_zs=3)
GRID_LEVELS = {
    "n_x": [10, 20, 30, 40, 50, 60],
    "n_zf": [3, 6, 9, 12, 15, 18],
    "n_za": [3, 6, 9, 12, 15, 18],
    "n_zs": [3, 6, 9, 12, 15, 18],
}


def reference_case():

    porosity = 0.3
    tortuosity = 1.23 * (1 - porosity)**(4 / 3) / porosity
    transport = PorousTransport(dp=600e-9, epsilon=porosity, tau=tortuosity)
    fuel = Layer(
        delta=10e-6, transport=transport,
        kinetic=ButlerVolmer(
            gas=Mixture((BasicSpecies.H2, BasicSpecies.H2O)), nu=[-1, 1],
            n_e=2, alpha=0.59, beta=0.41, theta=1,
            exchange_current_prefactor=6.044e1,
            E_act=1.600e5, p=[-0.1, 0.33],
        ),
        conductivity=Conductivity(sigma0=3.6e7 * (0.58 * 0.7 / 2.8),
                                  E_act=8e4, theta=-1),
        tpb_density=3.100e12,
        capacitance_volumetric=3.954e6 * 3.594e-1,
    )
    air = Layer(
        delta=30e-6, transport=transport,
        kinetic=ButlerVolmer(
            gas=Mixture((BasicSpecies.O2, BasicSpecies.N2)), nu=[-0.5, 0],
            n_e=2, alpha=0.65, beta=0.35, theta=1,
            exchange_current_prefactor=1e1,
            E_act=1.496e5, p=[0.22, 0],
        ),
        conductivity=Conductivity(sigma0=1.09e7 * (0.5 * 0.7 / 1.8),
                                  E_act=61.75e3, theta=-1),
        tpb_density=3.467e12,
        capacitance_volumetric=4.855e5 * 1.574e2,
    )
    cell = Cell(
        area=0.04 * 0.04, electrode_fuel=fuel, electrode_air=air,
        electrolyte=(
            Layer(delta=12e-6, conductivity=Conductivity(sigma0=3.6e7, E_act=8e4, theta=-1)),
            Layer(delta=8e-6, conductivity=Conductivity(sigma0=1.09e7, E_act=61.75e3, theta=-1)),
            Layer(delta=1e-6, conductivity=Conductivity(sigma0=0.2312, E_act=6.736e4, theta=1)),
        ),
        fuel_channel=ChannelGeometry(length=0.04, width=0.04, height=1e-3),
        air_channel=ChannelGeometry(length=0.04, width=0.04, height=1e-3),
        fuel_support=Layer(delta=300e-6, transport=transport),
    )
    # Standard flows: 1 bar, 273.15 K.
    fuel_flow = 24e-3 / 3600 * 1e5 / (R * 273.15)
    air_flow = 50e-3 / 3600 * 1e5 / (R * 273.15)
    boundary = BoundaryData(
        n_fuel=fuel_flow * np.array([0.5, 0.5]),
        n_air=air_flow * np.array([0.21, 0.79]),
        T=750+273.15, P=101325,
    )
    return cell, boundary


def main():
    cell, boundary = reference_case()
    voltage = 1.285     # [V]
    rows = []

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, (parameter, levels) in zip(axes.flat, GRID_LEVELS.items()):
        cases = []
        for n in levels:
            grid = BASE_GRID | {parameter: n}
            model = Model(cell, boundary, **grid)
            try:
                result = solve_steady(model, voltage=voltage, tol=1e-6)
            except RuntimeError as error:
                raise RuntimeError(f"Grid {grid}, voltage={voltage} V: {error}") from error
            if not result.success:
                raise RuntimeError(f"Steady solve failed for {grid}")

            _, _, c_pf, c_pa, c_cf, c_ca, _ = model.unpack_state(result.x)
            eta_f, eta_a = model.overpotentials(result.x)

            # Normalize locally before averaging along the interface.
            x_h2_interface = c_pf[0, -1, :] / c_pf[:, -1, :].sum(axis=0)
            x_o2_interface = c_pa[0, 0, :] / c_pa[:, 0, :].sum(axis=0)
            x_h2_outlet = c_cf[0, -1] / c_cf[:, -1].sum()
            x_o2_outlet = c_ca[0, -1] / c_ca[:, -1].sum()

            values = {
                "Current [A/m^2]": np.mean(model.current_density(result.x, voltage)),
                "Fuel activation overpotential [V]": np.mean(eta_f),
                "Air activation overpotential [V]": np.mean(eta_a),
                "Fuel interface x_H2 [-]": np.mean(x_h2_interface),
                "Air interface x_O2 [-]": np.mean(x_o2_interface),
                "Fuel outlet x_H2 [-]": x_h2_outlet,
                "Air outlet x_O2 [-]": x_o2_outlet,
            }
            cases.append(values)
            rows.append({"sweep": parameter, **grid, "T_K": boundary.T,
                         "voltage_V": voltage, **values})

        print(f"\n{parameter}={levels}: differences from finest grid [%]", flush=True)
        for name in cases[-1]:
            values = np.array([case[name] for case in cases])
            reference = values[-1]
            # Relative differences are undefined for a vanishing reference.
            errors = (100 * np.abs(values - reference) / abs(reference)
                      if abs(reference) > 1e-12 else np.full(values.shape, np.nan))
            formatted = ", ".join(f"{error:.4f}" for error in errors)
            print(f"  {name}: [{formatted}]", flush=True)
            ax.plot(levels, errors, "o-", label=name)
        ax.set(xlabel=parameter, ylabel="Difference from finest [%]")
        ax.grid(alpha=0.3)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=8)

    plt.show()


if __name__ == "__main__":
    main()
