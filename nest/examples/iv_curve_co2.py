"""
Package usage example for generating a iV curve operating with CO-CO2 mixture
"""

from time import process_time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from nest import properties, layers, cell
from nest.constants import R


def iv_curve_co2():
    """
    Example of using the package for generating a iV curve operating with CO-CO2 mixture
    """
    # Define reactants
    fuel_mix = properties.Mixture(
        (properties.BasicSpecies.CO, properties.BasicSpecies.CO2)
    )
    air_mix = properties.Mixture(
        (properties.BasicSpecies.O2, properties.BasicSpecies.N2)
    )
    Ni_YSZ = layers.Layer(
        delta=3e-4 + 10e-6,
        kinetic=layers.ButlerVolmer(
            gas=fuel_mix,
            alpha=0.59,
            beta=1 - 0.59,
            gamma=0.56 * 1.82527e6,
            theta=1,
            nu=np.array([-1, 1]),
            E_act=1.09 / 8.617333262e-5 * 8.314510,
            p=np.array([-0.1, 0.33]),
            n_e=2,
        ),
        transport=layers.BinaryFick(dp=6e-7, epsilon=0.3, tau=3),
    )
    YSZ = layers.Layer(
        delta=12e-6 - 2e-6,
        conductivity=layers.Conductivity(
            sigma0=3.6e7,
            E_act=8e4,
        ),
    )
    YSZ_CGO = layers.Layer(
        delta=4e-6,
        conductivity=layers.Conductivity(sigma0=1715, E_act=8785 * 8.314510, theta=0),
    )
    CGO = layers.Layer(
        delta=10e-6 - 2e-6,
        conductivity=layers.Conductivity(
            sigma0=1.09e7,
            E_act=0.64 / 8.617333262e-5 * 8.314510,
        ),
    )
    LSCF_CGO = layers.Layer(
        delta=3e-5,
        kinetic=layers.ButlerVolmer(
            gas=air_mix,
            alpha=0.65,
            beta=1 - 0.65,
            gamma=1.51556e8,
            nu=np.array([-0.5, 0.0]),
            E_act=1.45 / 8.617333262e-5 * 8.314510,
            p=np.array([0.22, 0.0]),
        ),
        transport=layers.BinaryFick(dp=6e-7, epsilon=0.3, tau=2.8),
    )

    # Define cell
    DTU_cell = cell.Cell(16e-4, Ni_YSZ, (YSZ, YSZ_CGO, CGO), LSCF_CGO)

    # Define boundary conditions
    n_fuel = (24 / 1e3 / 3600) * (1e5 / 8.314510 / 273.15)  # mol/s
    n_air = 140 / 1e3 / 3600 * (1e5 / 8.314510 / 273.15)  # mol/s
    x_CO = 0.6
    x_O2 = 0.21
    conditions = cell.BoundaryData(
        V=1.25,
        j=-1e4,
        n_fuel=np.array([n_fuel * x_CO, n_fuel * (1 - x_CO)]),
        n_air=np.array([n_air * x_O2, n_air * (1 - x_O2)]),
        T=750 + 273.15,
        P=1e5,
    )

    # Solve 1D problem for different voltages
    n = 10
    voltages = np.linspace(1.0, 1.25, n)
    currents = np.zeros(n)
    start_time = process_time()

    activity_max = np.zeros(n)
    for i, V in enumerate(voltages):
        activity = np.zeros(DTU_cell.elements)
        conditions.V = V
        solutions = DTU_cell.solve_for_voltage(conditions)
        currents[i] = np.sum(solutions[1]) / DTU_cell.elements
        # Calculating threshold for carbon formation
        T = conditions.T
        P_0 = 1e5
        Tapp = 0
        K_eq = np.exp(
            -(-128e3 - T * (-133)) / (R * (T - Tapp))
        )  # Duhn - https://orbit.dtu.dk/en/publications/development-of-highly-efficient-solid-oxide-electrolyzer-cell-sys/
        for k in range(DTU_cell.elements):
            j = solutions[1][i]
            n_fuel = np.zeros(len(conditions.n_fuel))
            for z in range(len(n_fuel)):
                n_fuel[z] = solutions[4 + z][k]
            Ps = conditions.P * n_fuel / sum(n_fuel)
            Ps_star = Ni_YSZ.Ps_star(j, T, Ps)
            # Calculate the activity for the TPB
            P_CO = Ps_star[0] / P_0
            P_CO2 = Ps_star[1] / P_0
            activity[k] = K_eq * P_CO**2 / P_CO2
        activity_max[i] = max(activity)
    print(f"Computation time : {process_time() - start_time} seconds")

    # Plotting iv curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8))
    ax1.plot(
        currents / 1e4, voltages, marker="o", markersize=5, label="Model", color="r"
    )
    ax1.set_xlabel("Current density (A/m^2)")
    ax1.set_ylabel("Cell voltage (V)")
    # Add experimental data for comparison - https://doi.org/10.1016/j.jpowsour.2017.10.097
    script_dir = Path(__file__).parent
    data = pd.read_csv(script_dir / "data/750C_60CO_100AIR.csv")
    ax1.scatter(
        data["j"], data["V"], label="Experiment", edgecolors="b", facecolor="none"
    )
    ax1.set_title("Current x voltage")
    ax1.legend()

    # Plotting carbon threshold
    ax2.plot(
        currents / 1e4, activity_max, marker="o", markersize=5, label="Model", color="r"
    )
    ax2.axvline(x=-0.35, ls="--", label="Experiment limit")
    ax2.set_title("Carbon threshold [a>=1]")
    ax2.set_ylabel("Carbon deposition activity, a")
    ax2.set_xlabel("Current density (A/m^2)")
    ax2.legend()

    plt.suptitle("60%CO+50%CO2 | 21%O2+79%N2 | 750 degC", fontsize=14)
    return plt.show()


if __name__ == "__main__":
    iv_curve_co2()
