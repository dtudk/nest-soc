from time import process_time
import numpy as np
import pandas as pd
from pathlib import Path

from nest import properties, layers, cell, degradation

import matplotlib.pyplot as plt


def durability_test():
    """
    Example of using the package to predict the voltage drop due to degradation
    """
    # Define reactants
    fuelMix = properties.Mixture(
        (properties.BasicSpecies.H2, properties.BasicSpecies.H2O)
    )
    airMix = properties.Mixture((properties.BasicSpecies.O2))

    # Define cell layers
    Ni_YSZ = layers.Layer(
        delta=1000e-6 + 10e-6,
        kinetic=layers.ButlerVolmer(
            fuelMix,
            nu=np.array([-1, 1]),
            n_e=2,
            alpha=0.59,
            beta=1 - 0.59,
            gamma=1.82527e6 / 3.1286E+12,
            p=np.array([-0.1, 0.33]),
            theta=1,
            E_act=1.09 / 8.617333262e-5 * 8.314510,
        ),
        transport=layers.BinaryFick(
            dp=600e-9,
            epsilon=0.3,
            tau=2,
        ),
        degradation=degradation.NickelAgglomeration(
            psi_ed=0.4,
            psi_el=0.6,
            r0_ed=300e-9,
            r0_el=350e-9,
            epsilon=0.3,
            alpha=3.04e8,
            r_ed_max=500e-9,
        ),
    )
    YSZ = layers.Layer(
        delta=8e-6 - 1.5e-6,
        conductivity=layers.Conductivity(
            sigma0=3.6e7,
            E_act=8e4,
        ),
    )
    YSZ_CGO = layers.Layer(
        delta=3e-6,
        conductivity=layers.Conductivity(sigma0=1715, E_act=8785 * 8.314510, theta=0),
    )
    CGO = layers.Layer(
        delta=5e-6 - 1.5e-6,
        conductivity=layers.Conductivity(
            sigma0=1.09e7,
            E_act=0.64 / 8.617333262e-5 * 8.314510,
        ),
    )
    LSCF_CGO = layers.Layer(
        delta=32e-6,
        kinetic=layers.ButlerVolmer(
            airMix,
            alpha=0.65,
            beta=1 - 0.65,
            gamma=1.51556e8,
            nu=np.array([-0.5]),
            E_act=1.45 / 8.617333262e-5 * 8.314510,
            p=np.array([0.22]),
        ),
        transport=layers.BinaryFick(dp=6e-7, epsilon=0.3, tau=2.8),
        degradation=degradation.ChromiumPoison(
            x_H2O=0.01,
            j0=4.27254e7,
        ),
    )
    Crofer22 = layers.Layer(
        delta=1, conductivity=layers.Conductivity(sigma0=1 / (0.2e-5), theta=0)
    )
    CrScale = layers.Layer(
        delta=0,
        conductivity=layers.Conductivity(sigma0=45.8, E_act=33.3e3, theta=0),
        degradation=degradation.InterconnectOxidation(
            k_mg=0.02361158401133492, E_ox=260_000
        ),
    )

    # Define cell
    cellModel = cell.Cell(
        16e-4, Ni_YSZ, (YSZ, YSZ_CGO, CGO, Crofer22, CrScale), LSCF_CGO, elements=10
    )

    # Define boundary conditions
    n_fuel = 8.23e-05  # mol/s
    n_air = 0.000345476  # mol/s
    x_H2 = 0.94
    x_O2 = 0.21
    conditions = cell.BoundaryData(
        V=0.9,
        j=0.7e4,
        n_fuel=np.array([n_fuel * x_H2, n_fuel * (1 - x_H2)]),
        n_air=np.array([n_air * x_O2, n_air * (1 - x_O2)]),
        T=700 + 273.15,
        P=1e5,
    )

    # Solve ODE problem
    start_time = process_time()
    solution = cellModel.solve_for_time(conditions, "current", 5000)
    print(f"Computation time : {process_time() - start_time} seconds")

    """
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(6.4*3,4.8))
    
    r_pol_fuel = np.zeros((len(solution.t),cellModel.elements))
    r_pol_air = np.zeros((len(solution.t),cellModel.elements))
    r_ohm = np.zeros((len(solution.t),cellModel.elements))

    l = np.array([1/cellModel.elements*(i+1) - 0.5*1/cellModel.elements for i in range(cellModel.elements)])
    r_pol_fuel_0 = Ni_YSZ.degradation.pol_deg(Ni_YSZ.degradation.m0)
    # Breakaway corrosion
    rho_Cr2O3 = 5220

    M_o = 15.9994   # g/mol
    M_Cr = 51.9961  # g/mol
    
    C_breakaway = 369.01 * (conditions.T**(-1.205)) * 100
    h_IC = 3e-4#2.5e-3
    L_cell = 4e-2
    w_cell = 4e-2
    n_segment = cellModel.elements
    h = h_IC / (1 + h_IC/(L_cell/n_segment) + h_IC/w_cell)       # Eq.(6.5) Reddy thesis and https://doi.org/10.1002/(SICI)1521-4176(200004)51:4%3C224::AID-MACO224%3E3.0.CO;2-B
    C_0 = 22.92
    rho_interconnect = 7700 # kg/m3
    h_max = (C_0 - C_breakaway)/100 * rho_interconnect * h/2 * (3/2) * (M_o / M_Cr) / rho_Cr2O3

    for t in range(len(solution.t)):
        for i in range(cellModel.elements):
            r_pol_fuel[t][i] = Ni_YSZ.degradation.pol_deg(solution.y[i*2][t])/r_pol_fuel_0
            r_pol_air[t][i] = solution.y[1+i*2][t]
            r_ohm[t][i] = 1-solution.y[2*cellModel.elements+i][t]/h_max
    
    for t in range(len(solution.t)):
        if solution.t[t] > 0 and solution.t[t] < 1:
            pass
        else:
            ax1.plot(l,r_pol_fuel[t],label=f"{round(solution.t[t])} h")
            ax1.set_xlabel("Normalized cell length [-]")
            ax1.set_ylabel("Fuel electrode area ratio [-]")

            ax2.plot(l,r_pol_air[t],label=f"{round(solution.t[t])} h")
            ax2.set_xlabel("Normalized cell length [-]")
            ax2.set_ylabel("Air electrode area ratio [-]")

            ax3.plot(l,r_ohm[t],label=f"{round(solution.t[t])} h")
            ax3.set_xlabel("Normalized cell length [-]")
            ax3.set_ylabel("Normalized corrosion thickness [-]")


    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    """

    """
    fig, ax = plt.subplots(figsize=(6.4*3,4.8))
    # Solve 1D problem for material conditions
    conditions.V = 0.75
    y_matrix = np.transpose(solution.y)
    V = np.zeros(len(y_matrix))
    for i, y in enumerate(y_matrix):
        steady_solution = cellModel.solve_time_step(y, conditions, "current")
        V[i] = np.sum(steady_solution[0]) / cellModel.elements
        conditions.V = V[i]

    ax.plot(solution.t, V, marker="o", markersize=5, label="Model", color="r")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Cell voltage (V)")

    # Add experimental data for comparison
    script_dir = Path(__file__).parent
    data = pd.read_csv(script_dir / "data/700C_75FU_40AIR.csv")
    ax.scatter(
        data["t"], data["V"], label="Experiment", edgecolors="b", facecolor="none"
    )
    plt.title("94%H2+6%H2O | 21%O2+79%N2 | 700 degC", fontsize=14, loc="center")
    plt.legend()
    """

    # Solve 1D problem for material conditions

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    conditions.V = 0.75
    y_matrix = np.transpose(solution.y)
    eta_fuel = np.zeros(len(y_matrix))
    eta_air = np.zeros(len(y_matrix))
    eta_ohm = np.zeros(len(y_matrix))
    ocv = np.zeros(len(y_matrix))
    for i, y in enumerate(y_matrix):
        steady_solution = cellModel.solve_time_step(y, conditions, "current")
        eta_fuel[i] = np.sum(steady_solution[8]) / cellModel.elements
        eta_air[i] = np.sum(steady_solution[9]) / cellModel.elements
        eta_ohm[i] = np.sum(steady_solution[10:]) / cellModel.elements
        ocv[i] = (
            np.sum(steady_solution[0]) / cellModel.elements
            + eta_fuel[i]
            + eta_air[i]
            + eta_ohm[i]
        )

        conditions.V = np.sum(steady_solution[0]) / cellModel.elements

    ax.plot(solution.t, ocv, label="OCV")
    ax.plot(solution.t, ocv - eta_ohm, label="Ohmic")
    ax.fill_between(solution.t, ocv, ocv - eta_ohm, alpha=0.5, color="C1")
    ax.plot(solution.t, ocv - eta_ohm - eta_fuel, label="Fuel")
    ax.fill_between(
        solution.t, ocv - eta_ohm, ocv - eta_ohm - eta_fuel, alpha=0.5, color="C2"
    )
    ax.plot(solution.t, ocv - eta_ohm - eta_fuel - eta_air, label="Air")
    ax.fill_between(
        solution.t,
        ocv - eta_ohm - eta_fuel,
        ocv - eta_ohm - eta_fuel - eta_air,
        alpha=0.5,
        color="C3",
    )
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Voltage (V)")

    script_dir = Path(__file__).parent
    data = pd.read_csv(script_dir / "data/700C_75FU_40AIR.csv")
    ax.scatter(
        data["t"], data["V"], label="Experiment", edgecolors="b", facecolor="none"
    )

    plt.title("94%H2+6%H2O | 21%O2+79%N2 | 700 degC", fontsize=14, loc="center")
    plt.legend()

    return plt.show()


if __name__ == "__main__":
    durability_test()
