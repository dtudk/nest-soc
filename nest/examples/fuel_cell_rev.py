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
    fuel_mix = properties.Mixture(
        (
            properties.BasicSpecies.H2,
            properties.BasicSpecies.H2O,
            properties.BasicSpecies.CO2,
            properties.BasicSpecies.CO,
        )
    )
    airMix = properties.Mixture((properties.BasicSpecies.O2, properties.BasicSpecies.N2))

    # Define cell layers
    Ni_YSZ = layers.Layer(
        delta=300e-6 + 10e-6,
        kinetic=layers.ButlerVolmer(
            gas=fuel_mix,
            alpha=0.59,
            beta=1 - 0.59,
            #gamma= 1.82527e6,
            gamma= 1.82527e6/ 4.19e12,
            theta=1,
            nu=np.array([-1, 1, 0, 0]),
            E_act=1.09 / 8.617333262e-5 * 8.314510,
            p=np.array([-0.1, 0.33, 0.0, 0.0]),
            n_e=2,
        ),
        transport=layers.StefanMaxwell(
            dp=600e-9,
            epsilon=0.3,
            tau=2,
        ),
        degradation=degradation.NickelAgglomeration(
            psi_ed=0.4,
            psi_el=0.6,
            r0_ed=250e-9,
            r0_el=350e-9,
            epsilon=0.3,
            alpha=3.04e8,
            r_ed_max=500e-9,
            #pol_active=False,
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
        #delta=32e-6,
        delta=15e-6,
        kinetic=layers.ButlerVolmer(
            airMix,
            alpha=0.65,
            beta=1 - 0.65,
            gamma=1.51556e8,
            nu=np.array([-0.5,0]),
            E_act=1.45 / 8.617333262e-5 * 8.314510,
            p=np.array([0.22,0]),
        ),
        transport=layers.BinaryFick(dp=6e-7, epsilon=0.3, tau=2.8),
        degradation=degradation.ChromiumPoison(
            x_H2O=0.002,
            j0=4.27254e7,
            E_act=1.45 / 8.617333262e-5 * 8.314510,
            #pol_active=False,
        ),
    )
    Crofer22 = layers.Layer(
        delta=1, conductivity=layers.Conductivity(sigma0=1 / (0.2e-5), theta=0)
    )
    CrScale = layers.Layer(
        delta=0,
        conductivity=layers.Conductivity(sigma0=45.8, E_act=33.3e3, theta=0),
        degradation=degradation.InterconnectOxidation(
            k_mg=0.02361158401133492, E_ox=260_000, #ohm_active=False,
        ),
    )

    # Define cell
    WGS = cell.WaterGasShift(
        k=0.0171 * 3e-4,
        E=103191,
        nu=np.array([1.0, -1.0, 1.0, -1.0]),
        index=np.array([0, 1, 2, 3]),
    )
    DTU_cell = cell.Cell(16e-4, Ni_YSZ, (YSZ, YSZ_CGO, CGO, Crofer22, CrScale), LSCF_CGO, elements=10, reactions=(WGS,))

    n_fuel = (10 / 1e3 / 3600) * (1e5 / 8.314510 / 273.15)  # mol/s
    n_air = 1.5*60 / 1e3 / 3600 * (1e5 / 8.314510 / 273.15)  # mol/s

    x_H2 = 0.36
    x_H2O = 0.31
    x_CO2 = 0.22
    x_CO = 0.12
    
    x_O2 = 0.21
    conditions = cell.BoundaryData(
        V=0.8,
        j=0.5e4,
        n_fuel=np.array([n_fuel * x_H2, n_fuel * x_H2O, n_fuel * x_CO2, n_fuel * x_CO]),
        n_air=np.array([n_air * x_O2, n_air*(1-x_O2)]),
        T=650 + 273.15,
        P=1e5,
    )

    # Solve 1D problem for different voltages
    """
    n = 10
    voltages = np.linspace(0.75, 1.01, n)
    currents = np.zeros(n)
    start_time = process_time()
    for i, V in enumerate(voltages):
        conditions.V = V
        solutions = DTU_cell.solve_for_voltage(conditions)
        currents[i] = np.sum(solutions[1]) / DTU_cell.elements
    print(f"Computation time : {process_time() - start_time} seconds")

    fig, ax = plt.subplots()
    ax.plot(
        currents / 1e4, voltages, marker="o", markersize=5, label="Model", color="r"
    )
    ax.set_xlabel("Current density (A/m^2)")
    ax.set_ylabel("Cell voltage (V)")

    # Add experimental data for comparison
    script_dir = Path(__file__).parent
    data = pd.read_csv(script_dir / "data/754C_10H2_40H2O_50CO2.csv")
    ax.scatter(
        data["j"], data["V"], label="Experiment", edgecolors="b", facecolor="none"
    )
    plt.legend()
    plt.title("10%H2+40%H2O+50%CO2 | 100% O2 | 754 degC", fontsize=14, loc="center")
    return plt.show()
    """
    
    # Solve ODE problem
    start_time = process_time()
    #solution = DTU_cell.solve_for_time(conditions, "current", 33000)
    #solution = DTU_cell.solve_for_time(conditions, "current", 40000)
    solution = DTU_cell.solve_for_time(conditions, "current", 40000)
    print(f"Computation time : {process_time() - start_time} seconds")

    script_dir = Path(__file__).parent
    np.savetxt(script_dir / "data/ivp.csv", np.column_stack((solution.t,np.transpose(solution.y))), delimiter=",", comments="")


    # Solve 1D problem for material conditions
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    conditions.V = 0.70
    conditions.j = 0.5e4
    conditions.T = 650 + 273.15
    y_matrix = np.transpose(solution.y)
    eta_fuel = np.zeros(len(y_matrix))
    eta_air = np.zeros(len(y_matrix))
    eta_ohm = np.zeros(len(y_matrix))
    ocv = np.zeros(len(y_matrix))
    for i, y in enumerate(y_matrix):
        #conditions.T = 650 + 273.15 + 10/10000*solution.t[i]
        steady_solution = DTU_cell.solve_time_step(y, conditions, "current")
        eta_fuel[i] = np.sum(steady_solution[10]) / DTU_cell.elements
        eta_air[i] = np.sum(steady_solution[11]) / DTU_cell.elements
        eta_ohm[i] = np.sum(steady_solution[12:]) / DTU_cell.elements
        ocv[i] = (
            np.sum(steady_solution[0]) / DTU_cell.elements
            + eta_fuel[i]
            + eta_air[i]
            + eta_ohm[i]
        )

        conditions.V = np.sum(steady_solution[0]) / DTU_cell.elements
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

    #data = pd.read_csv(script_dir / "data/700C_75FU_40AIR.csv")
    #ax.scatter(
    #    data["t"], data["V"], label="Experiment", edgecolors="b", facecolor="none"
    #)

    np.savetxt(script_dir / "data/overpotentials.csv", np.column_stack((solution.t, ocv, eta_ohm, eta_fuel, eta_air)), delimiter=",", header="time,ocv,eta_ohm,eta_fuel,eta_air", comments="")

    #plt.title("94%H2+6%H2O | 21%O2+79%N2 | 700 degC", fontsize=14, loc="center")
    plt.legend()
    

    return plt.show()


if __name__ == "__main__":
    durability_test()
