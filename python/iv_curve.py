"""
Package usage example for generating a iV curve
"""
from time import process_time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nest import cell, layers, problem, ideal_gas

def iv_curve():
    """
    Example of using the package for generating a iV curve
    """
    # Define reactants
    fuel_mix = ideal_gas.Mixture((ideal_gas.BasicSpecies.H2,ideal_gas.BasicSpecies.H2O))
    air_mix = ideal_gas.Mixture((ideal_gas.BasicSpecies.O2))

    # Define cell layers
    Ni_YSZ = layers.Layer(
        delta = 3e-4+10e-6,
        kinetic=layers.ButlerVolmer(
            mix=fuel_mix,
            alpha=0.59,
            beta=1-0.59,
            gamma= 0.56*1.82527e6,
            theta=1,
            coeff=np.array([-1,1]),
            E_act=1.09/8.617333262E-5*8.314510,
            p=np.array([-0.1,0.33]),
            n_e = 2
        ),
        transport=layers.BinaryFick(
            dp=6e-7,
            epsilon=0.3,
            tau=3
        )
    )
    YSZ = layers.Layer(
        delta = 12e-6-2e-6,
        conductivity=layers.Conductivity(
            sigma0=3.6e7,
            E_act=8e4,
        )
    )
    YSZ_CGO = layers.Layer(
        delta = 4e-6,
        conductivity=layers.Conductivity(
            sigma0=1715,
            E_act=8785*8.314510,
            theta=0
        )
    )
    CGO = layers.Layer(
        delta = 10e-6-2e-6,
        conductivity=layers.Conductivity(
            sigma0=1.09e7,
            E_act=0.64/8.617333262E-5*8.314510,
        )
    )
    LSCF_CGO = layers.Layer(
        delta=3e-5,
        kinetic=layers.ButlerVolmer(
            mix=air_mix,
            alpha=0.65,
            beta=1-0.65,
            gamma=1.51556e8,
            coeff = np.array([-0.5]),
            E_act=1.45/8.617333262E-5*8.314510,
            p = np.array([0.22]),
        ),
        transport=layers.BinaryFick(
            dp = 6e-7,
            epsilon = 0.3,
            tau = 2.8
        )
    )

    # Define cell
    DTU_cell = cell.Cell(16E-4,Ni_YSZ,(YSZ,YSZ_CGO,CGO),LSCF_CGO)
    # Define boundary conditions
    n_fuel = (24/1E3/3600)*(1E5/8.314510/273.15)    # mol/s
    n_air = 50/1E3/3600*(1E5/8.314510/273.15)   # mol/s
    x_H2 = 0.5
    x_O2 = 1
    conditions = problem.BoundaryData(
        V=1.25,
        j=-1E4,
        n_fuel=np.array([n_fuel*x_H2,n_fuel*(1-x_H2)]),
        n_air=np.array([n_air*x_O2]),
        T=858+273.15,
        P=1E5)

    # Solve 1D problem for different voltages
    n = 10
    voltages = np.linspace(0.70,1.25,n)
    currents = np.zeros(n)
    start_time = process_time()
    for i,V in enumerate(voltages):
        conditions.V = V
        solutions = problem.solve_area(conditions,DTU_cell)
        currents[i] = sum(s[1] for s in solutions)/DTU_cell.elements
    print(f"Computation time : {process_time()-start_time} seconds")

    fig, ax = plt.subplots()
    ax.plot(currents/1E4,voltages,marker="o",markersize=5,label="Model",color="r")
    ax.set_xlabel("Current density (A/m^2)")
    ax.set_ylabel("Cell voltage (V)")

    # Add experimental data for comparison
    data = pd.read_csv("python/data/858C_50H2_100O2.csv")
    ax.scatter(data["j"],data["V"],label="Experiment",edgecolors="b",facecolor="none")
    plt.legend()
    return plt.show()

iv_curve()
