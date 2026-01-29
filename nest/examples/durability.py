from time import process_time
import numpy as np
import pandas as pd
from pathlib import Path

from nest import properties,layers,cell,degradation

import matplotlib.pyplot as plt

def durability_test():
    fuelMix = properties.Mixture((properties.BasicSpecies.H2,properties.BasicSpecies.H2O))
    airMix = properties.Mixture((properties.BasicSpecies.O2))
    Ni_YSZ = layers.Layer(
        delta = 1000e-6+10e-6,
        kinetic=layers.ButlerVolmer(
            fuelMix,
            nu=np.array([-1,1]),
            n_e = 2,
            alpha=0.59,
            beta=1-0.59,
            gamma= 1.82527e6/(6.79E12-1.19E12-1.41E12),
            p=np.array([-0.1,0.33]),
            theta=1,
            E_act=1.09/8.617333262E-5*8.314510
        ),
        transport=layers.BinaryFick(
            dp=600e-9,
            epsilon=0.3,
            tau=2,
        ),
        degradation = degradation.NickelAgglomeration(
               psi_ed = 0.4,
               psi_el = 0.6,
               r0_ed= 300E-9,
               r0_el= 350e-9,
               epsilon=0.3,
               alpha=3.98e3,
               r_ed_max=500e-9
        )
    )
    YSZ = layers.Layer(
        delta = 8e-6-1.5e-6,
        conductivity=layers.Conductivity(
            sigma0=3.6e7,
            E_act=8e4,
        )
    )
    YSZ_CGO = layers.Layer(
        delta = 3e-6,
        conductivity=layers.Conductivity(
            sigma0=1715,
            E_act=8785*8.314510,
            theta=0
        )
    )
    CGO = layers.Layer(
        delta = 5e-6-1.5e-6,
        conductivity=layers.Conductivity(
            sigma0=1.09e7,
            E_act=0.64/8.617333262E-5*8.314510,
        )
    )
    LSCF_CGO = layers.Layer(
        delta=32e-6,
        kinetic=layers.ButlerVolmer(
            airMix,
            alpha=0.65,
            beta=1-0.65,
            gamma=1.51556e8,
            nu = np.array([-0.5]),
            E_act=1.45/8.617333262E-5*8.314510,
            p = np.array([0.22]),
        ),
        transport=layers.BinaryFick(
            dp = 6e-7,
            epsilon = 0.3,
            tau = 2.8
        ),
        degradation=degradation.ChromiumPoison(
            x_H2O = 0.01,
            j0=4.27254e7,
        )
    )
    Crofer22 = layers.Layer(
        delta = 1,
        conductivity=layers.Conductivity(
            sigma0=1/(0.2e-5),
            theta=0
        )
    )
    CrScale = layers.Layer(
        delta = 0,
        conductivity=layers.Conductivity(
            sigma0=45.8,
            E_act=33.3e3,
            theta=0
        ),
        degradation=degradation.InterconnectOxidation(
            k_mg = 0.02361158401133492,
            E_ox = 260_000
        )
    )

    cellModel = cell.Cell(16e-4,Ni_YSZ,(YSZ,YSZ_CGO,CGO,Crofer22,CrScale),LSCF_CGO,elements=5)
        
    V_fuel = 8.23E-05
    V_air = 0.000345476 
    x_H2 = 0.94
    x_O2 = 0.21
    conditions = cell.BoundaryData(
        V=0.9,
        j=0.7E4,
        n_fuel=np.array([V_fuel*x_H2,V_fuel*(1-x_H2)]),
        n_air=np.array([V_air*x_O2,V_air*(1-x_O2)]),
        T=700+273.15,
        P=1E5)
    
    start_time = process_time()
    solution = cellModel.solve_for_time(conditions,"current",5000)
    print(f"Computation time : {process_time()-start_time} seconds")

    fig, ax = plt.subplots()

    conditions.V = 0.75
    y_matrix = np.transpose(solution.y)
    V = np.zeros(len(y_matrix))
    for i,y in enumerate(y_matrix):
        steady_solution = cellModel.solve_time_step(y,conditions,"current")
        V[i] = np.sum(steady_solution[0])/cellModel.elements
        conditions.V = V[i]

    ax.plot(solution.t,V,marker="o",markersize=5,label="Model",color="r")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Cell voltage (V)")

    # Add experimental data for comparison
    script_dir = Path(__file__).parent
    data = pd.read_csv(script_dir / "data/700C_75FU_40AIR.csv")
    ax.scatter(data["t"],data["V"],label="Experiment",edgecolors="b",facecolor="none")
    plt.legend()

    return plt.show()