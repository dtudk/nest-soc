import numpy as np
import matplotlib.pyplot as plt
from sksundae.ida import IDA

from nest import properties, layers, cell
from nest.constants import R
# Defining the problem equation

def eis():
    fuel_mix = properties.Mixture(
        (properties.BasicSpecies.H2, properties.BasicSpecies.H2O)
    )
    air_mix = properties.Mixture((properties.BasicSpecies.O2))
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
            nu=np.array([-0.5]),
            E_act=1.45 / 8.617333262e-5 * 8.314510,
            p=np.array([0.22]),
        ),
        transport=layers.BinaryFick(dp=6e-7, epsilon=0.3, tau=2.8),
    )

    # Define cell
    exampleCell =  cell.Cell(16e-4, Ni_YSZ, (YSZ, YSZ_CGO, CGO), LSCF_CGO)

    n_fuel = (24 / 1e3 / 3600) * (1e5 / 8.314510 / 273.15)  # mol/s
    n_air = 50 / 1e3 / 3600 * (1e5 / 8.314510 / 273.15)  # mol/s
    x_H2 = 0.5
    x_O2 = 1
    conditions = cell.BoundaryData(
        V=1.00,
        j=-1e4,
        n_fuel=np.array([n_fuel * x_H2, n_fuel * (1 - x_H2)]),
        n_air=np.array([n_air * x_O2]),
        T=858 + 273.15,
        P=1e5,
    )

    dX = exampleCell.area**0.5/exampleCell.elements
    dY = exampleCell.area**0.5
    dZ = 0.0001 # m

    dV = dX*dY*dZ

    def resfn(t,y,yp,res):
        n = exampleCell.elements
        n_y = 5 # To be specified by the cell class
        C_fuel = 1.3 # F/m2
        C_air = 100 # F/m2


        ###
        V = conditions.V * (t >= 0)
        T = conditions.T
        Ps_fuel = conditions.Ps_fuel() # To be changed once we have the continuity equations
        Ps_air = conditions.Ps_air() # To be changed once we have the continuity equations

        n_species_fuel = len(Ps_fuel)
        n_species_air = len(Ps_air)

        A_sec = dY*dZ
        uf_x = sum(conditions.n_fuel)*(R*T/sum(Ps_fuel))/A_sec
        ua_x = sum(conditions.n_air)*(R*T/sum(Ps_air))/A_sec

        for i in range(n):
            j_segment = y[i+2*n]
            
            # Bulk concentrations - fuel
            source_f = exampleCell.electrode_fuel.kinetic.mol_flux(j_segment)
            u_ele = R*T/sum(Ps_fuel)*sum(source_f)
            y_bc = Ps_fuel/(R*T)
            uf_dx = uf_x + u_ele/dZ*dX

            Ps_fuel = np.zeros(n_species_fuel)
            for s in range(n_species_fuel):
                index = i + (s+3)*n
                if i > 0 :
                    res[index] = yp[index]  + (uf_dx*y[index]-uf_x*y[index-1])/dX  - source_f[s]/dZ
                else:
                    res[index] = yp[index]  + (uf_dx*y[index]-uf_x*y_bc[s])/dX + y[index]*u_ele/dZ  - source_f[s]/dZ 
                Ps_fuel[s] = y[index]*R*T
            uf_x = uf_x + u_ele/dZ*dX

            # Bulk concentrations - air
            source_a = exampleCell.electrode_air.kinetic.mol_flux(j_segment)
            u_ele = R*T/sum(Ps_air)*sum(source_a)
            y_bc = Ps_air/(R*T)
            ua_dx = ua_x + u_ele/dZ*dX
                        
            Ps_air = np.zeros(n_species_air)
            for s in range(n_species_air):
                index = i + (s+3+n_species_fuel)*n
                if i > 0 :
                    res[index] = yp[index]  + (ua_dx*y[index]-ua_x*y[index-1])/dX  - source_a[s]/dZ 
                else:
                    res[index] = yp[index]  + (ua_dx*y[index]-ua_x*y_bc[s])/dX + y[index]*u_ele/dZ  - source_a[s]/dZ 
                Ps_air[s] = y[index]*R*T
            ua_x = ua_x + u_ele/dZ*dX

            # Notes to myself: Try to add the ODE equations for continuity on the porous media, and then add the Stefan-Maxwell equations as algebraic equations. 
            
            # Fuel electrode
            j0 = exampleCell.electrode_fuel.kinetic.j_0(T,Ps_fuel)
            eta_f = y[i]
            d_eta_f = yp[i]
            res[i] = C_fuel*d_eta_f + exampleCell.electrode_fuel.kinetic.j(eta_f,T,j0) - j_segment

            # Air electrode
            j0 = exampleCell.electrode_air.kinetic.j_0(T,Ps_air)
            eta_a = y[i+n]
            d_eta_a = yp[i+n]
            res[i+n] = C_air*d_eta_a + exampleCell.electrode_air.kinetic.j(eta_a,T,j0) - j_segment

            # Voltage
            V_el = 0
            for j, layer in enumerate(exampleCell.electrolyte):
                V_el += layer.V(j_segment, T)
            res[i+2*n] = V - (exampleCell.V_nerst(T,Ps_fuel,Ps_air) - eta_f - eta_a - V_el)


    ## Initial conditions
    n_elements = exampleCell.elements
    n_species_fuel = len(conditions.n_fuel)
    n_species_air = len(conditions.n_air)
    n_variables = (3+n_species_fuel+n_species_air)*n_elements

    y0 = np.zeros(n_variables)
    yp0 = np.zeros(n_variables)

    for i in range(exampleCell.elements):
        y0[i+3*n_elements] = conditions.P/(R*conditions.T)*0.5
        y0[i+(3+1)*n_elements] = conditions.P/(R*conditions.T)*0.5
        y0[i+(3+2)*n_elements] = conditions.P/(R*conditions.T)*1

    solver = IDA(
        resfn,
        algebraic_idx=[i+2*exampleCell.elements for i in range(exampleCell.elements)],
        calc_initcond='yp0'
    )

    # Solve from t=0 to t=0.1 s
    tspan = [0.0, 0.1]
    sol = solver.solve(tspan, y0, yp0)
    print(sol)

    j_avg = np.mean(sol.y[:,2*exampleCell.elements:3*exampleCell.elements], axis=1)
    plt.plot(sol.t,j_avg)
    #plt.plot(sol.t,sol.y[:,3*n_elements])
    return plt.show()
