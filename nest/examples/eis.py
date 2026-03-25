import numpy as np
import matplotlib.pyplot as plt
from sksundae.ida import IDA
from time import process_time

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

    def resfn(t,y,yp,res):
        n = exampleCell.elements
        n_z = 5 # To be specified by the cell class

        # To be specified by the cell class
        C_fuel = 1.3 # F/m2
        C_air = 100 # F/m2

        # Boundary conditions
        V = conditions.V * (t >= 0)
        T = conditions.T

        Ps_fuel = conditions.Ps_fuel() # To be changed once we have the continuity equations
        Ps_air = conditions.Ps_air() # To be changed once we have the continuity equations
        
        n_species_fuel = len(Ps_fuel)
        n_species_air = len(Ps_air)
        
        A_sec = dY*dZ
        uf_left = sum(conditions.n_fuel)*(R*T/sum(Ps_fuel))/A_sec
        ua_left = sum(conditions.n_air)*(R*T/sum(Ps_air))/A_sec
        
        c_bc_fuel = Ps_fuel/(R*T)
        c_bc_air = Ps_air/(R*T)

        dz_f = exampleCell.electrode_fuel.delta/n_z
        epsilon_f = exampleCell.electrode_fuel.transport.epsilon

        for i in range(n):
            j_segment = y[i+2*n]
            
            # Bulk concentrations - fuel
            source_f = exampleCell.electrode_fuel.kinetic.mol_flux(j_segment)
            u_ele = R*T/sum(Ps_fuel)*sum(source_f)
            uf_right = uf_left + u_ele/dZ*dX

            for s in range(n_species_fuel):
                # Flux into porous layer
                index_c = (3+n_species_air+n_species_fuel)*n 
                index_c += s*n_z + i*n_z*n_species_fuel                     
                x_left = y[i+(3+s)*n] # Bulk concentration at the interface
                J_surf = -1*(y[index_c]-x_left) # High flux to ensure equilibrium at the interface

                # Continuity - Advection - Fuel
                index = i + (s+3)*n
                if i > 0 :
                    res[index] = yp[index]  + (uf_right*y[index]-uf_left*y[index-1])/dX  + J_surf/dZ 
                else:
                    res[index] = yp[index]  + (uf_right*y[index]-uf_left*c_bc_fuel[s])/dX  + J_surf/dZ + y[index]*u_ele/dZ
                Ps_fuel[s] = y[index]*R*T
            uf_left = uf_left + u_ele/dZ*dX

            for z in range(n_z-1):
                for s in range(n_species_fuel):
                    # Flux going outside the volume
                    index_J = (3+n_species_air+n_species_fuel)*n + n*n_z*n_species_fuel
                    index_J += z + s*(n_z-1) + i*(n_z-1)*n_species_fuel
                    J = y[index_J]

                    # Porous concentration
                    index_c = (3+n_species_air+n_species_fuel)*n
                    index_c += z + s*n_z + i*n_z*n_species_fuel
                    c_left = y[index_c]
                    c_right = y[index_c+1]

                    # Diffusion to get Js - simplified
                    D = 0.001
                    res[index_J] =  J + (c_right-c_left)/dz_f*D

            # Continuity
            for z in range(n_z):
                for s in range(n_species_fuel):
                    # Flux going outside the volume
                    index_J = (3+n_species_air+n_species_fuel)*n + n*n_z*n_species_fuel 
                    index_J += z + s*(n_z-1) + i*(n_z-1)*n_species_fuel

                    # Porous concentration
                    index_c = (3+n_species_air+n_species_fuel)*n 
                    index_c += z + s*n_z + i*n_z*n_species_fuel 
                    
                    if z == 0:
                        x_left = y[i+(3+s)*n] # Bulk concentration at the interface
                        J_left = -1*(y[index_c]-x_left) # High flux to ensure equilibrium at the interface
                        J_right = y[index_J]
                    elif z < n_z-1:
                        J_left = y[index_J-1]
                        J_right = y[index_J]
                    else:
                        J_left = y[index_J-1]
                        J_right = -source_f[s] # Surface reaction at the end of the porous layer

                    res[index_c] = yp[index_c]*epsilon_f + (J_right-J_left)/dz_f

            # Bulk concentrations - air
            source_a = exampleCell.electrode_air.kinetic.mol_flux(j_segment)
            u_ele = R*T/sum(Ps_air)*sum(source_a)
            ua_right = ua_left + u_ele/dZ*dX
                        
            Ps_air = np.zeros(n_species_air)
            for s in range(n_species_air):
                index = i + (s+3+n_species_fuel)*n
                if i > 0 :
                    res[index] = yp[index]  + (ua_right*y[index]-ua_left*y[index-1])/dX  - source_a[s]/dZ 
                else:
                    res[index] = yp[index]  + (ua_right*y[index]-ua_left*c_bc_air[s])/dX  - source_a[s]/dZ + y[index]*u_ele/dZ 
                Ps_air[s] = y[index]*R*T
            ua_left = ua_left + u_ele/dZ*dX
            
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
            eta_el = 0
            for layer in exampleCell.electrolyte:
                eta_el += layer.V(j_segment, T)
            res[i+2*n] = V - (exampleCell.V_nerst(T,Ps_fuel,Ps_air) - eta_f - eta_a - eta_el)


    ## Initial conditions
    n_elements = exampleCell.elements
    n_species_fuel = len(conditions.n_fuel)
    n_species_air = len(conditions.n_air)
    n_porous = 5
    n_variables = (3+n_species_fuel+n_species_air+n_porous*n_species_fuel+(n_porous-1)*n_species_fuel)*n_elements

    y0 = np.zeros(n_variables)
    yp0 = np.zeros(n_variables)

    for i in range(exampleCell.elements):
        y0[i+3*n_elements] = conditions.P/(R*conditions.T)*0.5
        y0[i+(3+1)*n_elements] = conditions.P/(R*conditions.T)*0.75
        y0[i+(3+2)*n_elements] = conditions.P/(R*conditions.T)*1

    index = (3+n_species_air+n_species_fuel)*n_elements
    for j in range(n_porous*n_species_fuel*n_elements):
        y0[index+j] = conditions.P/(R*conditions.T)*0.5

    voltage_alg = [i+2*exampleCell.elements for i in range(exampleCell.elements)]
    flux_alg = [i+(3+n_species_air+n_species_fuel+n_porous*n_species_fuel)*n_elements for i in range((n_porous-1)*n_species_fuel*n_elements)]

    solver = IDA(
        resfn,
        algebraic_idx=np.concatenate((voltage_alg, flux_alg)),
        calc_initcond='yp0'
    )

    # Solve from t=0 to t=0.1 s
    tspan = [0.0, 0.1]

    start_time = process_time()
    sol = solver.solve(tspan, y0, yp0)
    print(sol)
    print(f"Computation time : {process_time() - start_time} seconds")

    j_avg = np.mean(sol.y[:,2*exampleCell.elements:3*exampleCell.elements], axis=1)
    #plt.plot(sol.t,j_avg)
    #for i in range(n_porous*2):
    #    plt.plot(sol.t,sol.y[:,index+i]*R*conditions.T,label=f'c_{i}')
    for i in range(n_elements):
        plt.plot(sol.t,sol.y[:,i+3*n_elements]*R*conditions.T,label=f'c_{i}')
    plt.legend()
    #plt.plot(sol.t,sol.y[:,3*n_elements])
    return plt.show()
