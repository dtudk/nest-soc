import numpy as np
import matplotlib.pyplot as plt
from sksundae.ida import IDA
from time import process_time

from nest import properties, layers, cell
from nest.constants import R
# Defining the problem equation

def resfn(t, y, yp, res, exampleCell, conditions):
    # Numerical discretization parameters
    n = exampleCell.elements
    n_z = 5 # To be specified by the cell class
    dX = exampleCell.area**0.5/exampleCell.elements
    dY = exampleCell.area**0.5
    dZ = 0.0001 # m - To be specified by the cell class
    dz_f = exampleCell.electrode_fuel.delta/n_z
    dz_a = exampleCell.electrode_air.delta/n_z
    A_sec = dY*dZ

    # Properties to be specified by the cell class
    C_fuel = 1.3 # F/m2
    C_air = 100 # F/m2
    k_interface = 100 # mol/m2/s/Pa - To be specified by the cell class

    # Boundary conditions
    V = conditions.V * (t >= 0)
    T = conditions.T
    Ps_fuel = conditions.Ps_fuel().copy()
    P_fuel = np.sum(Ps_fuel)
    Ps_air = conditions.Ps_air().copy()
    P_air = np.sum(Ps_air)
    uf_left = np.sum(conditions.n_fuel)*(R*T/np.sum(Ps_fuel))/A_sec
    ua_left = np.sum(conditions.n_air)*(R*T/np.sum(Ps_air))/A_sec        
    c_bc_fuel = Ps_fuel/(R*T)
    c_bc_air = Ps_air/(R*T)

    # Misc
    n_species_fuel = len(Ps_fuel)
    n_species_air = len(Ps_air)
    epsilon_f = exampleCell.electrode_fuel.transport.epsilon
    epsilon_a = exampleCell.electrode_air.transport.epsilon

    for i in range(n):
        j_segment = y[i+2*n]
        
        # Continuity in the channel - Fuel
        source_f = exampleCell.electrode_fuel.kinetic.mol_flux(j_segment)
        u_ele = R*T/P_fuel*np.sum(source_f)
        uf_x = uf_left + u_ele/dZ*dX
        for s in range(n_species_fuel):
            # Concentration in the channel volume element
            index_x = (s+3)*n # offset
            index_x += i # update
            cf_x = y[index_x] # Bulk concentration at the interface
            
            # Flux into porous layer
            index_c = (3+n_species_air+n_species_fuel)*n # offset
            index_c += s*n_z + i*n_z*n_species_fuel # update
            J_surf = -k_interface*(y[index_c]-cf_x) # OBS: high flux to ensure equilibrium at the interface

            # Continuity - Advection - Fuel
            if i > 0 :
                cf_left = y[index_x-1]
            else:
                cf_left = c_bc_fuel[s]
            res[index_x] = yp[index_x]  + (uf_x*cf_x-uf_left*cf_left)/dX  + J_surf/dZ
            #Ps_fuel[s] = y[index_x]*R*T # Move this line to the end of the porous layer continuity
        uf_left += u_ele/dZ*dX # Update velocity for the next segment

        # To-do : Create a case exception for n_z = 1 to avoid this loop and directly set Ps_fuel at the end of the porous layer for the kinetic calculation
        # Diffusion in the porous layer - Fuel
        for z in range(n_z-1): # Forward difference finite elements
            index_J_start = (3+n_species_air+n_species_fuel)*n + n*n_z*n_species_fuel # offset
            index_J_start += z + i*(n_z-1)*n_species_fuel # update
            index_J_end = (3+n_species_air+n_species_fuel)*n + n*n_z*n_species_fuel # offset
            index_J_end += z + n_species_fuel*(n_z-1) + i*(n_z-1)*n_species_fuel # update
            Js = y[index_J_start:index_J_end:n_z-1]

            index_c_start = (3+n_species_air+n_species_fuel)*n # offset
            index_c_start += z + i*n_z*n_species_fuel # update
            index_c_end = (3+n_species_air+n_species_fuel)*n
            index_c_end += z + n_species_fuel*n_z + i*n_z*n_species_fuel # update
            cs = y[index_c_start:index_c_end:n_z]
            cs_bottom = y[index_c_start+1:index_c_end+1:n_z]
            
            # Algebraic equation
            rhs = exampleCell.electrode_fuel.transport.dc_dl(cs, Js, T, exampleCell.electrode_fuel.kinetic.gas)
            res[index_J_start:index_J_end:n_z-1] = (cs_bottom-cs)/dz_f + rhs

            # Defining Ps_fuel at the end of the porous layer for the kinetic calculation
            if z == n_z-2:
                Ps_fuel = cs_bottom*R*T

        # Continuity in the porous layer - Fuel
        for z in range(n_z):
            for s in range(n_species_fuel):
                # Flux going outside the volume
                index_J = (3+n_species_air+n_species_fuel)*n + n*n_z*n_species_fuel # offset 
                index_J += z + s*(n_z-1) + i*(n_z-1)*n_species_fuel # update

                # Porous concentration
                index_c = (3+n_species_air+n_species_fuel)*n # offset
                index_c += z + s*n_z + i*n_z*n_species_fuel # update
                c_y = y[index_c]
                    
                if z == 0:
                    c_top = y[i+(3+s)*n] # Bulk concentration at the interface
                    J_top = -k_interface*(c_y-c_top) # High flux to ensure equilibrium at the interface
                    J_y = y[index_J]
                elif z < n_z-1:
                    J_top = y[index_J-1]
                    J_y = y[index_J]
                else:
                    J_top = y[index_J-1]
                    J_y = -source_f[s] # Surface reaction at the end of the porous layer
                res[index_c] = yp[index_c]*epsilon_f + (J_y-J_top)/dz_f    

        # Continuity in the channel - Air
        source_a = exampleCell.electrode_air.kinetic.mol_flux(j_segment)
        u_ele = R*T/P_air*np.sum(source_a)
        ua_x = ua_left + u_ele/dZ*dX
        for s in range(n_species_air):
            index = (s+3+n_species_fuel)*n # offset
            index += i # update
            ca_x = y[index] # Bulk concentration at the interface
            if i > 0 :
                ca_left = y[index-1] 
            else:
                ca_left = c_bc_air[s] 
            res[index] = yp[index]  + (ua_x*ca_x-ua_left*ca_left)/dX  - source_a[s]/dZ 
            #Ps_air[s] = y[index]*R*T # Move this line to the end of the porous layer continuity
        ua_left += u_ele/dZ*dX
        
        # Diffusion in the porous layer - air
        for z in range(n_z-1): # Forward difference finite elements
            index_J_start = (3+n_species_air+n_species_fuel)*n + n*n_z*(n_species_fuel+n_species_air) + n*(n_z-1)*n_species_fuel  # offset
            index_J_start += z + i*(n_z-1)*n_species_air # update
            index_J_end = (3+n_species_air+n_species_fuel)*n + n*n_z*(n_species_fuel+n_species_air) + n*(n_z-1)*n_species_fuel  # offset
            index_J_end += z + n_species_air*(n_z-1) + i*(n_z-1)*n_species_air # update
            Js = y[index_J_start:index_J_end:n_z-1]

            index_c_start = (3+n_species_air+n_species_fuel)*n + n*n_z*n_species_fuel + n*(n_z-1)*n_species_fuel # offset
            index_c_start += z + i*n_z*n_species_air # update
            index_c_end = (3+n_species_air+n_species_fuel)*n + n*n_z*n_species_fuel + n*(n_z-1)*n_species_fuel # offset
            index_c_end += z + n_species_air*n_z + i*n_z*n_species_air # update
            cs = y[index_c_start:index_c_end:n_z]
            cs_bottom = y[index_c_start+1:index_c_end+1:n_z]
            
            # Algebraic equation
            rhs = exampleCell.electrode_air.transport.dc_dl(cs, Js, T, exampleCell.electrode_air.kinetic.gas)
            res[index_J_start:index_J_end:n_z-1] = (cs_bottom-cs)/dz_a + rhs

            # Defining Ps_fuel at the end of the porous layer for the kinetic calculation
            if z == n_z-2:
                Ps_air = cs_bottom*R*T

        # Continuity in the porous layer - air
        for z in range(n_z):
            for s in range(n_species_air):
                # Flux going outside the volume
                index_J = (3+n_species_air+n_species_fuel)*n + n*n_z*(n_species_fuel+n_species_air) + n*(n_z-1)*n_species_fuel  # offset
                index_J += z + s*(n_z-1) + i*(n_z-1)*n_species_air # update
                
                # Porous concentration
                index_c = (3+n_species_air+n_species_fuel)*n + n*n_z*n_species_fuel + n*(n_z-1)*n_species_fuel # offset
                index_c += z + s*n_z + i*n_z*n_species_air # update
                c_y = y[index_c]
                    
                if z == 0:
                    c_top = y[i+(3+s+n_species_fuel)*n] # Bulk concentration at the interface
                    J_top = -k_interface*(c_y-c_top) # High flux to ensure equilibrium at the interface
                    J_y = y[index_J]
                elif z < n_z-1:
                    J_top = y[index_J-1]
                    J_y = y[index_J]
                else:
                    J_top = y[index_J-1]
                    J_y = -source_a[s] # Surface reaction at the end of the porous layer
                res[index_c] = yp[index_c]*epsilon_a + (J_y-J_top)/dz_a   

        # Fuel electrode overpotential
        j0 = exampleCell.electrode_fuel.kinetic.j_0(T,Ps_fuel)
        eta_f = y[i]
        res[i] = C_fuel*yp[i] + exampleCell.electrode_fuel.kinetic.j(eta_f,T,j0) - j_segment

        # Air electrode overpotential
        j0 = exampleCell.electrode_air.kinetic.j_0(T,Ps_air)
        eta_a = y[i+n]
        res[i+n] = C_air*yp[i+n] + exampleCell.electrode_air.kinetic.j(eta_a,T,j0) - j_segment

        # Cell voltage
        eta_el = 0
        for layer in exampleCell.electrolyte:
            eta_el += layer.V(j_segment, T)
        res[i+2*n] = V - (exampleCell.V_nerst(T,Ps_fuel,Ps_air) - eta_f - eta_a - eta_el)

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
        transport=layers.StefanMaxwell(dp=6e-7, epsilon=0.3, tau=3),
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
        transport=layers.StefanMaxwell(dp=6e-7, epsilon=0.3, tau=2.8),
    )

    # Define cell
    exampleCell =  cell.Cell(16e-4, Ni_YSZ, (YSZ, YSZ_CGO, CGO), LSCF_CGO, elements=10)

    n_fuel = (24 / 1e3 / 3600) * (1e5 / 8.314510 / 273.15)  # mol/s
    n_air = 50 / 1e3 / 3600 * (1e5 / 8.314510 / 273.15)  # mol/s
    x_H2 = 0.5
    x_O2 = 1
    conditions = cell.BoundaryData(
        V=1.25,
        j=-1e4,
        n_fuel=np.array([n_fuel * x_H2, n_fuel * (1 - x_H2)]),
        n_air=np.array([n_air * x_O2]),
        T=858 + 273.15,
        P=1e5,
    )

    ## Initial conditions
    n_elements = exampleCell.elements
    n_species_fuel = len(conditions.n_fuel)
    n_species_air = len(conditions.n_air)
    n_porous = 5
    n_variables = (3+n_species_fuel+n_species_air+n_porous*(n_species_fuel+n_species_air)+(n_porous-1)*(n_species_fuel+n_species_air))*n_elements

    y0 = np.zeros(n_variables)
    yp0 = np.zeros(n_variables)

    for i in range(exampleCell.elements):
        y0[i+3*n_elements] = conditions.P/(R*conditions.T)*0.5
        y0[i+(3+1)*n_elements] = conditions.P/(R*conditions.T)*0.5
        y0[i+(3+2)*n_elements] = conditions.P/(R*conditions.T)*1

    index = (3+n_species_air+n_species_fuel)*n_elements
    index_f = index
    for j in range(n_porous*n_species_fuel*n_elements):
        y0[index+j] = conditions.P/(R*conditions.T)*0.5
    
    index = (3+n_species_air+n_species_fuel)*n_elements + n_elements*n_porous*n_species_fuel + n_elements*(n_porous-1)*n_species_fuel
    for j in range(n_porous*n_species_air*n_elements):
        y0[index+j] = conditions.P/(R*conditions.T)*1
    
    voltage_alg = [i+2*exampleCell.elements for i in range(exampleCell.elements)]
    flux_alg_f = [i+(3+n_species_air+n_species_fuel+n_porous*n_species_fuel)*n_elements for i in range((n_porous-1)*n_species_fuel*n_elements)]
    flux_alg_a = [i+(3+n_species_air+n_species_fuel+n_porous*(n_species_fuel+n_species_air)+(n_porous-1)*n_species_fuel)*n_elements for i in range((n_porous-1)*n_species_air*n_elements)]

    solver = IDA(
        lambda t, y, yp, res: resfn(t, y, yp, res, exampleCell, conditions),
        algebraic_idx=np.concatenate((voltage_alg, flux_alg_f, flux_alg_a)),
        calc_initcond='yp0'
    )

    # Solve from t=0 to t=0.1 s
    tspan = [0.0, 0.1]

    start_time = process_time()
    sol = solver.solve(tspan, y0, yp0)
    print(sol)
    print(f"Computation time : {process_time() - start_time} seconds")

    j_avg = np.mean(sol.y[:,2*exampleCell.elements:3*exampleCell.elements], axis=1)
    plt.plot(sol.t,j_avg, label='Transient - j')
    
    #for i in range(n_porous):
    #    plt.plot(sol.t,sol.y[:,index+i+(n_porous-1)*n_species_air*n_elements]*R*conditions.T,label=f'c_{i}')
    
    #for i in range(n_porous*2):
    #    plt.plot(sol.t,sol.y[:,index+i]*R*conditions.T,label=f'c_{i}')
    #for i in range(n_elements*2):
    #    plt.plot(sol.t,sol.y[:,i+3*n_elements]*R*conditions.T,label=f'c_{i}')
    #for i in range(n_elements):
    #    plt.plot(sol.t,sol.y[:,i+(3+2)*n_elements]*R*conditions.T,label=f'c_{i}')
    #plt.plot(sol.t,sol.y[:,3*n_elements])

    # Steady-state
    solutions = exampleCell.solve_for_voltage(conditions)
    current = np.mean(solutions[1])
    #P_H2_ss = [conditions.P*solutions[4][i]/(solutions[4][i]+solutions[5][i]) for i in range(n_elements)]
    #print(P_H2_ss)
    plt.plot(sol.t,[current for i in range(len(sol.t))], label='Steady-state - j')
    plt.legend()
    return plt.show()
