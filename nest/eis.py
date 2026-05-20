import numpy as np
from sksundae.ida import IDA

from nest.constants import R, F, P_0


def exp_signal(t, base=0, step=-0.001, tau=1e-7, offset=1e-3):
    if t < offset:
        return base
    else:
        return base + step * (1 - np.exp(-(t - offset) / tau))


def step_signal(t, base=0, step=0.001, offset=0):
    if t < offset:
        return base
    else:
        return base + step


def dynamic_problem_IDA(
    t, y, yp, res, cellObject, boundaryObject, signal, k_interface=1000
):
    # Numerical discretization parameters
    n = cellObject.elements
    n_zf = cellObject.electrode_fuel.elements
    n_za = cellObject.electrode_air.elements

    dX = cellObject.area**0.5 / cellObject.elements
    dY = cellObject.area**0.5
    dZ = cellObject.delta_channel  # m
    A_YZ = dY * dZ

    dz_f = cellObject.electrode_fuel.delta / n_zf
    dz_a = cellObject.electrode_air.delta / n_za

    # Boundary conditions
    V = signal(t, base=boundaryObject.V)
    T = boundaryObject.T

    Ps_fuel = boundaryObject.Ps_fuel().copy()
    Ps_air = boundaryObject.Ps_air().copy()
    c_bc_fuel = Ps_fuel / (R * T)
    c_bc_air = Ps_air / (R * T)
    P_fuel = np.sum(Ps_fuel)
    P_air = np.sum(Ps_air)
    uf_left = np.sum(boundaryObject.n_fuel) * (R * T / P_fuel) / A_YZ
    ua_left = np.sum(boundaryObject.n_air) * (R * T / P_air) / A_YZ

    # Misc
    n_sf = len(Ps_fuel)
    n_sa = len(Ps_air)
    epsilon_f = cellObject.electrode_fuel.transport.epsilon
    epsilon_a = cellObject.electrode_air.transport.epsilon
    C_fuel = cellObject.electrode_fuel.kinetic.C  # F/m2
    C_air = cellObject.electrode_air.kinetic.C  # F/m2

    # Constant parameters for constant temperature
    D_eff_f = cellObject.electrode_fuel.transport.D_eff(
        T, P_fuel, cellObject.electrode_fuel.kinetic.gas
    )
    D_eff_a = cellObject.electrode_air.transport.D_eff(
        T, P_air, cellObject.electrode_air.kinetic.gas
    )
    V_nernst_T = cellObject.Vt_nernst(T)

    n_e = cellObject.electrode_fuel.kinetic.n_e
    nu_f = cellObject.electrode_fuel.kinetic.nu
    nu_a = cellObject.electrode_air.kinetic.nu
    k_mol_f = nu_f / (n_e * F)
    k_mol_a = nu_a / (n_e * F)
    R_ohm = sum(
        layer.delta / layer.conductivity.sigma(T) for layer in cellObject.electrolyte
    )
    j0T_f = (
        cellObject.electrode_fuel.kinetic.gamma
        * T**cellObject.electrode_fuel.kinetic.theta
        * np.exp(-cellObject.electrode_fuel.kinetic.E_act / (R * T))
    )
    j0T_a = (
        cellObject.electrode_air.kinetic.gamma
        * T**cellObject.electrode_air.kinetic.theta
        * np.exp(-cellObject.electrode_air.kinetic.E_act / (R * T))
    )
    p_f = cellObject.electrode_fuel.kinetic.p
    p_a = cellObject.electrode_air.kinetic.p

    for i in range(n):
        j_segment = (
            y[i + 2 * n] * 10000
        )  # Lowering the order of magnitude for numerical stability.

        # Continuity in the channel - Fuel
        source_f = k_mol_f * j_segment
        u_ele = R * T / P_fuel * np.sum(source_f)
        uf_x = uf_left + u_ele / dZ * dX
        for s in range(n_sf):
            # Concentration in the channel volume element
            index_x = (s + 3) * n  # offset
            index_x += i  # update
            cf_x = y[index_x]  # Bulk concentration at the interface

            # Flux into porous layer
            index_c = (3 + n_sa + n_sf) * n  # offset
            index_c += s * n_zf + i * n_zf * n_sf  # update
            J_surf = -k_interface * (
                y[index_c] - cf_x
            )  # OBS: high flux to ensure equilibrium at the interface

            # Continuity - Advection - Fuel
            if i > 0:
                cf_left = y[index_x - 1]
            else:
                cf_left = c_bc_fuel[s]
            res[index_x] = (
                yp[index_x] + (uf_x * cf_x - uf_left * cf_left) / dX + J_surf / dZ
            )
            # Ps_fuel[s] = y[index_x]*R*T # Move this line to the end of the porous layer continuity
        uf_left += u_ele / dZ * dX  # Update velocity for the next segment

        # To-do : Create a case exception for n_z = 1 to avoid this loop and directly set Ps_fuel at the end of the porous layer for the kinetic calculation
        # Diffusion in the porous layer - Fuel
        for z in range(n_zf - 1):  # Forward difference finite elements
            index_J_start = (3 + n_sa + n_sf) * n + n * n_zf * n_sf  # offset
            index_J_start += z + i * (n_zf - 1) * n_sf  # update
            index_J_end = (3 + n_sa + n_sf) * n + n * n_zf * n_sf  # offset
            index_J_end += z + n_sf * (n_zf - 1) + i * (n_zf - 1) * n_sf  # update
            Js = y[index_J_start : index_J_end : n_zf - 1]

            index_c_start = (3 + n_sa + n_sf) * n  # offset
            index_c_start += z + i * n_zf * n_sf  # update
            index_c_end = (3 + n_sa + n_sf) * n
            index_c_end += z + n_sf * n_zf + i * n_zf * n_sf  # update
            cs = y[index_c_start:index_c_end:n_zf]
            cs_bottom = y[index_c_start + 1 : index_c_end + 1 : n_zf]

            # Algebraic equation
            rhs = cellObject.electrode_fuel.transport.dc_dl(
                cs, Js, D_eff_f, T, cellObject.electrode_fuel.kinetic.gas
            )
            res[index_J_start : index_J_end : n_zf - 1] = (cs_bottom - cs) / dz_f + rhs

            # Defining Ps_fuel at the end of the porous layer for the kinetic calculation
            if z == n_zf - 2:
                Ps_fuel = cs_bottom * R * T

        # Continuity in the porous layer - Fuel
        for z in range(n_zf):
            for s in range(n_sf):
                # Flux going outside the volume
                index_J = (3 + n_sa + n_sf) * n + n * n_zf * n_sf  # offset
                index_J += z + s * (n_zf - 1) + i * (n_zf - 1) * n_sf  # update

                # Porous concentration
                index_c = (3 + n_sa + n_sf) * n  # offset
                index_c += z + s * n_zf + i * n_zf * n_sf  # update
                c_y = y[index_c]

                if z == 0:
                    c_top = y[i + (3 + s) * n]  # Bulk concentration at the interface
                    J_top = -k_interface * (
                        c_y - c_top
                    )  # High flux to ensure equilibrium at the interface
                    J_y = y[index_J]
                elif z < n_zf - 1:
                    J_top = y[index_J - 1]
                    J_y = y[index_J]
                else:
                    J_top = y[index_J - 1]
                    J_y = -source_f[
                        s
                    ]  # Surface reaction at the end of the porous layer
                res[index_c] = yp[index_c] * epsilon_f + (J_y - J_top) / dz_f

        # Continuity in the channel - Air
        source_a = k_mol_a * j_segment
        u_ele = R * T / P_air * np.sum(source_a)
        ua_x = ua_left + u_ele / dZ * dX
        for s in range(n_sa):
            index = (s + 3 + n_sf) * n  # offset
            index += i  # update
            ca_x = y[index]  # Bulk concentration at the interface
            if i > 0:
                ca_left = y[index - 1]
            else:
                ca_left = c_bc_air[s]
            res[index] = (
                yp[index] + (ua_x * ca_x - ua_left * ca_left) / dX - source_a[s] / dZ
            )
            # Ps_air[s] = y[index]*R*T # Move this line to the end of the porous layer continuity
        ua_left += u_ele / dZ * dX

        # Diffusion in the porous layer - air
        for z in range(n_za - 1):  # Forward difference finite elements
            index_J_start = (
                (3 + n_sa + n_sf) * n
                + n * n_zf * n_sf
                + n * n_za * n_sa
                + n * (n_zf - 1) * n_sf
            )  # offset
            index_J_start += z + i * (n_za - 1) * n_sa  # update
            index_J_end = (
                (3 + n_sa + n_sf) * n
                + n * n_zf * n_sf
                + n * n_za * n_sa
                + n * (n_zf - 1) * n_sf
            )  # offset
            index_J_end += z + n_sa * (n_za - 1) + i * (n_za - 1) * n_sa  # update
            Js = y[index_J_start : index_J_end : n_za - 1]

            index_c_start = (
                (3 + n_sa + n_sf) * n + n * n_zf * n_sf + n * (n_zf - 1) * n_sf
            )  # offset
            index_c_start += z + i * n_za * n_sa  # update
            index_c_end = (
                (3 + n_sa + n_sf) * n + n * n_zf * n_sf + n * (n_zf - 1) * n_sf
            )  # offset
            index_c_end += z + n_sa * n_za + i * n_za * n_sa  # update
            cs = y[index_c_start:index_c_end:n_za]
            cs_bottom = y[index_c_start + 1 : index_c_end + 1 : n_za]

            # Algebraic equation
            rhs = cellObject.electrode_air.transport.dc_dl(
                cs, Js, D_eff_a, T, cellObject.electrode_air.kinetic.gas
            )
            res[index_J_start : index_J_end : n_za - 1] = (cs_bottom - cs) / dz_a + rhs

            # Defining Ps_fuel at the end of the porous layer for the kinetic calculation
            if z == n_za - 2:
                Ps_air = cs_bottom * R * T

        # Continuity in the porous layer - air
        for z in range(n_za):
            for s in range(n_sa):
                # Flux going outside the volume
                index_J = (
                    (3 + n_sa + n_sf) * n
                    + n * n_zf * n_sf
                    + n * n_za * n_sa
                    + n * (n_zf - 1) * n_sf
                )  # offset
                index_J += z + s * (n_za - 1) + i * (n_za - 1) * n_sa  # update

                # Porous concentration
                index_c = (
                    (3 + n_sa + n_sf) * n + n * n_zf * n_sf + n * (n_zf - 1) * n_sf
                )  # offset
                index_c += z + s * n_za + i * n_za * n_sa  # update
                c_y = y[index_c]

                if z == 0:
                    c_top = y[
                        i + (3 + s + n_sf) * n
                    ]  # Bulk concentration at the interface
                    J_top = -k_interface * (
                        c_y - c_top
                    )  # High flux to ensure equilibrium at the interface
                    J_y = y[index_J]
                elif z < n_za - 1:
                    J_top = y[index_J - 1]
                    J_y = y[index_J]
                else:
                    J_top = y[index_J - 1]
                    J_y = -source_a[
                        s
                    ]  # Surface reaction at the end of the porous layer
                res[index_c] = yp[index_c] * epsilon_a + (J_y - J_top) / dz_a

        # Fuel electrode overpotential
        j0 = j0T_f * np.prod((Ps_fuel / P_0) ** p_f)
        eta_f = y[i]
        res[i] = (
            C_fuel * yp[i]
            + cellObject.electrode_fuel.kinetic.j(eta_f, T, j0)
            - j_segment
        )

        # Air electrode overpotential
        j0 = j0T_a * np.prod((Ps_air / P_0) ** p_a)
        eta_a = y[i + n]
        res[i + n] = (
            C_air * yp[i + n]
            + cellObject.electrode_air.kinetic.j(eta_a, T, j0)
            - j_segment
        )

        # Cell voltage
        eta_el = R_ohm * j_segment
        eta_conc = (
            R
            * T
            / (n_e * F)
            * np.log(np.prod((Ps_fuel / P_0) ** nu_f * (P_0 / Ps_air) ** nu_a))
        )
        res[i + 2 * n] = V - (V_nernst_T - eta_f - eta_a - eta_el - eta_conc)


def sparsity_pattern(cellObject, boundaryObject):
    n_x, n_zf, n_za, n_sf, n_sa = variable_indices(cellObject, boundaryObject)
    n_variables = (
        (3 + n_sf + n_sa + n_zf * n_sf + n_za * n_sa)
        + (n_zf - 1) * (n_sf)
        + (n_za - 1) * (n_sa)
    ) * n_x
    sparsity = np.ones((n_variables, n_variables))

    def set_zero_block(sparsity, i, j):
        sparsity[i, j] = 0
        sparsity[i + n_x, j] = 0
        sparsity[i + 2 * n_x, j] = 0
        for k in range(n_sf):
            sparsity[i + (3 + k) * n_x, j] = 0
        for k in range(n_sa):
            sparsity[i + (3 + n_sf + k) * n_x, j] = 0
        for z in range(n_zf):
            for s in range(n_sf):
                index_c_start = (3 + n_sa + n_sf) * n_x  # offset
                index_c_start += z + s * n_zf + i * n_zf * n_sf  # update
                sparsity[index_c_start, j] = 0
        for z in range(n_zf - 1):
            for s in range(n_sf):
                index_J = (3 + n_sa + n_sf) * n_x + n_x * n_zf * n_sf  # offset
                index_J += z + s * (n_zf - 1) + i * (n_zf - 1) * n_sf  # update
                sparsity[index_J, j] = 0
        for z in range(n_za):
            for s in range(n_sa):
                index_c = (
                    (3 + n_sa + n_sf) * n_x
                    + n_x * n_zf * n_sf
                    + n_x * (n_zf - 1) * n_sf
                )  # offset
                index_c += z + s * n_za + i * n_za * n_sa  # update
                sparsity[index_c, j] = 0
        for z in range(n_za - 1):
            for s in range(n_sa):
                index_J = (
                    (3 + n_sa + n_sf) * n_x
                    + n_x * n_zf * n_sf
                    + n_x * n_za * n_sa
                    + n_x * (n_zf - 1) * n_sf
                )  # offset
                index_J += z + s * (n_za - 1) + i * (n_za - 1) * n_sa  # update
                sparsity[index_J, j] = 0
        return sparsity

    for i in range(n_x):
        for j in range(n_x):
            if j > i:
                set_zero_block(sparsity, i, j)
        for j in range(n_x, 2 * n_x):
            if j > i + n_x:
                set_zero_block(sparsity, i, j)
        for j in range(2 * n_x, 3 * n_x):
            if j > i + 2 * n_x:
                set_zero_block(sparsity, i, j)
        for s in range(n_sf):
            for j in range(3 * n_x, (3 + s) * n_x):
                if j > i + (3 + s) * n_x:
                    set_zero_block(sparsity, i, j)
        for s in range(n_sa):
            for j in range((3 + n_sf) * n_x, (3 + n_sf + s) * n_x):
                if j > i + (3 + n_sf + s) * n_x:
                    set_zero_block(sparsity, i, j)
        for z in range(n_zf):
            for s in range(n_sf):
                index_c_start = (3 + n_sa + n_sf) * n_x  # offset
                for j in range(index_c_start, index_c_start + n_zf * n_sf):
                    if j > index_c_start + z + s * n_zf + n_zf * n_sf:
                        set_zero_block(sparsity, i, j)
        for z in range(n_zf - 1):
            for s in range(n_sf):
                index_J = (3 + n_sa + n_sf) * n_x + n_x * n_zf * n_sf  # offset
                for j in range(index_J, index_J + (n_zf - 1) * n_sf):
                    if j > index_J + z + s * (n_zf - 1) + (n_zf - 1) * n_sf:
                        set_zero_block(sparsity, i, j)
        for z in range(n_za):
            for s in range(n_sa):
                index_c = (
                    (3 + n_sa + n_sf) * n_x
                    + n_x * n_zf * n_sf
                    + n_x * (n_zf - 1) * n_sf
                )  # offset
                for j in range(index_c, index_c + n_za * n_sa):
                    if j > index_c + z + s * n_za + n_za * n_sa:
                        set_zero_block(sparsity, i, j)
        for z in range(n_za - 1):
            for s in range(n_sa):
                index_J = (
                    (3 + n_sa + n_sf) * n_x
                    + n_x * n_zf * n_sf
                    + n_x * n_za * n_sa
                    + n_x * (n_zf - 1) * n_sf
                )  # offset
                for j in range(index_J, index_J + (n_za - 1) * n_sa):
                    if j > index_J + z + s * (n_za - 1) + (n_za - 1) * n_sa:
                        set_zero_block(sparsity, i, j)
    return sparsity


def number_of_variables(cellObject, boundaryObject):
    n_x, n_zf, n_za, n_sf, n_sa = variable_indices(cellObject, boundaryObject)
    return (
        (3 + n_sf + n_sa + n_zf * n_sf + n_za * n_sa)
        + (n_zf - 1) * (n_sf)
        + (n_za - 1) * (n_sa)
    ) * n_x


def variable_indices(cellObject, boundaryObject):
    n_x = cellObject.elements
    n_sf = len(boundaryObject.n_fuel)
    n_sa = len(boundaryObject.n_air)
    n_zf = cellObject.electrode_fuel.elements
    n_za = cellObject.electrode_air.elements
    return n_x, n_zf, n_za, n_sf, n_sa


def algebraic_variable_indices(cellObject, boundaryObject):
    n_x, n_zf, n_za, n_sf, n_sa = variable_indices(cellObject, boundaryObject)

    current_var = np.array([i + 2 * n_x for i in range(n_x)])
    flux_var_f = np.array(
        [
            i + (3 + n_sa + n_sf + n_zf * n_sf) * n_x
            for i in range((n_zf - 1) * n_sf * n_x)
        ]
    )
    flux_var_a = np.array(
        [
            i + (3 + n_sa + n_sf + n_zf * n_sf + n_za * n_sa + (n_zf - 1) * n_sf) * n_x
            for i in range((n_za - 1) * n_sa * n_x)
        ]
    )
    algebraic_idx = np.concatenate((current_var, flux_var_f, flux_var_a))
    return algebraic_idx


def initial_conditions(cellObject, boundaryObject):
    ## Variable indexing
    n_x, n_zf, n_za, n_sf, n_sa = variable_indices(cellObject, boundaryObject)
    n_variables = (
        (3 + n_sf + n_sa + n_zf * n_sf + n_za * n_sa)
        + (n_zf - 1) * (n_sf)
        + (n_za - 1) * (n_sa)
    ) * n_x

    # Intial conditions
    y0 = np.zeros(n_variables)
    x_f = boundaryObject.n_fuel / np.sum(boundaryObject.n_fuel)
    x_a = boundaryObject.n_air / np.sum(boundaryObject.n_air)
    for x in range(n_x):
        for s in range(n_sf):
            y0[x + (3 + s) * n_x] = boundaryObject.P / (R * boundaryObject.T) * x_f[s]
        for s in range(n_sa):
            y0[x + (3 + n_sf + s) * n_x] = (
                boundaryObject.P / (R * boundaryObject.T) * x_a[s]
            )
    for x in range(n_x):
        for y in range(n_zf):
            for s in range(n_sf):
                index = (3 + n_sa + n_sf) * n_x  # offset
                index += y + s * n_zf + x * n_zf * n_sf  # update
                y0[index] = boundaryObject.P / (R * boundaryObject.T) * x_f[s]
    for x in range(n_x):
        for y in range(n_za):
            for s in range(n_sa):
                index = (
                    (3 + n_sa + n_sf) * n_x
                    + n_x * n_zf * n_sf
                    + n_x * (n_zf - 1) * n_sf
                )  # offset
                index += y + s * n_za + x * n_za * n_sa  # update
                y0[index] = boundaryObject.P / (R * boundaryObject.T) * x_a[s]
    return y0


def H_star(v, H, t):
    H = np.asarray(H, dtype=float)
    t = np.asarray(t, dtype=float)

    if H.shape != t.shape:
        raise ValueError("H and t must have the same shape")
    if H.size < 2:
        raise ValueError("H and t must have at least two elements")

    a = (H[1:] - H[:-1]) / (t[1:] - t[:-1])
    b = H[:-1] - a * t[:-1]

    w = 2.0 * np.pi * v
    iw = 1j * w

    terms = (-1j * a + w * b + w * a * t[1:]) * np.exp(-iw * t[1:]) - (
        -1j * a + w * b + w * a * t[:-1]
    ) * np.exp(-iw * t[:-1])

    return (1j / (w**2)) * np.sum(terms)


def eis(cellObject, boundaryObject):
    y0 = initial_conditions(cellObject, boundaryObject)
    yp0 = np.zeros(number_of_variables(cellObject, boundaryObject))

    # Problem definition and solver
    # OBS: The Jacobian is relatively dense and the indexes are not optimally arranged, so sparsity methods have no gains.
    solver = IDA(
        lambda t, y, yp, res: dynamic_problem_IDA(
            t, y, yp, res, cellObject, boundaryObject, exp_signal
        ),
        algebraic_idx=algebraic_variable_indices(cellObject, boundaryObject),
        calc_initcond="yp0",
        atol=1e-8,
        rtol=1e-6,
    )
    tspan = [0.0, 5]

    # Solution
    sol = solver.solve(tspan, y0, yp0)
    print(sol)

    # Ploting dynamic response in log scale
    j_avg = np.mean(sol.y[:, 2 * cellObject.elements : 3 * cellObject.elements], axis=1)
    offset = 1e-3
    for i, t in enumerate(sol.t):
        if t > offset:
            index_start = i
            break

    # EIS plot
    t = sol.t[index_start:]
    t = np.append(t, tspan[1] * 10)

    j = -j_avg[index_start:]  # A/cm2
    j = np.append(j, 0)

    v = np.array([exp_signal(x, base=boundaryObject.V) for x in sol.t])
    v = v[index_start:] - boundaryObject.V
    v = np.append(v, 0)

    fs = np.logspace(-3, 6, num=1000)
    impedance = [H_star(f, v, t) / H_star(f, j, t) for f in fs]

    return fs, impedance
