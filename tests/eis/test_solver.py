"""Steady-state and transient solver tests."""

import numpy as np

from nest.eis import solve_steady, solve_transient, voltage_ramp


def test_steady(model):
    initial = model.initial_state()
    voltage = model.voltage_for_current(initial) + 0.01
    steady = solve_steady(model, voltage)

    # convergence test and negative electrolysis current
    assert steady.success
    assert steady.residual_norm < 1e-7
    assert model.current_density(steady.x, voltage).mean() < 0


def test_voltage_ramp(model):
    initial = model.initial_state()
    ocv = model.voltage_for_current(initial)

    times = np.concatenate(([0.0], np.geomspace(1e-8, 1e-3, 15)))

    # a 50 mV voltage ramp from open circuit
    protocol = voltage_ramp(ocv, ocv + 0.05, tau=1e-5)

    result = solve_transient(
        model,
        protocol,
        (0, times[-1]),
        initial_state=initial,
        t_eval=times,
    )

    # Verify successful integration and electrolysis current response.
    assert result.success

    initial_current = model.current_density(result.y[:, 0], protocol(times[0])).mean()

    final_current = model.current_density(result.y[:, -1], protocol(times[-1])).mean()

    assert final_current < initial_current
