"""validate eis using an R + parallel-RC circuit."""

import numpy as np

from nest.eis import impedance


R0 = 1.1e-5   # Ohmic resistance
RP = 3.0e-5   # Polarization resistance
TAU = 0.03    # time constant

TIME = np.concatenate(([0.0], np.geomspace(1e-9, 30 * TAU, 10000)))
FREQUENCIES = np.concatenate(([0.0], np.geomspace(0.01, 1e6 , 100)))


def expected_impedance():
    """Analytical impedance of a series resistance and parallel RC."""
    return R0 + RP / (1 + 2j * np.pi * FREQUENCIES * TAU)


def test_current_interrupt_impedance():
    # the analytical voltage relaxation after interrupting the current.
    step = 0.01
    current_before = -step
    voltage_before = -(R0 + RP) * current_before

    current = np.zeros_like(TIME)
    voltage = voltage_before - step * (
        R0 + RP * (1 - np.exp(-TIME / TAU))
    )

    # Verify that current interrupt EIS matches the analytical impedance.
    z = impedance(
        TIME, voltage, current, FREQUENCIES,
        voltage_before=voltage_before,
        current_before=current_before,
    )
    np.testing.assert_allclose(z, expected_impedance(), rtol=2e-5, atol=1e-6)


def test_voltage_step_impedance():
    # analytical current relaxation after a voltage step.
    step = 0.01
    effective_tau = TAU * R0 / (R0 + RP)

    voltage = np.full_like(TIME, step)
    current = -step / (R0 + RP) * (
        1 + RP / R0 * np.exp(-TIME / effective_tau)
    )

    # Verify that voltage ramp EIS matches the same analytical impedance.
    z = impedance(
        TIME, voltage, current, FREQUENCIES,
        voltage_before=0,
        current_before=0,
    )
    np.testing.assert_allclose(z, expected_impedance(), rtol=2e-5, atol=1e-6)



