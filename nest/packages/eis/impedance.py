"""Impedance from voltage and mean-current-density transients."""

import numpy as np

__all__ = ["impedance"]



def impedance(time, voltage, current_density, frequencies,
              voltage_before, current_before):
    """Calculate impedance from transient voltage and current responses.

        Z(f) = -F{dV/dt} / F{dJ/dt}

    Parameters
    ----------
    time : array_like
        Sample times [s], a nondecreasing 1D array; spacing may vary.
    voltage : array_like
        Air-minus-fuel voltage [V], with the same shape as time.
    current_density : array_like
        Mean current density [A/m^2], with the same shape as time.
        Positive in fuel-cell operation.
    frequencies : float or array_like
        Frequencies [Hz], scalar or 1D array.
    voltage_before : float
        Steady voltage [V] immediately before the transient.
    current_before : float
        Steady mean current density [A/m^2] before the transient.

    Returns
    -------
    Z : ndarray
        Complex area-specific impedance [ohm m^2], one value per
        frequency. NaN where current excitation is negligible.

    Notes
    -----
    Signals are assumed constant after the record. Record through relaxation
    and resolve the fastest response. For a later jump, repeat its time with
    values immediately before and after the jump. Small perturbations give
    local impedance; a large interruption can include nonlinear effects.
    The minus sign follows from using fuel-cell-positive current.

    The Fourier transform of the time derivative of a signal X(t):

        F{dX/dt} = integral (dX/dt) exp(-2*pi*i*f*(t-t0)) dt

    The sampled signal is approximated by piecewise-linear segments.
    Within each interval [t_k, t_(k+1)], its derivative is constant:

        dX/dt = (X_(k+1) - X_k) / (t_(k+1) - t_k)

    The Fourier integral is then evaluated analytically over each
    segment:

        I_k(f) = dX_k * exp(-2*pi*i*f*(t_mid_k-t0))
                 * sinc(f*dt_k)

    where:

        dX_k    = X_(k+1) - X_k
        dt_k    = t_(k+1) - t_k
        t_mid_k = (t_k + t_(k+1)) / 2

    and sinc(x) = sin(pi*x)/(pi*x).

    The initial instantaneous change from the pre-transient steady
    state contributes X_0 - X_before. The complete transform is:

        F{dX/dt} = (X_0 - X_before) + sum_k I_k(f)

    """

    time = np.asarray(time, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current_density, dtype=float)
    dt = np.diff(time)
    midpoint = (time[:-1] + time[1:]) / 2 - time[0]
    dv, dj = np.diff(voltage), np.diff(current)
    current_scale = abs(current[0] - current_before) + np.sum(np.abs(dj))
    result = []
    for f in np.atleast_1d(frequencies):
        weight = np.exp(-2j * np.pi * f * midpoint) * np.sinc(f * dt)
        v_star = voltage[0] - voltage_before + np.sum(dv * weight)
        j_star = current[0] - current_before + np.sum(dj * weight)
        result.append(-v_star / j_star if abs(j_star) > 1e-12 * current_scale else complex(np.nan, np.nan))
    return np.asarray(result)
