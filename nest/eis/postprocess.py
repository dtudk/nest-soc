"""Plot and save steady-state, transient, and impedance results.

Notes
-----
This code was AI-generated and has not been reviewed.

"""

from dataclasses import fields, is_dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nest.properties import BasicSpecies

__all__ = ["save_results", "plot_impedance", "plot_transient"]
_ROW_LIMIT = 1_048_576


def _voltage(model, state, protocol, t):
    """Get voltage for either voltage or current control.

    Returns
    -------
    voltage : float
        Cell voltage [V].

    """
    value = protocol(t)
    if protocol.control_mode == "current":
        return model.voltage_for_current(state, value)
    return value


def _species(model, side):
    """Get gas names in the model's species order.

    Returns
    -------
    names : list of str
        Species names, or ``species_<i>`` for unknown species.

    """
    gas = getattr(model.cell, f"electrode_{side}").kinetic.gas
    known = {v: k for k, v in vars(BasicSpecies).items() if not k.startswith("_")}
    return [known.get(s, f"species_{i}") for i, s in enumerate(gas.species)]


def _parameters(name, value):
    """Flatten simulation inputs, including lambda closure parameters."""
    if is_dataclass(value):
        items = ((f.name, getattr(value, f.name)) for f in fields(value))
    elif callable(value) and getattr(value, "__closure__", None):
        items = zip(
            value.__code__.co_freevars, (c.cell_contents for c in value.__closure__)
        )
    elif isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (list, tuple, np.ndarray)):
        items = enumerate(value)
    elif hasattr(value, "__dict__") and not callable(value):
        items = vars(value).items()
    else:
        yield name, str(value) if callable(value) else value
        return
    for key, item in items:
        yield from _parameters(f"{name}.{key}", item)


def save_results(
    model,
    filename,
    *,
    steady=None,
    solution=None,
    protocol=None,
    frequencies=None,
    Z=None,
    circuit=None,
    source_voltage=0.0,
):
    """Save simulation data to an Excel workbook.

    Parameters
    ----------
    model : Model
        Model containing fields, material properties, and geometry.
    filename : str or Path
        Output .xlsx path.
    steady : tuple, optional
        (steady solution or state vector, voltage [V]).
    solution : OdeResult, optional
        Transient solution.
    protocol : callable, optional
        Command created by voltage_ramp or current_interrupt; required with solution.
    frequencies, Z : array_like, optional
        Frequencies [Hz] and complex impedance [ohm m^2].
    circuit : ExternalCircuit, optional
        Circuit used in the simulation; saves L, C, and both resistances.
    source_voltage : float, optional
        Source voltage used in the simulation [V]. Default is 0.

    Returns
    -------
    Path
        Saved Excel workbook.

    Notes
    -----
    Exports all spatial fields and splits worksheets at Excel's row limit.
    Circuit solutions also export capacitor voltage and total current.

    """
    if solution is not None and protocol is None:
        raise ValueError("protocol is required when solution is given")
    if Z is not None and frequencies is None:
        raise ValueError("frequencies is required when Z is given")

    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell

    book = Workbook(write_only=True)
    sheets = {}

    def add(name, header, row):
        sheet, count, part = sheets.get(name, (None, _ROW_LIMIT, 0))
        if count == _ROW_LIMIT:
            part += 1
            sheet = book.create_sheet(name if part == 1 else f"{name}_{part}")
            sheet.freeze_panes = "A2"
            sheet.append(header)
            count = 1
        values = []
        for value in row:
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                value = str(value)
            if isinstance(value, str):
                cell = WriteOnlyCell(sheet, value=value)
                cell.data_type = "s"  # Avoid interpreting text as Excel formulas.
                value = cell
            values.append(value)
        sheet.append(values)
        sheets[name] = sheet, count + 1, part

    def snapshot(prefix, t, state, voltage):
        psi_f, psi_a, c_f, c_a, ch_f, ch_a, support = model.unpack_state(state)
        eta_f, eta_a = model.overpotentials(state)
        current = model.current_density(state, voltage)
        add(
            f"{prefix}_terminal",
            ["time_s", "voltage_V", "mean_current_A_m2"],
            [t, voltage, current.mean()],
        )
        for x, j in zip(model.grid.x, current):
            add(f"{prefix}_current", ["time_s", "x_m", "current_A_m2"], [t, x, j])

        for region, side, c, depths, psi, eta in (
            ("fuel", "fuel", c_f, model.grid.z_fuel, psi_f, eta_f),
            ("air", "air", c_a, model.grid.z_air, psi_a, eta_a),
            ("support", "fuel", support, model.grid.z_support, None, None),
            ("fuel_channel", "fuel", ch_f[:, None], [None], None, None),
            ("air_channel", "air", ch_a[:, None], [None], None, None),
        ):
            names = _species(model, side)
            header = ["time_s", "x_m", "z_m"]
            if psi is not None:
                header += ["psi_V", "eta_V"]
            header += [f"c_{s}_mol_m3" for s in names] + [f"x_{s}" for s in names]
            fractions = c / c.sum(axis=0)
            for iz, depth in enumerate(depths):
                for ix, x in enumerate(
                    model.grid.x_air if side == "air" else model.grid.x
                ):
                    row = [t, x, depth]
                    if psi is not None and eta is not None:
                        row += [psi[iz, ix], eta[iz, ix]]
                    row += [*c[:, iz, ix], *fractions[:, iz, ix]]
                    add(f"{prefix}_{region}", header, row)

    for name, value in (
        ("cell", model.cell),
        ("boundary", model.boundary),
        ("protocol", protocol),
        ("sigma_f_S_m", model.sigma_f),
        ("sigma_a_S_m", model.sigma_a),
        ("resistance_ohm_m2", model.resistance),
    ):
        for entry in _parameters(name, value):
            add("parameters", ["parameter_SI", "value"], entry)

    if circuit is not None:
        for name, value in (
            ("circuit.inductance_H", circuit.inductance),
            ("circuit.wiring_resistance_ohm", circuit.wiring_resistance),
            ("circuit.shunt_resistance_ohm", circuit.shunt_resistance),
            ("circuit.resistance_ohm", circuit.resistance),
            ("circuit.capacitance_F", circuit.capacitance),
            ("circuit.source_voltage_V", source_voltage),
        ):
            add("parameters", ["parameter_SI", "value"], [name, value])

    for region, faces, centers in (
        ("channel", model.grid.x_faces, model.grid.x),
        ("air_channel", model.grid.x_air_faces, model.grid.x_air),
        ("fuel", model.grid.z_fuel_faces, model.grid.z_fuel),
        ("air", model.grid.z_air_faces, model.grid.z_air),
        ("support", model.grid.z_support_faces, model.grid.z_support),
    ):
        for i, center in enumerate(centers):
            add(
                "grid",
                ["region", "index", "left_m", "center_m", "right_m"],
                [region, i, faces[i], center, faces[i + 1]],
            )

    if steady is not None:
        state, voltage = steady
        snapshot("steady", None, getattr(state, "x", state), voltage)
    if solution is not None and protocol is not None:
        for index, (t, state) in enumerate(zip(solution.t, solution.y.T)):
            voltage = (
                solution.voltage[index]
                if hasattr(solution, "voltage")
                else _voltage(model, state, protocol, t)
            )
            snapshot("transient", t, state, voltage)
            if hasattr(solution, "capacitor_voltage"):
                add(
                    "transient_circuit",
                    ["time_s", "current_A", "capacitor_voltage_V"],
                    [t, solution.current[index], solution.capacitor_voltage[index]],
                )
            add(
                "command",
                ["time_s", "control", "value"],
                [t, protocol.control_mode, protocol(t)],
            )
    if Z is not None and frequencies is not None:
        for f, z in zip(frequencies, Z):
            add(
                "impedance",
                [
                    "frequency_Hz",
                    "real_ohm_m2",
                    "imag_ohm_m2",
                    "minus_imag_ohm_m2",
                    "magnitude_ohm_m2",
                    "phase_deg",
                ],
                [f, z.real, z.imag, -z.imag, abs(z), np.angle(z, deg=True)],
            )

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    return path


def _save_plot(fig, filename):
    path = Path(filename).with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(path, dpi=600)
    plt.close(fig)
    return path


def plot_impedance(
    cases, filename, *, experiment=None, imaginary_ylim=None, real_lim=None
):
    """Save Nyquist and frequency plots.

    Parameters
    ----------
    cases : dict
        Labels mapped to (frequencies [Hz], complex impedance [ohm m^2]).
    filename : str or Path
        Output SVG path.
    experiment : str or Path, optional
        Data file containing frequency, Re(Z), and -Im(Z).
    imaginary_ylim : tuple, optional
        Display limits (minimum, maximum) for -Im(Z) [ohm m^2].
    real_lim : tuple, optional
        Display limits (minimum, maximum) for Re(Z) [ohm m^2].

    Returns
    -------
    Path
        Saved SVG file.

    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
    if experiment is not None:
        f, re, neg_im = np.loadtxt(experiment, usecols=(0, 1, 2), ndmin=2).T
        axes[0].plot(re, neg_im, "o", mfc="none", label="Experiment")
        axes[1].semilogx(f, re, "o", mfc="none", label="Experiment")
        axes[2].semilogx(f, neg_im, "o", mfc="none", label="Experiment")
    for label, (f, z) in cases.items():
        z = np.asarray(z)
        axes[0].plot(z.real, -z.imag, label=label)
        axes[1].semilogx(f, z.real, label=label)
        axes[2].semilogx(f, -z.imag, label=label)
    real = r"$Z_{\mathrm{real}}$ [$\Omega$ m$^2$]"
    imag = r"$-Z_{\mathrm{imag}}$ [$\Omega$ m$^2$]"
    for ax, xlabel, ylabel in zip(
        axes, (real, "Frequency [Hz]", "Frequency [Hz]"), (imag, real, imag)
    ):
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, alpha=0.2)
    if imaginary_ylim is not None:
        axes[0].set_ylim(imaginary_ylim)
        axes[2].set_ylim(imaginary_ylim)
    if real_lim is not None:
        axes[0].set_xlim(real_lim)
        axes[1].set_ylim(real_lim)
    axes[1].legend()
    return _save_plot(fig, filename)


def _trace(
    model, state, protocol, t, quantity, side, region, species, location, voltage=None
):
    if voltage is None:
        voltage = _voltage(model, state, protocol, t)
    if quantity == "voltage":
        return voltage
    if quantity == "current":
        values = model.current_density(state, voltage)
    elif quantity == "overpotential":
        fuel, air = model.overpotentials(state)
        values = {"fuel": fuel[-1], "air": air[0]}[side]
    else:
        fields = model.unpack_state(state)
        concentrations = {
            "interface": {"fuel": fields[2][:, -1], "air": fields[3][:, 0]},
            "channel": {"fuel": fields[4], "air": fields[5]},
        }[region][side]
        values = (concentrations / concentrations.sum(axis=0))[
            _species(model, side).index(species)
        ]
    if location == "mean":
        return values.mean()
    coordinates = model.grid.x_air if side == "air" else model.grid.x
    channel = getattr(model, f"{side}_channel")
    x = {
        "inlet": coordinates[0],
        "center": channel.length / 2,
        "outlet": coordinates[-1],
    }[location]
    return np.interp(x, coordinates, values)


def plot_transient(
    cases,
    filename,
    *,
    quantity="voltage",
    location="mean",
    side="fuel",
    region="interface",
    species="H2",
    log_time=True,
):
    """Plot the response to a transient excitation.

    Parameters
    ----------
    cases : dict
        Labels mapped to (model, solution, protocol).
        Use the voltage_ramp or current_interrupt command passed to solve_transient.
    filename : str or Path
        Output SVG path.
    quantity : {'voltage', 'current', 'overpotential', 'species'}, optional
        Quantity to plot.
    location : {'mean', 'inlet', 'center', 'outlet'}, optional
        Channel location.
    side : {'fuel', 'air'}, optional
        Electrode side.
    region : {'interface', 'channel'}, optional
        Gas concentration location.
    species : str, optional
        Species name for mole fraction plots.
    log_time : bool, optional
        Use a logarithmic time axis.

    Returns
    -------
    Path
        Saved SVG file.

    """
    ylabel = {
        "voltage": "Voltage [V]",
        "current": r"Current density [A/m$^2$]",
        "overpotential": f"{side.capitalize()} overpotential [V]",
        "species": f"{species} mole fraction [-]",
    }[quantity]
    fig, ax = plt.subplots(figsize=(5.6, 3.6), layout="constrained")
    for label, (model, solution, protocol) in cases.items():
        voltages = getattr(solution, "voltage", [None] * len(solution.t))
        values = [
            _trace(model, y, protocol, t, quantity, side, region, species, location, v)
            for t, y, v in zip(solution.t, solution.y.T, voltages)
        ]
        before = (
            values[0]
            if hasattr(solution, "voltage")
            else _trace(
                model,
                solution.y[:, 0],
                protocol,
                np.nextafter(solution.t[0], -np.inf),
                quantity,
                side,
                region,
                species,
                location,
            )
        )
        ax.plot(
            np.r_[solution.t[0], solution.t] + (1e-10 if log_time else 0),
            [before, *values],
            label=label,
        )
    if log_time:
        ax.set_xscale("log")
    ax.set(xlabel="Time [s]", ylabel=ylabel)
    ax.grid(True, alpha=0.2)
    ax.legend()
    return _save_plot(fig, filename)
