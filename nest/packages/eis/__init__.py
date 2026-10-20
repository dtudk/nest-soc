"""Distributed electrochemical and gas-transport model."""

from .model import Model
from .solver import solve_steady, solve_transient
from .protocol import voltage_ramp, current_interrupt
from .impedance import impedance
from .postprocess import plot_impedance, plot_transient, save_results

__all__ = [
    "Model", 
    "solve_steady", "solve_transient", 
    "voltage_ramp", "current_interrupt",
    "impedance", "plot_impedance", "plot_transient", "save_results"
]
