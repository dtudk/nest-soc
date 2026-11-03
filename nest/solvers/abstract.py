from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from nest import Problem

@dataclass
class NestResult:
    """
    Class to store the results of a NEST simulation.
    """
    time: float | np.ndarray
    voltage: float | np.ndarray
    current_density: float | np.ndarray

class Solver(ABC):

    @abstractmethod
    def solve(self, problem: Problem) -> NestResult:
        """
        Solve the given problem and return the results.

        Parameters
        ----------
        problem : Problem
            The problem to be solved.

        Returns
        -------
        NestResult
            The results of the simulation.
        """
