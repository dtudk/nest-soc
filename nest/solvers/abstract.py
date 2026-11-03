from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    import numpy as np
    from nest import Problem

class Mode(Enum):
    CURRENT = "current"
    VOLTAGE = "voltage"

    def __str__(self) -> str:
        return self.value

@dataclass
class NestResult(ABC):
    """
    Class to store the results of a NEST simulation.
    """
    time: float | np.ndarray
    voltage: float | np.ndarray
    current_density: float | np.ndarray

#     @abstractmethod
#     def __str__(self) -> str:
#         """
#         Return a string representation of the NestResult.
#
#         Returns
#         -------
#         str
#             A string representation of the NestResult.
#         """
#         pass

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
