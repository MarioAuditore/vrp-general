from typing import List
from abc import ABC, abstractmethod

from .routing_manager import RoutingManager


class InitialSolutionBuilder(ABC):
    @abstractmethod
    def get_initial_solution(self, routing_manager: RoutingManager) -> List[List[int]]:
        pass
