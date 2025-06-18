import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Dict, Tuple, List, Optional

import numpy as np

from .data_model import TariffCost, Route
from .data_model import Tariff, Cargo, Node

logger = logging.getLogger(name='routing_model')


@dataclass
class InnerNode:
    id: int
    early_time: int
    late_time: int
    service_time: int
    demand: int
    is_transit: bool = field()
    routing_node: Optional[Node] = field(default=None)


@dataclass
class InnerCar:
    id: int = field(hash=True)  # идентификатор машины
    tariff: Tariff = field(hash=False)  # тариф мащины
    capacity: int = field(hash=False)  # вместимость по массе
    tariff_cost: TariffCost = field()  # цена и километраж тарифа

    end_node: InnerNode
    start_node: InnerNode
    use_when_empty: bool


@dataclass(
    init=False
)
class RoutingManager(ABC):
    distance_matrix: Dict[Tuple[object, object], float] = field(init=False)
    time_matrix: Dict[Tuple[object, object], float] = field(init=False)
    max_time_minutes: float = field(init=False, default=15)
    max_solution_number: int = field(init=False, default=-1)

    def __init__(self):
        self._nodes: List[Node] = []
        self._cargos: List[Cargo] = []
        self._tariffs: List[Tariff] = []

        self._pick_up_and_delivery: List[List[Node]] = []
        self._node_to_inner_node: Dict[Node, InnerNode] = {}

        self._np_dsts: Optional[np.ndarray] = None
        self._np_time: Optional[np.ndarray] = None

        self._inner_nodes: List[InnerNode] = []
        self._inner_cars: List[InnerCar] = []

    def cargos(self) -> List[Cargo]:
        return self._cargos

    def tariffs(self) -> List[Tariff]:
        return self._tariffs

    def add_tariff(self, tariff: Tariff):
        self._tariffs.append(tariff)

    def add_tariffs(self, tariffs: Iterable[Tariff]):
        for t in tariffs:
            self.add_tariff(t)

    def add_node(self, node: Node):
        self._nodes.append(node)

    def add_nodes(self, nodes: Iterable[Node]):
        for n in nodes:
            self.add_node(n)

    def add_cargo(self, cargo: Cargo):
        self._cargos.append(cargo)
        self._pick_up_and_delivery.append(cargo.nodes)
        for n in cargo.nodes:
            self.add_node(n)

    def add_cargos(self, cargos: Iterable[Cargo]):
        for c in cargos:
            self.add_cargo(c)

    def get_distance(self, index_a: int, index_b: int) -> float:
        return float(self._np_dsts[index_a, index_b])

    def get_distance_address(self, address_a: object, address_b: object) -> float:
        return self.distance_matrix[address_a, address_b]

    def get_time_address(self, address_a: object, address_b: object) -> float:
        return self.time_matrix[address_a, address_b]

    def get_time(self, index_a: int, index_b: int) -> float:
        return float(self._np_time[index_a, index_b])

    def build(self):
        self._validate()
        self._build()

    def get_pick_up_and_delivery_nodes(self) -> List[List[InnerNode]]:
        return [[self._node_to_inner_node[n] for n in ll] for ll in self._pick_up_and_delivery]

    def cars(self) -> List[InnerCar]:
        return self._inner_cars

    def nodes(self) -> List[InnerNode]:
        return self._inner_nodes

    def starts_ids(self):
        return [car.start_node.id for car in self._inner_cars]

    def ends_ids(self):
        return [car.end_node.id for car in self._inner_cars]

    def _validate(self):
        pass
        # starts = {crg.nodes[0].address for crg in self._cargos}
        # if len(starts) != 1:
        #     raise Exception()

    def _build(self):

        self._create_inner_nodes()

        self._node_to_inner_node = {n.routing_node: n for n in self._inner_nodes if n.routing_node is not None}

        self._create_inner_cars()

        self._sort_nodes()

        self._build_distance_matrix()

    @abstractmethod
    def _create_inner_nodes(self):
        ...

    @abstractmethod
    def _create_inner_cars(self):
        ...

    def _sort_nodes(self):
        starts = set(car.start_node.id for car in self._inner_cars)
        ends = set(car.end_node.id for car in self._inner_cars)

        def key(node: InnerNode):
            if node.id in starts:
                return -1
            if node.id in ends:
                return 1
            return 0

        self._inner_nodes.sort(key=key)
        for i, n in enumerate(self._inner_nodes):
            n.id = i

    def _build_distance_matrix(self):
        num_nodes = len(self._inner_nodes)
        dsts = np.zeros((num_nodes, num_nodes))
        time = np.zeros((num_nodes, num_nodes))

        for i, n1 in enumerate(self._inner_nodes):
            for j, n2 in enumerate(self._inner_nodes):
                if n1.routing_node is None or n2.routing_node is None or i == j:
                    continue
                if n1.routing_node.id == n2.routing_node.id:
                    continue
                dsts[i, j] = self.distance_matrix[n1.routing_node.id, n2.routing_node.id]
                time[i, j] = self.time_matrix[n1.routing_node.id, n2.routing_node.id]
        self._np_dsts = dsts
        self._np_time = time


class PDRoutingManager(RoutingManager):

    def __init__(self):
        super().__init__()
        self._start_node: Optional[InnerNode] = None
        self._common_end_node: Optional[InnerNode] = None
        self._depo: Optional[Node] = None

    def add_depo(self, depo: Node):
        self._depo = depo

    def get_depo_inner_node(self):
        return self._start_node

    def _create_inner_nodes(self):
        start_node = InnerNode(
            id=0,
            early_time=self._depo.start_time,
            late_time=self._depo.end_time,
            service_time=0,
            demand=0,
            is_transit=False,
            routing_node=self._depo
        )
        self._start_node = start_node
        self._common_end_node = start_node
        self._inner_nodes.append(start_node)

        for crg in self._cargos:
            self._inner_nodes += [
                InnerNode(
                    id=len(self._inner_nodes) + i,
                    service_time=crg.nodes[i].service_time,
                    early_time=crg.nodes[i].start_time,
                    late_time=crg.nodes[i].end_time,
                    demand=crg.nodes[i].capacity,
                    is_transit=True,
                    routing_node=crg.nodes[i]
                )
                for i in range(2)
            ]

    def _create_inner_cars(self):
        cars: List[InnerCar] = []
        for tariff in self._tariffs:
            count = tariff.max_count
            for i in range(count):
                for tc in tariff.cost_per_distance:
                    # дублирование автопарка, чтобы использовать все имеющиеся машины
                    cars.append(
                        InnerCar(
                            id=len(cars),
                            tariff=tariff,
                            capacity=tariff.capacity,
                            tariff_cost=tc,
                            start_node=self._start_node,
                            end_node=self._common_end_node,
                            use_when_empty=False
                        ))

        self._inner_cars = cars

    def create_sub_problem(self, cargos: List[Cargo], tariffs: List[Tariff]) -> RoutingManager:
        result = PDRoutingManager()

        result.distance_matrix = self.distance_matrix
        result.time_matrix = self.time_matrix

        result.add_cargos(cargos)
        result.add_tariffs(tariffs)

        result.add_depo(self._depo)

        result.build()

        return result
