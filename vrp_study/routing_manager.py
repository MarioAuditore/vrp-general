from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from loguru import logger as log

from .configs import ModelConfig
from .data_model import Cargo, Node, Route, Tariff, TariffCost


@dataclass
class InnerNode:
    id: int
    start_time: int
    end_time: int
    service_time: int
    demand: int
    is_transit: bool = field()
    pdp_id: int = field(default=-1)
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


@dataclass
class Pdp:
    id: int = field(hash=True)
    nodes: List[InnerNode] = field()


@dataclass
class RoutingManager:
    _model_config: ModelConfig = field()
    _dsts: np.ndarray
    _time: np.ndarray
    _inner_nodes: List[InnerNode]
    _inner_cars: List[InnerCar]
    _pick_up_and_delivery_nodes: List[Pdp]
    _depo_index: int

    def get_distance(self, node_a: InnerNode, node_b: InnerNode) -> float:
        return float(self._dsts[node_a.id, node_b.id])

    def get_time(self, node_a: InnerNode, node_b: InnerNode) -> float:
        return float(self._time[node_a.id, node_b.id])

    def starts_ids(self):
        return [car.start_node.id for car in self._inner_cars]

    def ends_ids(self):
        return [car.end_node.id for car in self._inner_cars]

    def cars(self) -> List[InnerCar]:
        return self._inner_cars

    def nodes(self) -> List[InnerNode]:
        return self._inner_nodes

    def get_pick_up_and_delivery_nodes(self) -> List[List[int]]:
        node_id2index = {n.id: i for i, n in enumerate(self._inner_nodes)}
        res = [[node_id2index[n.id] for n in pdp.nodes]
               for pdp in self._pick_up_and_delivery_nodes]
        return res

    def get_depo_index(self) -> int:
        return self._depo_index

    def get_model_config(self) -> ModelConfig:
        return self._model_config

    def sub_problem(self, nodes: List[InnerNode], cars: List[InnerCar], model_config: Optional[ModelConfig]):
        assert len(cars) > 0
        pdp_ids = {n.pdp_id for n in nodes}
        nodes_ids = {n.id for n in nodes}
        if self._depo_index not in nodes_ids:
            nodes = [self._inner_nodes[self._depo_index]] + nodes
        res = RoutingManager(
            _model_config=model_config if model_config is not None else self._model_config,
            _dsts=self._dsts,
            _time=self._time,
            _inner_nodes=nodes,
            _inner_cars=cars,
            _pick_up_and_delivery_nodes=[
                pdp for pdp in self._pick_up_and_delivery_nodes if pdp.id in pdp_ids],
            _depo_index=self._depo_index
        )
        return res


class RoutingManagerBuilder(ABC):
    def __init__(self,
                 distance_matrix: Dict[Tuple[object, object], float],
                 time_matrix: Dict[Tuple[object, object], float],
                 model_config: Optional[ModelConfig] = None
                 ):

        self.distance_matrix: Dict[Tuple[object,
                                         object], float] = distance_matrix
        self.time_matrix: Dict[Tuple[object, object], float] = time_matrix

        if model_config is None:
            self.model_config = ModelConfig()
        else:
            self.model_config = model_config
        self._nodes: List[Node] = []
        self._cargos: List[Cargo] = []
        self._tariffs: List[Tariff] = []

        self._np_dsts: Optional[np.ndarray] = None
        self._np_time: Optional[np.ndarray] = None

        self._inner_nodes: List[InnerNode] = []
        self._inner_cars: List[InnerCar] = []

        self._pdp: List[Pdp] = []

        self._depo: Optional[Node] = None

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
        for n in cargo.nodes:
            self.add_node(n)

    def add_cargos(self, cargos: Iterable[Cargo]):
        for c in cargos:
            self.add_cargo(c)

    def add_depo(self, depo: Node):
        self._depo = depo

    def build(self) -> RoutingManager:
        self._validate()
        return self._build()

    def _validate(self):
        # todo сделать валидатор
        # raise NotImplementedError
        ...

    def _build(self) -> RoutingManager:

        self._create_inner_nodes()

        self._node_to_inner_node = {
            n.routing_node: n for n in self._inner_nodes if n.routing_node is not None}

        self._create_inner_cars()

        self._sort_nodes()

        self._build_distance_matrix()

        return RoutingManager(
            _model_config=self.model_config,
            _dsts=self._np_dsts,
            _time=self._np_time,
            _inner_nodes=self._inner_nodes,
            _inner_cars=self._inner_cars,
            _pick_up_and_delivery_nodes=self._pdp,
            _depo_index=0
        )

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
                dsts[i, j] = self.distance_matrix[n1.routing_node.id,
                                                  n2.routing_node.id]
                time[i, j] = self.time_matrix[n1.routing_node.id, n2.routing_node.id]
        self._np_dsts = dsts
        self._np_time = time


def convert_solution(solution: List[List[InnerNode]]) -> List[Route[Node]]:
    # todo сделать конвертер
    raise NotImplementedError
