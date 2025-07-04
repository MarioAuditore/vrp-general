from typing import Dict, Tuple, Optional, List

from vrp_study.configs import ModelConfig
from vrp_study.routing_manager import RoutingManagerBuilder, InnerNode, Pdp, InnerCar


class PDRoutingManagerBuilder(RoutingManagerBuilder):

    def __init__(
            self,
            distance_matrix: Dict[Tuple[object, object], float],
            time_matrix: Dict[Tuple[object, object], float],
            model_config: Optional[ModelConfig] = None
    ):
        super().__init__(
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            model_config=model_config
        )
        self._start_node: Optional[InnerNode] = None
        self._common_end_node: Optional[InnerNode] = None

    def _create_inner_nodes(self):
        start_node = InnerNode(
            id=0,
            start_time=self._depo.start_time,
            end_time=self._depo.end_time,
            service_time=0,
            demand=0,
            is_transit=False,
            routing_node=self._depo
        )
        self._start_node = start_node
        self._common_end_node = start_node
        self._inner_nodes.append(start_node)

        for crg in self._cargos:
            nodes = [
                InnerNode(
                    id=len(self._inner_nodes) + i,
                    service_time=crg.nodes[i].service_time,
                    start_time=crg.nodes[i].start_time,
                    end_time=crg.nodes[i].end_time,
                    demand=crg.nodes[i].capacity,
                    is_transit=True,
                    routing_node=crg.nodes[i]
                )
                for i in range(2)
            ]
            self._inner_nodes += nodes
            for n in nodes:
                n.pdp_id = len(self._pdp)
            self._pdp.append(Pdp(
                id=len(self._pdp),
                nodes=nodes
            ))

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
