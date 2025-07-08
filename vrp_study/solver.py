from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from loguru import logger as log
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.util.optional_boolean_pb2 import BOOL_FALSE, BOOL_TRUE

from .configs import ConstraintConfig
from .data_model import Route
# from ..initial_solution_builder import InitialSolutionBuilder
from .initial_solution_builder import InitialSolutionBuilder
from .routing_manager import RoutingManager


@dataclass
class VehicleRoutingSolver:

    def __init__(self,
                 routing_manager: RoutingManager,
                 search_parameters: pywrapcp.DefaultRoutingSearchParameters,
                 constraints: List[Callable],
                 solution_processing: Callable = None,
                 solution_callback: Callable = None,
                 initial_solution_builder: Optional[InitialSolutionBuilder] = None
                 ):
        """
        Солвер для решения задач VRP.
        :param routing_manager: менеджер машин, грузов и вершин, отвечающий за коммуникацию между логикой задачи и логикой солвера
        :param search_parameters:  Параметры поиска решения, по умолчанию берутся из pywrapcp.DefaultRoutingSearchParameters()
        :param constraints:
        :param init_solution: init_solution
        """

        self.routing_manager = routing_manager
        self.search_parameters = search_parameters

        if solution_processing is None:
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        self.custom_solution_processing = solution_processing  # get_solution

        self.solution_callback = solution_callback
        self.initial_solution_builder = initial_solution_builder
        self.constraints = constraints

    def process_solution(self, manager, routing, solution):
        """
        Восстановление маршрутов из найденного решения.

        :param manager: менеджер решения
        :param routing: модель
        :param solution: найденное решение
        :return: Score решения, словарь маршрутов Route
        """
        total = 0
        routes = {}

        for vehicle_id, car in enumerate(self.routing_manager.cars()):
            if not routing.IsVehicleUsed(solution, vehicle_id):
                continue
            index = routing.Start(vehicle_id)
            path = []

            route_cost = 0
            while not routing.IsEnd(index):
                path.append(manager.IndexToNode(index))

                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_cost += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            route_cost += routing.GetFixedCostOfVehicle(vehicle_id)
            path.append(manager.IndexToNode(index))

            if len(path) == 2 and path[0] == path[1] == 0:
                path = []
            total += route_cost
            routes[vehicle_id] = Route(
                id=vehicle_id, path=path, tariff=car.tariff)

        return total, routes

    def solve_routing(self, init_solution=None) -> Optional[tuple[float, list[list[int]], list[list[int]]]]:
        """
        Описание основной проблемы.
        :param routing_manager: менеджер
        :param init_solution: init_solution
        :param search_parameters:  Параметры поиска, по умолчанию берутся из get_optimal_model_params()
        :return:  Либо картеж (скор, список путей, где путь это лист индексов посещенных нод) если решение найдено,
        либо None если не найдено.
        """
        routing_manager = self.routing_manager
        search_parameters = self.search_parameters

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(routing_manager.nodes()),
            len(routing_manager.cars()),
            routing_manager.starts_ids(),
            routing_manager.ends_ids()
        )
        log.info("Начало создания модели")
        routing = pywrapcp.RoutingModel(manager)

        for constraint in self.constraints:
            constraint(routing_manager, routing, manager)

        conf = routing_manager.get_model_config()

        search_parameters.time_limit.seconds = int(
            60 * conf.max_execution_time_minutes)
        if conf.max_solution_number > 0:
            search_parameters.solution_limit = conf.max_solution_number

        search_parameters.log_search = False

        if self.solution_callback is not None:
            routing.AddAtSolutionCallback(self.solution_callback(routing))

        routing.CloseModelWithParameters(search_parameters)

        log.info('Начало решения')
        if init_solution is not None:
            sols = init_solution
            log.info(f'use initial_solution: {len(sols)}')
            assignment = routing.ReadAssignmentFromRoutes(sols, True)
            if not assignment:
                log.warning(f'Bad Initial Solutions: {sols}')
            solution = routing.SolveFromAssignmentWithParameters(
                assignment, search_parameters
            )
        else:
            solution = routing.SolveWithParameters(
                search_parameters
            )

        if solution:
            return routing, manager, solution
        else:
            log.warning("No solution found !")
            return None

    def solve(self,):
        '''
        Решение задачи VRP
        '''

        # Ищем начальное решение, если можем
        init_solution = None
        if self.initial_solution_builder is not None:
            init_solution = self.initial_solution_builder.get_initial_solution(
                self.routing_manager)
            log.info(f'use initial_solution: {len(init_solution)}')

        # Если нашли начальное решение, приведем его к индексам
        if init_solution is not None:
            id2index = {n.id: i for i, n in enumerate(
                self.routing_manager.nodes())}
            init_solution = [[id2index[n.id]
                              for n in s] for s in init_solution]

        log.info(f'problem size: {len(self.routing_manager.nodes())}')
        # Решаем задачу
        result = self.solve_routing(init_solution=init_solution)
        if result is None:
            return None
        else:
            routing, manager, solution = result

        log.info('Solution found')

        # Восстанавливаем маршруты
        if self.custom_solution_processing:
            score, solution, times = self.custom_solution_processing(
                routing_manager=self.routing_manager, manager=manager, routing=routing, solution=solution)
        else:
            return self.process_solution(manager=manager, routing=routing, solution=solution)

        log.info(f"best_score: {len([s for s in solution if len(s) > 0])}")
        log.info(f"best_score: {score / 100:.2f}")

        return solution, times
