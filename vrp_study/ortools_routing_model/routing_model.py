import logging
from itertools import groupby
from threading import Lock
from typing import Optional

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.util.optional_boolean_pb2 import BOOL_FALSE, BOOL_TRUE

from ..initial_solution_builder import InitialSolutionBuilder

__all__ = [
    'find_optimal_paths'
]

from ..routing_manager import RoutingManager, InnerCar

logger = logging.getLogger(name='routing_model')

DEFAULT_MINUTES_FOR_MODEL = 1
"""
Верхняя граница расстояния для машины в метрах
"""
CAR_DISTANCE_UPPER_BOUND = int(1e6)


class SolutionCallback:

    def __init__(self, routing: pywrapcp.RoutingModel):
        self.model = routing
        self._best_objective = 1e10
        self.lock = Lock()

    def __call__(self):
        with self.lock:
            value = self.model.CostVar().Max()
            self._best_objective = min(self._best_objective, value)
            best = self._best_objective
        logger.info(f'find new solution: {value}, best solution: {best}')


def get_optimal_model_params() -> pywrapcp.DefaultRoutingSearchParameters:
    """
    Оптимальные параметры для модели.
    При изменении модели (например новые ограничения), эти параметры могут стать не самыми лучшими и потребуется доп
    тюнинг.
    Менять параметры стоит для эксперимента или при необходимости.
    :return: Оптимальные параметры для модели
    """
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.guided_local_search_lambda_coefficient = 0.0725

    search_parameters.local_search_operators.use_relocate = BOOL_FALSE
    search_parameters.local_search_operators.use_relocate_pair = BOOL_TRUE
    search_parameters.local_search_operators.use_light_relocate_pair = BOOL_TRUE

    search_parameters.local_search_operators.use_relocate_neighbors = BOOL_FALSE
    search_parameters.local_search_operators.use_relocate_subtrip = BOOL_TRUE

    search_parameters.local_search_operators.use_exchange = BOOL_FALSE
    search_parameters.local_search_operators.use_exchange_pair = BOOL_TRUE
    search_parameters.local_search_operators.use_exchange_subtrip = BOOL_TRUE

    search_parameters.local_search_operators.use_relocate_expensive_chain = BOOL_FALSE
    search_parameters.local_search_operators.use_two_opt = BOOL_FALSE
    search_parameters.local_search_operators.use_or_opt = BOOL_FALSE
    search_parameters.local_search_operators.use_lin_kernighan = BOOL_FALSE
    search_parameters.local_search_operators.use_tsp_opt = BOOL_FALSE

    search_parameters.local_search_operators.use_make_active = BOOL_FALSE
    search_parameters.local_search_operators.use_relocate_and_make_active = BOOL_FALSE
    search_parameters.local_search_operators.use_exchange_and_make_active = BOOL_FALSE
    search_parameters.local_search_operators.use_exchange_path_start_ends_and_make_active = BOOL_FALSE
    search_parameters.local_search_operators.use_make_inactive = BOOL_FALSE
    search_parameters.local_search_operators.use_make_chain_inactive = BOOL_FALSE
    search_parameters.local_search_operators.use_swap_active = BOOL_FALSE

    search_parameters.local_search_operators.use_extended_swap_active = BOOL_FALSE
    search_parameters.local_search_operators.use_shortest_path_swap_active = BOOL_FALSE
    search_parameters.local_search_operators.use_shortest_path_two_opt = BOOL_FALSE
    search_parameters.local_search_operators.use_node_pair_swap_active = BOOL_FALSE

    # LNS
    search_parameters.local_search_operators.use_path_lns = BOOL_FALSE
    search_parameters.local_search_operators.use_full_path_lns = BOOL_FALSE
    search_parameters.local_search_operators.use_tsp_lns = BOOL_FALSE
    search_parameters.local_search_operators.use_inactive_lns = BOOL_FALSE

    search_parameters.local_search_operators.use_global_cheapest_insertion_path_lns = BOOL_FALSE
    search_parameters.local_search_operators.use_local_cheapest_insertion_path_lns = BOOL_TRUE

    search_parameters.local_search_operators.use_relocate_path_global_cheapest_insertion_insert_unperformed = BOOL_FALSE

    # вроде как не учитывает pd
    search_parameters.local_search_operators.use_global_cheapest_insertion_expensive_chain_lns = BOOL_FALSE
    search_parameters.local_search_operators.use_local_cheapest_insertion_expensive_chain_lns = BOOL_FALSE

    # отменяет часть узлов (включая pick up and delivery) и заново вставляет используя указанные эвристики
    # global|local _cheapest_insertion
    search_parameters.local_search_operators.use_global_cheapest_insertion_close_nodes_lns = BOOL_FALSE
    search_parameters.local_search_operators.use_local_cheapest_insertion_close_nodes_lns = BOOL_TRUE
    search_parameters.heuristic_close_nodes_lns_num_nodes = 4

    search_parameters.use_cp_sat = True
    search_parameters.use_generalized_cp_sat = True
    search_parameters.sat_parameters.num_search_workers = 16
    return search_parameters


def get_solution(routing_manager: RoutingManager, manager, routing, solution) -> tuple[
    float, list[list[int]], list[list[int]]]:
    """
    Восстановление маршрутов из найденного решения.

    :param routing_manager: Данные модели
    :param manager: менеджер решения
    :param routing: модель
    :param solution: найденное решение
    :return: скор и лист путей (пути в номерах, исключая депо (точку 0), а не в исходных нодах)
    """
    total = 0
    result = []
    time_result = []
    time_dim = routing.GetDimensionOrDie('time')
    for vehicle_id, car in enumerate(routing_manager.cars()):
        if not routing.IsVehicleUsed(solution, vehicle_id):
            result.append([])
            time_result.append([])
            continue
        index = routing.Start(vehicle_id)
        path = []
        times = []
        route_cost = 0
        while not routing.IsEnd(index):
            path.append(manager.IndexToNode(index))
            time_var = time_dim.CumulVar(index)
            times.append((solution.Min(time_var), solution.Max(time_var)))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_cost += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        route_cost += routing.GetFixedCostOfVehicle(vehicle_id)
        path.append(manager.IndexToNode(index))
        time_var = time_dim.CumulVar(index)
        times.append((solution.Min(time_var), solution.Max(time_var)))

        if len(path) == 2 and path[0] == path[1] == 0:
            path = []
        total += route_cost
        result.append(path)
        time_result.append(times)
    return total, result, time_result


def add_mass_constraint(
        routing_manager: RoutingManager,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager
):
    """
    Добавление ограничение на вместимость по массе
    :param routing_manager: данные модели
    :param routing: солвер модели
    :param manager: менеджер модели
    :return: None
    """

    def demand_mass_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return routing_manager.nodes()[from_node].demand

    logger.info('Добавление ограничений для массы')

    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_mass_callback),
        0,  # null capacity slack
        [car.capacity for car in routing_manager.cars()],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity_mass"
    )


def add_time_window(
        routing_manager: RoutingManager,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager,
        time_dimension_name: str = 'time'
):
    logger.info('add time')

    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(routing_manager.get_time(from_node, to_node) + routing_manager.nodes()[from_node].service_time)

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    routing.AddDimension(
        transit_callback_index,
        max(node.late_time for node in routing_manager.nodes()),  # allow waiting time
        max(node.late_time for node in routing_manager.nodes()),  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time_dimension_name,
    )

    time_dimension = routing.GetDimensionOrDie(time_dimension_name)

    for node in routing_manager.nodes():

        from_time = node.early_time
        to_time = node.late_time
        if node.is_transit:
            index = manager.NodeToIndex(node.id)
            time_dimension.CumulVar(index).SetRange(from_time, to_time)
        else:
            for car in routing_manager.cars():
                if car.end_node.id == node.id:
                    index = routing.End(car.id)
                    time_dimension.CumulVar(index).SetRange(from_time, to_time)
                elif car.start_node.id == node.id:
                    index = routing.Start(car.id)
                    time_dimension.CumulVar(index).SetRange(from_time, to_time)


def add_vehicles_cost(routing_manager: RoutingManager,
                      routing: pywrapcp.RoutingModel,
                      manager: pywrapcp.RoutingIndexManager):
    """
    Добавление обработчиков на стоимость машины.
    :param routing_manager: Данные модели
    :param routing: Солвер модели
    :param manager: Менеджер модели
    """
    logger.info('Добавление стоимостей машин')

    def cost_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(routing_manager.get_distance(from_node, to_node) * 100)

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(cost_callback))


def add_pick_up_and_delivery(
        routing_manager: RoutingManager,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager,
        count_dimension_name: str = 'count'):
    """
    Добавление ограничения на порядок посещения.
    :param count_dimension_name: Название размерности для отслеживания порядка
    :param routing_manager: данные модели
    :param routing: солвер модели
    :param manager: менеджер модели
    """
    # Define Transportation Requests.
    logger.info('Добавление ограничения для порядка доставки')
    count_dimension = routing.GetDimensionOrDie(count_dimension_name)
    for request in routing_manager.get_pick_up_and_delivery_nodes():
        for i in range(len(request) - 1):
            pickup_index = manager.NodeToIndex(request[i].id)
            delivery_index = manager.NodeToIndex(request[i + 1].id)

            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            # поднять и сбросить груз должна одна машина.
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
            )
            # номер ноды c "поднятием" груза раньше ноды со "сбрасыванием" груза
            routing.solver().Add(
                count_dimension.CumulVar(pickup_index) <=
                count_dimension.CumulVar(delivery_index)
            )


def add_distance_dimension(routing_manager: RoutingManager,
                           routing: pywrapcp.RoutingModel,
                           manager: pywrapcp.RoutingIndexManager,
                           distance_dimension_name: str = 'distance'):
    """
    Добавление размерности для пройденного расстояния.
    :param routing_manager: менеджер
    :param distance_dimension_name: Имя размерности
    :param routing: Солвер модели
    :param manager: Менеджер модели
    :return: distance_dimension
    """
    logger.info('Добавление размерности для расстояния')

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(routing_manager.get_distance(from_node, to_node))

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        CAR_DISTANCE_UPPER_BOUND,  # vehicle maximum travel distance
        True,  # start cumul to zero
        distance_dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(distance_dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(0)

    return distance_dimension


def add_count_dimension(
        routing_manager: RoutingManager,
        routing: pywrapcp.RoutingModel,
        count_dimension_name: str = 'count'):
    """
    Добавление размерности для пройденного расстояния.
    :param routing_manager: менеджер
    :param count_dimension_name: Имя размерности для подсчета кол-ва посещенных точек
    :param routing: Солвер модели
    :return count_dimension
    """
    logger.info('Добавление размерности для расстояния')

    def count_callback(*args):
        return 1

    count_callback_index = routing.RegisterTransitCallback(count_callback)

    routing.AddDimension(
        count_callback_index,
        0,  # no slack
        1000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        count_dimension_name,
    )
    count_dimension = routing.GetDimensionOrDie(count_dimension_name)
    count_dimension.SetGlobalSpanCostCoefficient(0)
    return count_dimension


def do_solve(
        routing_manager: RoutingManager,
        *,
        initial_solution_builder: Optional[InitialSolutionBuilder] = None,
        time: float = DEFAULT_MINUTES_FOR_MODEL,
        solution_limit=None,
        search_parameters: Optional[pywrapcp.DefaultRoutingSearchParameters] = None,
        init_solution=None
) -> Optional[tuple[float, list[list[int]], list[list[int]]]]:
    """
        Описание основной проблемы.
    :param routing_manager: менежер
    :param initial_solution_builder
    :param time: Ограничение по времени, по умолчанию DEFAULT_MINUTES_FOR_MODEL.
    :param solution_limit: Ограничение на кол-во найденных решений.
    :param search_parameters:  Параметры поиска, по умолчанию берутся из get_optimal_model_params()
    :return:  Либо картеж (скор, список путей, где путь это лист индексов посещенных нод) если решение найдено,
     либо None если не найдено.
    """

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(routing_manager.nodes()),
        len(routing_manager.cars()),
        routing_manager.starts_ids(),
        routing_manager.ends_ids()
    )
    logger.info("Начало создания модели")
    routing = pywrapcp.RoutingModel(manager)

    add_distance_dimension(routing_manager, routing, manager)
    add_count_dimension(routing_manager, routing)

    add_pick_up_and_delivery(routing_manager, routing, manager)
    add_vehicles_cost(routing_manager, routing, manager)
    add_time_window(routing_manager, routing, manager)

    add_mass_constraint(routing_manager, routing, manager)

    if not search_parameters:
        search_parameters = get_optimal_model_params()
        search_parameters.time_limit.seconds = int(60 * time)
        if solution_limit is not None and solution_limit > 0:
            search_parameters.solution_limit = solution_limit

    search_parameters.log_search = False

    routing.AddAtSolutionCallback(SolutionCallback(routing))

    routing.CloseModelWithParameters(search_parameters)

    logger.info(f'Начало решения')
    if init_solution is not None:
        sols = init_solution
        logger.info(f'use initial_solution: {len(sols)}')
        assignment = routing.ReadAssignmentFromRoutes(sols, True)
        if not assignment:
            logging.warning(f'Bad Initial Solutions: {sols}')
        solution = routing.SolveFromAssignmentWithParameters(
            assignment, search_parameters
        )

    elif initial_solution_builder is not None:
        sols = initial_solution_builder.get_initial_solution(routing_manager)
        logger.info(f'use initial_solution: {len(sols)}')
        assignment = routing.ReadAssignmentFromRoutes(sols, True)
        if not assignment:
            logging.warning(f'Bad Initial Solutions: {sols}')
        solution = routing.SolveFromAssignmentWithParameters(
            assignment, search_parameters
        )

    else:
        solution = routing.SolveWithParameters(
            search_parameters
        )

    if solution:
        logger.info(f'find solution')
        return get_solution(routing_manager, manager, routing, solution)
    else:
        logger.warning("No solution found !")
        return None


def find_optimal_paths(
        routing_manager: RoutingManager,
        initial_solution_builder: Optional[InitialSolutionBuilder] = None,
        init_solution=None
):
    logger.info(f'problem size: {len(routing_manager.nodes())}')
    score, solution, times = do_solve(
        routing_manager,
        initial_solution_builder=initial_solution_builder,
        time=routing_manager.max_time_minutes,
        solution_limit=routing_manager.max_solution_number,
        init_solution=init_solution
    )
    logger.info(f"best_score: {len([s for s in solution if len(s) > 0])}")

    logger.info(f"best_score: {score / 100:.2f}")

    return solution, times
