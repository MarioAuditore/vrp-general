import dataclasses
import math
from typing import List

import networkx as nx
import numpy as np
from loguru import logger as log
from tqdm.auto import tqdm

from vrp_study.configs import ModelConfig
from vrp_study.initial_solution_builder import InitialSolutionBuilder
from vrp_study.routing_manager import RoutingManager, InnerNode
from .pdptw_routing_manager_builder import PDRoutingManagerBuilder
from vrp_study.pdptw_model.routing_model import find_optimal_paths
from vrp_study.data_model import Cargo


# def solve_sub_cargos(cargos, routing_manager, init_sols=None):
#     sub_manager = routing_manager.create_sub_problem(cargos, routing_manager.tariffs())
#     sub_manager.max_time_minutes = 1
#     sub_manager.max_solution_number = 20
#     sols = find_optimal_paths(sub_manager, init_solution=init_sols)[0]
#     for s in sols:
#         if len(s) == 0:
#             continue
#         if s[0] == 0:
#             s = s[1:-1]
#         yield [routing_manager._node_to_inner_node[sub_manager.nodes()[index].routing_node].id for index in s]

def get_epsilon(g: nx.Graph, path: list[int]):
    pass


@dataclasses.dataclass
class SolutionBuilder(InitialSolutionBuilder):
    
    max_problem_size: int = 25
    inverse_weight : bool = False
    
    def get_initial_solution(self, routing_manager: RoutingManager) -> List[List[InnerNode]]:
        cg = nx.Graph()
        start2end: dict[int, list[InnerNode]] = {}
        for pd in routing_manager.get_pick_up_and_delivery_nodes():
            a: InnerNode = routing_manager.nodes()[pd[0]]
            b: InnerNode = routing_manager.nodes()[pd[1]]
            cg.add_node(a.id)
            start2end[a.id] = [a, b]

        for pd1 in routing_manager.get_pick_up_and_delivery_nodes():
            pd1: list[InnerNode] = [routing_manager.nodes()[i] for i in pd1]
            for pd2 in routing_manager.get_pick_up_and_delivery_nodes():
                pd2: list[InnerNode] = [routing_manager.nodes()[i] for i in pd2]
                a, b = pd1[0], pd1[1]
                c, d = pd2[0], pd2[1]
                if a.id == c.id:
                    continue
                l0 = routing_manager.get_distance(a, b) + routing_manager.get_distance(c, d) + 0.01
                l1 = min(
                    get_len([a, b, c, d], routing_manager),
                    get_len([c, d, a, b], routing_manager),

                    get_len([a, c, d, b], routing_manager),
                    get_len([a, c, b, d], routing_manager),

                    get_len([c, a, b, d], routing_manager),
                    get_len([c, a, d, b], routing_manager),
                )
                if l1 > 0 and math.isinf(l1):
                    continue
                cost = min((l1 - l0) / l0, 2)
                cost = np.exp(cost)
                # print(l)
                if cost < 1.75:
                    if (a.id, c.id) in cg.edges():
                        cg.edges()[a.id, c.id]['length'] = min(cost, cg.edges()[a.id, c.id]['length'])
                    else:
                        cg.add_edge(a.id, c.id, length=cost)

        print(len(cg.nodes()), len(cg.edges), nx.is_connected(cg))

        if nx.is_connected(cg):
            graphs = [cg]
        else:
            res = []
            for i, c in enumerate(nx.connected_components(cg)):
                res.append(cg.subgraph(c))
            graphs = res

        car2path = {}

        for cg in graphs:
            l, r = 0.1, 128
            iterations = 0
            cms = nx.community.louvain_communities(cg, weight='length', resolution=(l + r) / 2)
            max_len_cms = max(len(c) for c in cms)
            while max_len_cms > self.max_problem_size or max_len_cms < 20:
                if max_len_cms > self.max_problem_size:
                    l = (r + l) / 2
                else:
                    r = (l + r) / 2
                cms = nx.community.louvain_communities(cg, weight='length', resolution=(l + r) / 2)
                max_len_cms = max(len(c) for c in cms)
                log.info(f"{max_len_cms, (l + r) / 2}")
                iterations += 1
                if iterations == 10:
                    break

            for i, c in enumerate(tqdm(cms)):
                nodes = [ccc for cc in c for ccc in start2end[cc]]
                cars = [car for car in routing_manager.cars() if car.id not in car2path]

                part = routing_manager.sub_problem(
                    nodes,
                    cars,
                    ModelConfig(max_solution_number=50, max_execution_time_minutes=0.5))

                solution = find_optimal_paths(part)[0]
                for i, s in enumerate(solution):
                    if len(s) > 0:
                        car2path[part.cars()[i].id] = [part.nodes()[point] for point in s if
                                                       part.nodes()[point].id not in {part.cars()[i].start_node.id,
                                                                                      part.cars()[i].end_node.id}]
                log.info(solution)
                if i > 0 and i % 7 == 0:
                    cars = [car for car in routing_manager.cars() if car.id in car2path]
                    nodes = [p for path in car2path.values() for p in path]

                    part = routing_manager.sub_problem(
                        nodes,
                        cars,
                        ModelConfig(max_execution_time_minutes=0.5)
                    )

                    solution = []
                    for i, car in enumerate(routing_manager.cars()):
                        if car.id in car2path:
                            solution.append(car2path[car.id])
                        # else:
                        #     solution.append([])
                    solution = find_optimal_paths(part, init_solution=solution)[0]
                    for i, s in enumerate(solution):
                        if len(s) > 0:
                            car2path[part.cars()[i].id] = [part.nodes()[point] for point in s if
                                                           part.nodes()[point].id not in {part.cars()[i].start_node.id,
                                                                                          part.cars()[i].end_node.id}]
                        else:
                            del car2path[part.cars()[i].id]

        solution = []
        for i, car in enumerate(routing_manager.cars()):
            if car.id in car2path:
                solution.append(car2path[car.id])
            else:
                solution.append([])
            # solution.append(car2path[car.id])
        return solution

        #
        # initial_solutions = []
        # not_solved_cargos = []
        # all_cargos = []
        # for c in cms:
        #     cargos = []
        #     for i in c:
        #         cargos.append(Cargo(id=len(cargos), nodes=[i.routing_node for i in start2end[i]]))
        #
        #     if len(cargos) < 2:
        #         for c in cargos:
        #             not_solved_cargos.append(Cargo(id=len(not_solved_cargos), nodes=[n for n in c.nodes]))
        #         if len(not_solved_cargos) > 10:
        #             cargos = not_solved_cargos
        #             not_solved_cargos = []
        #         else:
        #             continue
        #
        #     # all_cargos.append(cargos)
        #     for s in solve_sub_cargos(cargos, routing_manager):
        #         initial_solutions.append(s)
        #
        # if len(not_solved_cargos) > 0:
        #     print('solve not resolved')
        #     for s in solve_sub_cargos(not_solved_cargos, routing_manager):
        #         initial_solutions.append(s)
        # print(len(set(x for a in initial_solutions for x in a)))
        # return initial_solutions
        # raise Exception


def get_len(nodes: list[InnerNode], routing_manager: RoutingManager) -> float:
    time = nodes[0].start_time + nodes[0].service_time
    total_length = 0
    prev = nodes[0]
    for node in nodes[1:]:
        time += routing_manager.get_time(prev, node)
        total_length += routing_manager.get_distance(prev, node)
        a, b = node.start_time, node.end_time
        if time > b:
            # log.info(f'time limit: {time}//{b}')
            return float('inf')
        time = max(time, a) + node.service_time
        prev = node
    return total_length
