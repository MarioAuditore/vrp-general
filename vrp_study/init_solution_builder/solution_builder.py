import dataclasses
from typing import List

import networkx as nx
import numpy as np

from scripts.initial_solution_builder import InitialSolutionBuilder
from scripts.routing_manager import RoutingManager, InnerNode, PDRoutingManager
from .routing_model import find_optimal_paths
from ..data_model import Cargo


def solve_sub_cargos(cargos, routing_manager, init_sols=None):
    sub_manager = routing_manager.create_sub_problem(cargos, routing_manager.tariffs())
    sub_manager.max_time_minutes = 1
    sub_manager.max_solution_number = 20
    sols = find_optimal_paths(sub_manager, init_solution=init_sols)[0]
    for s in sols:
        if len(s) == 0:
            continue
        if s[0] == 0:
            s = s[1:-1]
        yield [routing_manager._node_to_inner_node[sub_manager.nodes()[index].routing_node].id for index in s]


@dataclasses.dataclass
class SolutionBuilder(InitialSolutionBuilder):
    max_problem_size: int = 25

    def get_initial_solution(self, routing_manager: PDRoutingManager) -> List[List[int]]:
        cg = nx.Graph()
        start2end: dict[int, list[InnerNode]] = {}
        for pd in routing_manager.get_pick_up_and_delivery_nodes():
            a: InnerNode = pd[0]
            b: InnerNode = pd[1]
            cg.add_node(a.id)
            start2end[a.id] = [a, b]

        class Dsts:
            def __call__(self, *args: InnerNode):
                n1, n2 = args
                return routing_manager.get_distance(n1.id, n2.id)

        dsts = Dsts()
        depo = routing_manager.get_depo_inner_node()
        for pd1 in routing_manager.get_pick_up_and_delivery_nodes():
            for pd2 in routing_manager.get_pick_up_and_delivery_nodes():
                a, b = pd1[0], pd1[1]
                c, d = pd2[0], pd2[1]
                if a.id == c.id:
                    continue
                l0 = dsts(a, b) + dsts(c, d) + 0.001 + dsts(depo, a) + dsts(depo, c) + dsts(b, depo) + dsts(d, depo)
                l1 = min(
                    dsts(a, b) + dsts(b, c) + dsts(c, d) + dsts(depo, a) + dsts(d, depo),
                    dsts(c, d) + dsts(d, a) + dsts(a, b) + dsts(depo, c) + dsts(d, depo),

                    dsts(a, c) + dsts(c, d) + dsts(d, b) + dsts(depo, a) + dsts(b, depo),
                    dsts(a, c) + dsts(c, b) + dsts(b, d) + dsts(depo, a) + dsts(d, depo),

                    dsts(c, a) + dsts(a, b) + dsts(b, d) + dsts(depo, c) + dsts(d, depo),
                    dsts(c, a) + dsts(a, d) + dsts(d, b) + dsts(depo, c) + dsts(b, depo)
                )
                l = min((l1 - l0) / l0, 2)
                l = np.exp(l)
                # print(l)
                if l < 0.8:
                    if (a.id, c.id) in cg.edges():
                        cg.edges()[a.id, c.id]['length'] = min(l, cg.edges()[a.id, c.id]['length'])
                    else:
                        cg.add_edge(a.id, c.id, length=l)
        print(len(cg.nodes()), len(cg.edges))
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
            print(max_len_cms, (l + r) / 2)
            iterations += 1
            if iterations == 10:
                break

        initial_solutions = []
        not_solved_cargos = []
        all_cargos = []
        for c in cms:
            cargos = []
            for i in c:
                cargos.append(Cargo(id=len(cargos), nodes=[i.routing_node for i in start2end[i]]))

            if len(cargos) < 2:
                for c in cargos:
                    not_solved_cargos.append(Cargo(id=len(not_solved_cargos), nodes=[n for n in c.nodes]))
                if len(not_solved_cargos) > 10:
                    cargos = not_solved_cargos
                    not_solved_cargos = []
                else:
                    continue

            # all_cargos.append(cargos)
            for s in solve_sub_cargos(cargos, routing_manager):
                initial_solutions.append(s)

        if len(not_solved_cargos) > 0:
            print('solve not resolved')
            for s in solve_sub_cargos(not_solved_cargos, routing_manager):
                initial_solutions.append(s)
        print(len(set(x for a in initial_solutions for x in a)))
        return initial_solutions
