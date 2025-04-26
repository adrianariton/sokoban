import heapq
import math
import random
import os
from search_methods.solver import Solver
from search_methods.utils import print_map
from sokoban.map import Map
from sokoban.moves import *

KEY_SCORE = 0
KEY_STATE = 1
KEY_MOVES = 2


import heapq
import math
import random

from search_methods.solver import Solver
from sokoban.map import Map
from sokoban.moves import *
from search_methods.heuritics import sokoban_player_seeded_target, sokoban_player_target

KEY_SCORE = 0
KEY_STATE = 1
KEY_MOVES = 2


class LRTAStar(Solver):
    def __init__(self, problem: Map, **kw_args):
        self.problem = problem
        self.result = {}
        self.H = {}
        self.s = None
        self.a = None
        super().__init__(problem, **kw_args)
        self.algo_name = "LRTAStar"
        self.expanded_states = 0

    def heuristic(self, state: Map):
        return sokoban_player_seeded_target(state, verbose=False)

    def actions(self, state: Map):
        return [x for x in state.filter_possible_moves() if x <= 4]

    def action_cost(self, s: Map, a, s_prime: Map):
        return 1

    def lrta_cost(self, s: Map, a, s_prime: Map = None):
        if s_prime is None:
            s_prime = s.copy()
            s_prime.apply_move(a)
            self.expanded_states += 1
        if s_prime not in self.H:
            self.H[s_prime] = self.heuristic(s_prime)
        return self.action_cost(s, a, s_prime) + self.H[s_prime]

    def lrta_agent(self, s_prime: Map):
        if s_prime.is_solved():
            return None  # STOP

        if s_prime not in self.H:
            self.H[s_prime] = self.heuristic(s_prime)

        if self.s is not None and self.a is not None:
            self.result[(self.s, self.a)] = s_prime

            # Update H[s] ← min over b ∈ ACTIONS(s) of LRTA-COST(s, b, result[s, b], H)
            min_cost = float("inf")
            for b in self.actions(self.s):
                sp = self.result.get((self.s, b))
                cost = self.lrta_cost(self.s, b, sp)
                if cost < min_cost:
                    min_cost = cost
            self.H[self.s] = min_cost

        # a ← argmin over b ∈ ACTIONS(s') of LRTA-COST(s', b, result[s', b], H)
        min_cost = float("inf")
        best_action = None
        for b in self.actions(s_prime):
            sp = self.result.get((s_prime, b))
            cost = self.lrta_cost(s_prime, b, sp)
            if cost < min_cost:
                min_cost = cost
                best_action = b
        self.s = s_prime
        self.a = best_action

        return best_action

    def solve(self):
        test_name = self.problem.test_name
        print(f"{test_name=}")
        m = self.problem.copy()
        moves = []
        for _ in range(4):
            m = self.problem.copy()
            moves = []

            for i in range(self.max_iter):
                with self.logger as log:
                    log(i)
                action = self.lrta_agent(m)
                if action is None:
                    obj = (self.heuristic(m), m, moves), True

                    self.logger.pickle(obj, file_name=f"{self.algo_name}/{test_name}_result-ok")
                    return (self.heuristic(m), m, moves), True  # Solved

                m = m.copy()
                m.apply_move(action)
                moves.append(action)
                with self.logger as log:
                    log("\n\n")
                    log(m)
                    m_print = print_map(m, heuristic=self.heuristic)
                    log(m_print)
                    log(f"!!!!!|SCORE={self.heuristic(m)}|\n\n")
                    log(f"{len(moves)=}")

        self.logger.pickle(obj, file_name=f"{test_name}_result-notfinished")
        return (self.heuristic(m), m, moves), False  # Failed
