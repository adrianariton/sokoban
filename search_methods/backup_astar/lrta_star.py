import heapq
import math
import random

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
        self.result = {}  # result[(s, a)] = s'
        self.H = {}  # H[s] = heuristic estimate of cost to goal
        self.s = None  # previous state
        self.a = None  # previous action
        super().__init__(problem, **kw_args)

    def heuristic(self, state: Map):
        return sokoban_player_seeded_target(state)

    def actions(self, state: Map):
        return [x for x in state.filter_possible_moves() if x <= 4]

    def action_cost(self, s: Map, a, s_prime: Map):
        return 1  # generic action cost, can be changed if needed

    def lrta_cost(self, s: Map, a, s_prime: Map = None):
        if s_prime is None:
            s_prime = s.copy()
            s_prime.apply_move(a)
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
                cost = self.lrta_cost(self.s, b)
                if cost < min_cost:
                    min_cost = cost
            self.H[self.s] = min_cost

        # a ← argmin over b ∈ ACTIONS(s') of LRTA-COST(s', b, result[s', b], H)
        min_cost = float("inf")
        best_action = None
        for b in self.actions(s_prime):
            cost = self.lrta_cost(s_prime, b)
            if cost < min_cost:
                min_cost = cost
                best_action = b
        # print(f"{best_action=}")
        self.s = s_prime
        self.a = best_action
        return best_action

    def solve(self):
        m = self.problem.copy()
        moves = []

        for i in range(self.max_iter):
            print(i)
            action = self.lrta_agent(m)
            if action is None:
                return (0, m, moves), True  # Solved

            m = m.copy()
            m.apply_move(action)
            moves.append(action)
            print("\n\n")
            print(m)
            print_map(m, heuristic=self.heuristic)
            print(f"!!!!!|SCORE={self.heuristic(m)}|\n\n")
            print(f"{len(moves)=}")

        return (1, m, moves), False  # Failed
