import heapq
import math
import random

from search_methods.solver import Solver
from sokoban.map import Map
from sokoban.moves import *
from search_methods.heuritics import (
    sokoban_score_brute,
    sokoban_score_least,
    sokoban_player_seeded_target,
)

KEY_SCORE = 0
KEY_STATE = 1
KEY_MOVES = 2


def get_cycling_moves(moves: list[int]):
    moves_before = moves[1:]
    cycles = 0
    for m1, m2 in zip(moves_before, moves):
        if m1 == UP and m2 == DOWN:
            cycles += 1
        if m1 == DOWN and m2 == UP:
            cycles += 1
        if m1 == LEFT and m2 == RIGHT:
            cycles += 1
        if m1 == RIGHT and m2 == LEFT:
            cycles += 1
    return cycles


class BeamSearch(Solver):
    def __init__(self, map: Map, **kw_args) -> None:
        self.map = map
        self.K = kw_args.get("K", 10)
        self.stochastic = kw_args.get("stochastic", False)
        self.temp = kw_args.get("temp", 1)
        self.temp_scale_factor = kw_args.get("temp_scale_factor", 0.95)
        self.slalom_tolerance = kw_args.get("slalom_tolerance", 5)
        super().__init__(map, **kw_args)
        self.algo_name = f"BeamSearch_K_{self.K}"
        self.expanded_states = 0
        self.pulls = kw_args.get("pulls", False)

    def sample(self, state: Map, pull=False):
        valid_moves = state.filter_possible_moves()
        if pull:
            return valid_moves
        else:
            return list([move for move in valid_moves if move <= DOWN])

    def move_score(self, state: Map, old_state: Map):
        boxes_1 = set(state.boxes)
        boxes_2 = set(old_state.boxes)
        if boxes_1 != boxes_2:
            return 1.0  # incentive to push boxes
        return 0.0

    def score(self, state: Map, moves: list[int], old_state: Map = None):
        if old_state is not None:
            move_scr = self.move_score(state, old_state)
        else:
            move_scr = 0
        pure_score = sokoban_player_seeded_target(state)
        pull_moves = len(list([move for move in moves if move > DOWN]))
        moves_length = len(moves)
        return pure_score + move_scr + get_cycling_moves(moves) + pull_moves * 5

    def compute_probability(self, score: float, temp: float):
        return math.exp(-score / 1000.0 / temp)

    def choose_candidates(
        self, candidates: list[tuple[float, Map, list[int]]], temp: float, weighted=True
    ):
        if len(candidates) < self.K:
            return candidates
        if not weighted:
            return random.choices(candidates[: 2 * self.K], k=self.K)
        if temp < 0.01:
            return random.choices(candidates[: 2 * self.K], k=self.K)
        probs = [self.compute_probability(score, temp) for score, _, _ in candidates]
        return list(random.choices(candidates, weights=probs, k=self.K))

    def apply_copy(self, state: Map, move: int):
        new_state = state.copy()
        new_state.apply_move(move)
        return new_state

    def solve(self):
        test_name = self.map.test_name
        print(f"{self.algo_name=}")

        final, is_ok = self._solve()
        if is_ok:
            obj = final, is_ok
            self.logger.pickle(obj, file_name=f"{self.algo_name}/{test_name}_result-ok")
            return final, is_ok
        for i in range(40):
            final, is_ok = self._solve()
            if is_ok:
                obj = final, is_ok
                self.logger.pickle(obj, file_name=f"{self.algo_name}/{test_name}_result-ok")
                return final, is_ok
        obj = final, is_ok
        self.logger.pickle(obj, file_name=f"{self.algo_name}/{test_name}_result-notfinished")
        return final, is_ok

    def _solve(self):
        begin_state = self.map.copy()
        queue: list[tuple[float, Map, list[int]]] = []
        queue.append((self.score(begin_state, []), begin_state, []))
        temperature = self.temp
        iterations = 0
        best_candidate_score = float("inf")
        best_queue_score = float("inf")
        slalom_tolerance = self.slalom_tolerance
        seen = set()
        while True:
            iterations += 1
            with self.logger as log:
                log(iterations)
            if iterations > self.max_iter:
                best_queue_state = heapq.heappop(queue)
                return best_queue_state, best_queue_state[KEY_STATE].is_solved()

            is_improvement = False
            candidates = []
            for score, state, moves in queue:
                next_valid_moves = self.sample(state, pull=self.pulls)
                if len(next_valid_moves) == 0:
                    continue
                next_valid_states = [
                    (self.apply_copy(state, move), move) for move in next_valid_moves
                ]
                self.expanded_states += 1
                next_valid_states_scores = [
                    self.score(n_state, moves + [move], old_state=state)
                    for n_state, move in next_valid_states
                ]
                next_valid_state_queue = [
                    (n_score, n_state, moves + [n_move])
                    for (n_state, n_move), n_score in zip(
                        next_valid_states, next_valid_states_scores
                    )
                ]
                next_valid_state_queue = sorted(
                    next_valid_state_queue, key=lambda x: x[0]
                )  # sort and choose the K smallest states

                best_score = next_valid_state_queue[0][KEY_SCORE]
                if best_score < score:
                    is_improvement = True
                else:
                    if random.random() < temperature / (1 + math.fabs(best_score - score)):
                        is_improvement = True  # introduce some randomness

                for queue_element in next_valid_state_queue:
                    # if (queue_element[KEY_STATE].signature(), state.signature()) in seen:
                    #     continue
                    seen.add((queue_element[KEY_STATE].signature(), state.signature()))
                    heapq.heappush(candidates, queue_element)

            final_candidate_states = list(filter(lambda x: x[KEY_STATE].is_solved(), candidates))
            if len(final_candidate_states) > 0:
                # we have a final state
                return final_candidate_states[0], True

            with self.logger as log:
                log(f"{is_improvement=}\n{best_candidate_score=}\n{best_queue_score=}")
            if not is_improvement:  # none of the beams resulted in an improvement
                best_queue_state = heapq.heappop(queue)
                return best_queue_state, best_queue_state[KEY_STATE].is_solved()
            if len(candidates) == 0:
                best_queue_state = heapq.heappop(queue)
                return best_queue_state, best_queue_state[KEY_STATE].is_solved()

            chosen_states = self.choose_candidates(candidates, temp=temperature)
            chosen_states = sorted(chosen_states, key=lambda x: x[KEY_SCORE])
            best_candidate_score = chosen_states[0][KEY_SCORE]
            best_queue_score = queue[0][KEY_SCORE]
            if best_candidate_score == best_queue_score and best_candidate_score > 1.0:
                slalom_tolerance -= 1
                if slalom_tolerance == 0:
                    temperature /= self.temp_scale_factor
                    slalom_tolerance = self.slalom_tolerance
                    with self.logger as log:
                        log("slalom")
            temperature *= self.temp_scale_factor
            queue = []
            with self.logger as log:
                log(chosen_states[0][KEY_STATE])
            # sokoban_score_brute(chosen_states[0][KEY_STATE], verbose=True)
            ok_states = 0
            for _state_tuple in chosen_states:
                if _state_tuple[KEY_SCORE] < 1000:
                    ok_states += 1
                heapq.heappush(queue, _state_tuple)

            if ok_states == 0:
                best_queue_state = heapq.heappop(queue)
                return best_queue_state, best_queue_state[KEY_STATE].is_solved()
