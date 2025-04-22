from sokoban.map import Map, OBSTACLE_SYMBOL, BOX_SYMBOL, TARGET_SYMBOL
from copy import deepcopy
import heapq
from collections import deque
from sokoban.map import Map, OBSTACLE_SYMBOL
from functools import lru_cache
from itertools import permutations


def empty_like(map: list[list[int]]):
    map2 = deepcopy(map)
    for i in range(len(map2)):
        map2[i] = [0] * len(map2[i])
    return map2


def is_inbound(sokoban_map: Map, position):
    px, py = position
    return 0 <= px < sokoban_map.length and 0 <= py < sokoban_map.width


def is_obstacle(sokoban_map: Map, position):
    px, py = position
    return sokoban_map.map[px][py] == OBSTACLE_SYMBOL


def reachable_positions(sokoban_map: Map, start, forbidden):
    """
    Returns (visited, dists) where
    visited is the set of all cells reachable from `start`
    dists[(x,y)] is the min number of steps from start to (x,y)

    `forbidden` is a set of cells you should not walk into,
    and OBSTACLE_SYMBOL cells are also off‑limits.
    """
    visited = {start}
    dists = {start: 0}
    queue = deque([start])

    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if not is_inbound(sokoban_map, (nx, ny)):
                continue
            if (nx, ny) in visited or (nx, ny) in forbidden:
                continue
            if sokoban_map.map[nx][ny] == OBSTACLE_SYMBOL:
                continue

            visited.add((nx, ny))
            dists[(nx, ny)] = dists[(x, y)] + 1
            queue.append((nx, ny))

    return visited, dists


def box_to_target_bfs(
    sokoban_map: Map, box_pos, player_pos, target, blocked, box_block=False, update_reachable=True
):
    if box_block:
        reachable, player_dists = reachable_positions(sokoban_map, player_pos, blocked | {box_pos})
    else:
        reachable, player_dists = reachable_positions(sokoban_map, player_pos, blocked)

    heap = [(0.0, box_pos)]
    seen = set()
    best = {box_pos: 0.0}

    if box_pos == target:
        return 0.0

    while heap:
        cost, (bx, by) = heapq.heappop(heap)

        if (bx, by) == target:
            return cost

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = bx + dx, by + dy  # new box pos
            px, py = bx - dx, by - dy  # where player must stand
            if (bx, by, dx, dy) in seen:
                continue
            seen.add((bx, by, dx, dy))
            if not is_inbound(sokoban_map, (nx, ny)) or not is_inbound(sokoban_map, (px, py)):
                continue
            if is_obstacle(sokoban_map, (nx, ny)) or is_obstacle(sokoban_map, (px, py)):
                continue
            if (nx, ny) in blocked:
                continue
            if update_reachable:
                reachable, player_dists = reachable_positions(
                    sokoban_map, player_pos, blocked | {(nx, ny)}
                )

            if (px, py) not in reachable:
                continue
            if (px, py) in blocked:
                continue
            push_cost = player_dists.get((px, py), float("inf"))
            new_cost = cost + push_cost

            if new_cost < best.get((nx, ny), float("inf")):
                best[(nx, ny)] = new_cost
                heapq.heappush(heap, (new_cost, (nx, ny)))

    # unreachable → very bad
    return 1000.0


def bfs_distance(sokoban_map: Map, start, targets, blocked_symbols):
    visited = set()
    queue = deque([(start, 0)])
    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) in targets:
            return dist
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < sokoban_map.length
                and 0 <= ny < sokoban_map.width
                and (nx, ny) not in visited
                and sokoban_map.map[nx][ny] not in blocked_symbols
            ):
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
    return 1000.0  # Large penalty if unreachable


@lru_cache
def sokoban_score_brute(sokoban_map: Map) -> int:
    player_pos = (sokoban_map.player.x, sokoban_map.player.y)
    boxes = list(sokoban_map.boxes.values())
    targets = sokoban_map.targets
    obstacles = sokoban_map.obstacles

    if sokoban_map.is_solved():
        return 0.0

    box_positions = []
    fixed_box_positions = []
    for box in boxes:
        if not (box.x, box.y) in targets:
            box_positions.append((box.x, box.y))
        else:
            fixed_box_positions.append((box.x, box.y))
    all_box_positions = []
    for box in boxes:
        all_box_positions.append((box.x, box.y))

    def solve_sokoban(box_ind_to_target):
        scores = []
        for i, j in enumerate(box_ind_to_target):
            other_boxes = [all_box_positions[k] for k in range(len(all_box_positions)) if k != i]
            blocked_blocks = set(sokoban_map.obstacles)
            box_dist = box_to_target_bfs(
                sokoban_map,
                (boxes[i].x, boxes[i].y),
                player_pos,
                targets[j],
                blocked=blocked_blocks,
            )
            scores.append(box_dist)
        return scores

    score = float("inf")
    max_index = -1
    box_indices = list(range(len(boxes)))
    target_indices = list(range(len(targets)))
    for perm in permutations(target_indices, len(box_indices)):
        distances = solve_sokoban(perm)
        distance_sum = sum(distances) / len(distances)
        if score > distance_sum:
            max_index = max([(i, val) for i, val in enumerate(distances)], key=lambda x: x[1])[0]
            score = min(distance_sum, score)

    visited, _dists = reachable_positions(sokoban_map, player_pos, forbidden=obstacles)
    score += max(0, _dists.get(all_box_positions[max_index], 20.0) - 1)

    for pos in box_positions:
        x, y = pos
        is_wall = (
            lambda a, b: not (0 <= a < sokoban_map.length and 0 <= b < sokoban_map.width)
            or sokoban_map.map[a][b] == OBSTACLE_SYMBOL
        )
        if (
            (is_wall(x - 1, y) and is_wall(x, y - 1))
            or (is_wall(x - 1, y) and is_wall(x, y + 1))
            or (is_wall(x + 1, y) and is_wall(x, y - 1))
            or (is_wall(x + 1, y) and is_wall(x, y + 1))
        ):
            score += 100.0  # Corner penalty
    return score


# ----- bs ------

import heapq
import math
import random

from search_methods.solver import Solver
from sokoban.map import Map
from sokoban.moves import *
from search_methods.heuritics import sokoban_score_brute

KEY_SCORE = 0
KEY_STATE = 1
KEY_MOVES = 2


class BeamSearch(Solver):
    def __init__(self, map: Map, **kw_args) -> None:
        self.map = map
        self.K = kw_args.get("K", 10)
        self.stochastic = kw_args.get("stochastic", False)
        self.temp = kw_args.get("temp", 1)
        self.temp_scale_factor = kw_args.get("temp_scale_factor", 0.95)
        self.slalom_tolerance = kw_args.get("slalom_tolerance", 5)
        super().__init__(map)

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
            return 0.0
        return 0.0

    def score(self, state: Map, moves: list[int], old_state: Map = None):
        if old_state is not None:
            move_scr = self.move_score(state, old_state)
        else:
            move_scr = 0
        pure_score = sokoban_score_brute(state)
        pull_moves = len(list([move for move in moves if move > DOWN]))
        moves_length = len(moves)
        return pure_score + move_scr

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
        print(f"{probs=}")
        return list(random.choices(candidates, weights=probs, k=self.K))

    def apply_copy(self, state: Map, move: int):
        new_state = state.copy()
        new_state.apply_move(move)
        return new_state

    def solve(self):
        final, is_ok = self._solve()
        if is_ok:
            return final, is_ok
        for i in range(40):
            final, is_ok = self._solve()
            if is_ok:
                return final, is_ok
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
            print(iterations)
            if iterations > self.max_iter:
                best_queue_state = heapq.heappop(queue)
                return best_queue_state, best_queue_state[KEY_STATE].is_solved()

            is_improvement = False
            candidates = []
            for score, state, moves in queue:
                next_valid_moves = self.sample(state)
                if len(next_valid_moves) == 0:
                    continue
                next_valid_states = [
                    (self.apply_copy(state, move), move) for move in next_valid_moves
                ]
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
                        is_improvement = True

                for queue_element in next_valid_state_queue:
                    # if queue_element[KEY_STATE].signature() in seen:
                    #     continue
                    seen.add(queue_element[KEY_STATE].signature())
                    heapq.heappush(candidates, queue_element)

            final_candidate_states = list(filter(lambda x: x[KEY_STATE].is_solved(), candidates))
            if len(final_candidate_states) > 0:
                # we have a final state
                return final_candidate_states[0], True
            print(
                f"{is_improvement=}\n{len(candidates)=}\n{best_candidate_score=}\n{best_queue_score=}"
            )
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
            if best_candidate_score == best_queue_score and best_candidate_score > 5.0:
                slalom_tolerance -= 1
                if slalom_tolerance == 0:
                    temperature /= self.temp_scale_factor
                    slalom_tolerance = self.slalom_tolerance
            temperature *= self.temp_scale_factor
            queue = []
            for _state_tuple in chosen_states:
                heapq.heappush(queue, _state_tuple)


def box_to_targets_bfs(
    sokoban_map: Map, box_pos, player_pos, targets, blocked, box_block=False, update_reachable=True
):
    if box_block:
        reachable, player_dists = reachable_positions(sokoban_map, player_pos, blocked | {box_pos})
    else:
        reachable, player_dists = reachable_positions(sokoban_map, player_pos, blocked)

    if box_pos in targets:
        return 0, box_pos

    min_box_dists = []
    for _dx, _dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        bpos = box_pos[0] - _dx, box_pos[1] - _dy
        ppos = box_pos[0] + _dx, box_pos[1] + _dy

        if not is_inbound(sokoban_map, bpos) or not is_inbound(sokoban_map, ppos):
            continue
        if is_obstacle(sokoban_map, bpos) or is_obstacle(sokoban_map, ppos):
            continue

        # do a modified bfs
        best = {bpos: 0}
        heap = [(0, bpos, player_pos)]
        added = False
        while heap:
            cost, (bx, by), (pl_x, pl_y) = heapq.heappop(heap)
            if (bx, by) in targets:
                added = True
                min_box_dists.append(((_dx, dy), cost))
                break
            print(f"{pl_x=}, {pl_y}")
            if box_block:
                reachable, player_dists = reachable_positions(sokoban_map, (pl_x, pl_y), blocked)
            else:
                reachable, player_dists = reachable_positions(sokoban_map, (pl_x, pl_y), blocked)

            for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                nx, ny = bx + dx, by + dy
                px, py = bx - dx, by - dy
                print(f"{nx=}, {ny} , {px=}, {py}")
                if not (px, py) not in reachable:
                    print(f"{px}, {py} not reachable!")
                    continue
                if (not is_inbound(sokoban_map, (px, py))) or is_obstacle(sokoban_map, (px, py)):

                    print(f"{px}, {py} not inbound!")
                    continue
                if (not is_inbound(sokoban_map, (nx, ny))) or is_obstacle(sokoban_map, (nx, ny)):

                    print(f"{nx}, {ny} not inbound!")
                    continue

                new_cost = cost + 1
                if new_cost < best.get((nx, ny), float("inf")):
                    best[(nx, ny)] = new_cost
                    print("push!")
                    heapq.heappush(heap, (new_cost, (nx, ny), (px, py)))
        if not added:
            min_box_dists.append(((_dx, dy), 1000.0))
    min_cost = float("inf")
    min_ppos = box_pos
    for dir, cost in min_box_dists:
        ppos = box_pos[0] + dir[0], box_pos[1] + dir[1]
        if min_cost > cost:
            min_cost = cost
            min_ppos = ppos
    print(f"{cost=} {min_ppos=}")
    return cost, min_ppos
