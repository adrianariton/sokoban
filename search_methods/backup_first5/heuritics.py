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

    heap = [(0.0, 0.0, box_pos)]
    seen = set()
    best = {box_pos: 0.0}

    if box_pos == target:
        return 0.0

    while heap:
        cost, cst, (bx, by) = heapq.heappop(heap)

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
            push_cost = 1  # player_dists.get((px, py), float("inf"))
            new_cost = cst + push_cost

            if new_cost < best.get((nx, ny), float("inf")):
                best[(nx, ny)] = new_cost
                heapq.heappush(heap, (new_cost, 1 + cst, (nx, ny)))
    # unreachable -> very bad
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
def sokoban_score_brute(sokoban_map: Map, verbose=False) -> int:
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
    box_dists = []
    for perm in permutations(target_indices, len(box_indices)):
        distances = solve_sokoban(perm)
        distance_sum = sum(distances) / len(distances)
        if score > distance_sum:
            box_dists = distances
            max_index = max([(i, val) for i, val in enumerate(distances)], key=lambda x: x[1])[0]
            score = min(distance_sum, score)
    if verbose:
        print(sokoban_map)
        print(f"{box_dists=}")
        print("\n\n")
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
