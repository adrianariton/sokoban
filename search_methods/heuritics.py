from sokoban.map import Map, OBSTACLE_SYMBOL, BOX_SYMBOL, TARGET_SYMBOL
from copy import deepcopy
import heapq
from collections import deque
from sokoban.map import Map, OBSTACLE_SYMBOL
from functools import lru_cache
from itertools import permutations
from search_methods.Logger import Logger


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


def is_box(sokoban_map: Map, position):
    px, py = position
    boxposes = list(sokoban_map.positions_of_boxes.keys())
    return (px, py) in boxposes


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


def box_to_targets_bfs(
    sokoban_map: Map,
    box_pos,
    player_pos,
    targets,
    blocked,
    box_block=False,
    update_reachable=True,
    verbose=False,
):
    if box_block:
        reachable, player_dists = reachable_positions(sokoban_map, player_pos, blocked | {box_pos})
    else:
        reachable, player_dists = reachable_positions(sokoban_map, player_pos, blocked)

    if box_pos in targets:
        return 0, box_pos

    min_box_dists = []
    for _dx, _dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        bpos = box_pos[0] + _dx, box_pos[1] + _dy
        ppos = box_pos[0] - _dx, box_pos[1] - _dy

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
                min_box_dists.append(((_dx, _dy), cost))
                break
            if box_block:
                reachable, player_dists = reachable_positions(
                    sokoban_map, (pl_x, pl_y), blocked | {(bx, by)}
                )
            else:
                reachable, player_dists = reachable_positions(sokoban_map, (pl_x, pl_y), blocked)

            for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                nx, ny = bx + dx, by + dy
                px, py = bx - dx, by - dy
                if (not is_inbound(sokoban_map, (px, py))) or is_obstacle(sokoban_map, (px, py)):
                    continue
                if (not is_inbound(sokoban_map, (nx, ny))) or is_obstacle(sokoban_map, (nx, ny)):
                    continue
                if (px, py) in blocked or (nx, ny) in blocked:
                    continue
                if (px, py) not in reachable:
                    continue

                if (nx, ny) in targets:
                    sc = 0
                else:
                    sc = 1

                new_cost = cost + player_dists.get((px, py), 1) + sc
                if new_cost < best.get((nx, ny), float("inf")):
                    best[(nx, ny)] = new_cost
                    heapq.heappush(heap, (new_cost, (nx, ny), (px, py)))
        if not added:
            min_box_dists.append(((_dx, _dy), 1000.0))

    if box_block:
        reachable, player_dists = reachable_positions(sokoban_map, player_pos, blocked | {box_pos})
    else:
        reachable, player_dists = reachable_positions(sokoban_map, player_pos, blocked)

    min_cost = 10050.0
    min_ppos = box_pos
    for dir, cost in min_box_dists:
        ppos = box_pos[0] - dir[0], box_pos[1] - dir[1]
        if min_cost > cost:
            min_cost = cost
            min_ppos = ppos
    if min_ppos not in reachable:
        return 10050.0, min_ppos
    return min_cost, min_ppos


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
    _dirs = {box_pos: None}

    if box_pos == target:
        return 0.0, None

    while heap:
        cost, cst, (bx, by) = heapq.heappop(heap)

        if (bx, by) == target:
            return cost, (box_pos[0], box_pos[1])

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
                _dirs[(bx, by)] = (dx, dy)
                heapq.heappush(heap, (new_cost, 1 + cst, (nx, ny)))
    return 55555.0, None


def bfs_distance(sokoban_map: Map, start, targets, forbidden=[], blocked_symbols={OBSTACLE_SYMBOL}):
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
                and (nx, ny) not in forbidden
            ):
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
    return 44444.0  # Large penalty if unreachable


def is_redundant(sokoban_map: Map):
    player_pos = (sokoban_map.player.x, sokoban_map.player.y)
    unbounded = 0
    box = 0
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        px, py = player_pos[0] + dx, player_pos[1] + dy
        if not is_inbound(sokoban_map, (px, py)) or is_obstacle(sokoban_map, (px, py)):
            unbounded += 1
        elif is_box(sokoban_map, (px, py)):
            box += 1

    if unbounded == 3 and box < 1:
        return True
    return False


def is_separated(sokoban_map: Map, box_pos=None, ignore=set([])):
    player_pos = (sokoban_map.player.x, sokoban_map.player.y)
    boxes = list(sokoban_map.boxes.values())
    targets = sokoban_map.targets
    obstacles = sokoban_map.obstacles
    boxposes = list(sokoban_map.positions_of_boxes.keys())
    for box in boxposes:
        if box_pos is not None and box != box_pos:
            continue
        if box_pos in ignore:
            continue
        if box in targets:  # don;t count targets when counting separations
            continue
        dist = bfs_distance(
            sokoban_map, player_pos, [box], forbidden=(set(boxposes) - set(ignore)) | set(obstacles)
        )
        if dist > 100:
            return True
    return False


def is_separating(sokoban_map: Map, box_pos):
    is_sep_orig = is_separated(sokoban_map, None, ignore=set([]))
    is_sep = is_separated(sokoban_map, None, ignore=set([box_pos]))
    if is_sep_orig and not is_sep:
        return True
    return False


def is_deadlocked(sokoban_map: Map):
    player_pos = (sokoban_map.player.x, sokoban_map.player.y)
    boxes = list(sokoban_map.boxes.values())
    targets = sokoban_map.targets
    obstacles = sokoban_map.obstacles
    boxposes = list(sokoban_map.positions_of_boxes.keys())

    free_targets = []
    for t in targets:
        if t not in boxposes:
            free_targets.append(t)

    def target_on_line(x):
        for i in range(sokoban_map.width):
            if (x, i) in free_targets:
                return True
        return False

    def target_on_col(y):
        for i in range(sokoban_map.length):
            if (i, y) in free_targets:
                return True
        return False

    for pos in boxposes:
        x, y = pos
        if (x, y) in targets:
            continue
        is_wall = (
            lambda a, b: not (0 <= a < sokoban_map.length and 0 <= b < sokoban_map.width)
            or sokoban_map.map[a][b] == OBSTACLE_SYMBOL
        )
        is_not_bound = lambda a, b: not (0 <= a < sokoban_map.length and 0 <= b < sokoban_map.width)
        if (
            (is_wall(x - 1, y) and is_wall(x, y - 1))
            or (is_wall(x - 1, y) and is_wall(x, y + 1))
            or (is_wall(x + 1, y) and is_wall(x, y - 1))
            or (is_wall(x + 1, y) and is_wall(x, y + 1))
        ):
            return True

        if is_not_bound(x - 1, y):
            if not target_on_line(x):
                return True
        elif is_not_bound(x + 1, y):
            if not target_on_line(x):
                return True
        elif is_not_bound(x, y + 1):
            if not target_on_col(y):
                return True
        elif is_not_bound(x, y - 1):
            if not target_on_col(y):
                return True

    return False


def sokoban_player_target(sokoban_map: Map, verbose=False) -> float:
    player_pos = (sokoban_map.player.x, sokoban_map.player.y)
    boxes = list(sokoban_map.boxes.values())
    targets = sokoban_map.targets
    obstacles = sokoban_map.obstacles
    boxposes = list(sokoban_map.positions_of_boxes.keys())

    free_targets = []
    for t in targets:
        if t not in boxposes:
            free_targets.append(t)
    if is_deadlocked(sokoban_map):
        return 100000.0
    box_in_targets = 0
    cost = float("inf")
    sums = []
    for box in boxposes:
        if box in targets:
            box_in_targets += 1
            continue
        other_boxes = set(boxes) - {box}
        score, dir = box_to_targets_bfs(
            sokoban_map,
            box,
            player_pos,
            free_targets,
            blocked=set(obstacles) | other_boxes,
            box_block=True,
        )

        dist = bfs_distance(sokoban_map, player_pos, [dir], forbidden=boxposes + obstacles)
        box_cost = score + dist
        cost = min(cost, box_cost)
        sums += [(score, dist)]
    # print(f" > {score=} {dir=} {dist=} {player_pos=}")
    if box_in_targets == len(boxes):
        return -(box_in_targets) * 100

    fsc = 0
    max_d = 0
    for s, d in sums:
        max_d = max(max_d, d)
        fsc += s
    return cost - (box_in_targets) * 100 + max_d


def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


@lru_cache
def sokoban_player_seeded_target(sokoban_map: Map, verbose=False) -> float:
    player_pos = (sokoban_map.player.x, sokoban_map.player.y)
    targets = sokoban_map.targets
    obstacles = sokoban_map.obstacles
    boxposes = list(sokoban_map.positions_of_boxes.keys())

    free_targets = []
    fixed_targets = []
    for t in targets:
        if t not in boxposes:
            free_targets.append(t)
        else:
            fixed_targets.append(t)
    if is_deadlocked(sokoban_map):
        return 100000.0

    def solve(perm, cost_red="min"):
        box_in_targets = 0
        if cost_red == "min":
            cost = float("inf")
        else:
            cost = 0
        sums = []

        _fixed_perm_targets = []
        for _i, box in enumerate(boxposes):
            if perm is not None:
                if box == targets[perm[_i]]:
                    _fixed_perm_targets.append(box)

        if perm is not None:
            t_dist = [(i, manhattan_dist(b, targets[perm[i]])) for i, b in enumerate(boxposes)]
            t_dist = [x for x in t_dist if x[1] <= 2]
            tb_indices = [x[0] for x in t_dist]
            future_fixed_targets = [targets[perm[i]] for i in tb_indices]
        else:
            future_fixed_targets = []
        for _i, box in enumerate(boxposes):
            other_future_fixed_targets = set(future_fixed_targets) - set([targets[perm[_i]]])

            if perm is not None:
                if box == targets[perm[_i]]:
                    box_in_targets += 1
                    continue
            if perm is None:
                if box in free_targets:
                    box_in_targets += 1
                    continue
            other_boxes = set(boxposes) - {box}
            score, dir = box_to_targets_bfs(
                sokoban_map,
                box,
                player_pos,
                set([targets[perm[_i]]]) if perm is not None else free_targets,
                blocked=set(obstacles) | set(_fixed_perm_targets),
                box_block=True,
            )

            score_f, dir_f = box_to_targets_bfs(
                sokoban_map,
                box,
                player_pos,
                set([targets[perm[_i]]]) if perm is not None else free_targets,
                blocked=set(obstacles) | set(_fixed_perm_targets) | set(other_boxes),
                box_block=True,
            )

            dist = bfs_distance(
                sokoban_map,
                player_pos,
                [dir],
                forbidden=set(obstacles) | set(_fixed_perm_targets),
            )

            if score_f > 100:
                box_in_targets -= 1
                dist = bfs_distance(
                    sokoban_map,
                    player_pos,
                    [dir],
                    forbidden=set(obstacles) | set(_fixed_perm_targets),
                )

            box_cost = score + dist

            if cost_red == "min":
                cost = min(cost, box_cost)
            else:
                cost = cost + box_cost
            sums += [(score, dist, dir)]

        if box_in_targets == len(boxposes):
            return 0  # -(box_in_targets) * 100

        fsc = 0
        max_d = float("inf")
        for s, d, _ in sums:
            max_d = min(max_d, d)
            fsc += s
        not_box_in_targets = len(boxposes) - box_in_targets
        return cost + (not_box_in_targets) * 5 + max_d

    box_indices = list(range(len(boxposes)))
    target_indices = list(range(len(targets)))
    _score = float("inf")
    min_perm = box_indices
    for perm in permutations(target_indices, len(box_indices)):
        sc = solve(perm, cost_red="sum")

        if sc < _score:
            min_perm = perm
        _score = min(_score, sc)
    if verbose:
        print(f"{min_perm=}")
        box_to_targs = {boxposes[i]: targets[min_perm[i]] for i in box_indices}
        print(f"{box_to_targs=}")
        print("\n")
    if is_redundant(sokoban_map):
        _score += 2
    return _score


def sokoban_score_least(sokoban_map: Map, verbose=False) -> float:
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

    # print(f"{box_positions=}")
    min_dist, min_dir = float("inf"), 0
    for bpos in box_positions:
        box_to_targets_dist, direction = box_to_targets_bfs(
            sokoban_map, bpos, player_pos, targets, blocked=obstacles, verbose=verbose
        )
        if direction is None:
            return 1000.0

        if min_dist > box_to_targets_dist:
            min_dist = box_to_targets_dist
            min_dir = direction
    score = 0
    box_in_targets = len(box_positions)
    if box_in_targets == len(boxes):
        return -(box_in_targets) * 100
    else:
        score = -(box_in_targets) * 100
    score += min_dist
    if verbose:
        print(f"{box_positions=}")
        print(f"{min_dir=}")
        print(f"{min_dist=}")
    score += bfs_distance(
        sokoban_map,
        start=player_pos,
        targets=set([min_dir]),
        blocked_symbols={OBSTACLE_SYMBOL},
        forbidden=fixed_box_positions + box_positions,
    )
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
            box_dist, dir = box_to_target_bfs(
                sokoban_map,
                (boxes[i].x, boxes[i].y),
                player_pos,
                targets[j],
                blocked=blocked_blocks,
            )

            scores.append((box_dist, dir))
        return scores

    score = float("inf")
    max_index = -1
    box_indices = list(range(len(boxes)))
    target_indices = list(range(len(targets)))
    box_dists = []
    for perm in permutations(target_indices, len(box_indices)):
        result = solve_sokoban(perm)
        distances = [r[0] for r in result]
        distance_sum = sum(distances) / len(distances)
        if score > distance_sum:
            box_dists = result
            max_index = max([(i, val) for i, val in enumerate(distances)], key=lambda x: x[1])[0]
            score = min(distance_sum, score)
    if verbose:
        print(sokoban_map)
        print(f"{box_dists=}")
        print("\n\n")

    reached_blocks = [b[1] for b in box_dists if b != None]
    visited, _dists = reachable_positions(sokoban_map, player_pos, forbidden=obstacles)
    score += max(0, _dists.get(all_box_positions[max_index], 20.0) - 1)

    dist_arr = [_dists.get(b, 20.0) for b in reached_blocks]
    score += min(dist_arr)
    print(f"{dist_arr=}\n{reached_blocks=}")
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
