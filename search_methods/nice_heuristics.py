from sokoban.map import Map, OBSTACLE_SYMBOL, BOX_SYMBOL, TARGET_SYMBOL
from copy import deepcopy
import heapq
from collections import deque
from sokoban.map import Map, OBSTACLE_SYMBOL
from functools import lru_cache
from itertools import permutations


class Heuristic:
    def __init__(self, **kw_args):
        pass

    def __call__(self, map: Map):
        pass
