from sokoban.map import Map
import random
from search_methods.Logger import Logger


class Solver:
    def __init__(self, map: Map, **kw_args) -> None:
        self.map = map
        self.seed = kw_args.get("seed", 123)
        self.max_iter = kw_args.get("max_iter", 2000)
        self.logger = kw_args.get("logger", Logger())
        random.seed(self.seed)

    def solve(self):
        raise NotImplementedError
