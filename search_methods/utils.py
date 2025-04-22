import heapq
import math
import random

from search_methods.solver import Solver
from sokoban.map import Map
from sokoban.moves import *
from search_methods.heuritics import sokoban_player_target


def print_map(map: Map, heuristic=sokoban_player_target):
    name = ""
    for i in range(map.length):
        for j in range(map.width):

            if i == map.player.x + 1 and j == map.player.y:
                copy_map = map.copy()
                try:
                    copy_map.apply_move(UP)
                    name += f"{heuristic(copy_map)} "
                    continue
                except:
                    pass
            elif i == map.player.x - 1 and j == map.player.y:
                copy_map = map.copy()
                try:
                    copy_map.apply_move(DOWN)
                    name += f"{heuristic(copy_map)} "
                    continue
                except:
                    pass
            elif i == map.player.x and j == map.player.y + 1:
                copy_map = map.copy()
                try:
                    copy_map.apply_move(RIGHT)
                    name += f"{heuristic(copy_map)} "
                    continue
                except:
                    pass
            elif i == map.player.x and j == map.player.y - 1:
                copy_map = map.copy()
                try:
                    copy_map.apply_move(LEFT)
                    name += f"{heuristic(copy_map)} "
                    continue
                except:
                    pass

            if map.player.x == i and map.player.y == j:
                name += f"{map.player.get_symbol()} "
            elif map.map[i][j] == 1:
                name += f"/ "
            elif map.map[i][j] == 2:
                name += f"B "
            elif map.map[i][j] == 3:
                name += f"X "
            else:
                name += f"_ "

        name += "\n"

    pieces = name.split("\n")
    aligned_corner = reversed(pieces)
    return "\n".join(aligned_corner)
