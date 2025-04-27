import os
import pickle as pk
from sokoban.map import Map
from sokoban.moves import *
from sokoban.gif import save_images

ALGO = "BeamSearch_K_20"
FOLDER_PATH = "logs/" + ALGO

for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".pkl"):
        file_path = os.path.join(FOLDER_PATH, filename)
        with open(file_path, "rb") as f:
            data = pk.load(f)
            state_name = file_path.split("/")[-1][:-4]
            state_name = state_name.split("_result")[0]
            print(data)
            (score, state, moves), is_ok = data

            map = Map.from_yaml(f"tests/{state_name}.yaml")
            print(state_name)
            print(score)
            print(moves)
            print("-----------------------\n")

            maps = [map]
            c_map = map.copy()
            for move in moves:
                c_map = c_map.copy()
                c_map.apply_move(move)
                maps.append(c_map)

            folder_gif_path = f"logs/gifs/{ALGO}/{state_name}"
            save_images(maps, folder_gif_path)
