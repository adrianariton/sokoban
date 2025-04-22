import os
import pickle as pk
from sokoban.map import Map
from sokoban.moves import *
from sokoban.gif import create_gif

ALGO = "BeamSearch_K_10"
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

            folder_gif_path = f"logs/gifs/{ALGO}/{state_name}"
            os.makedirs(f"logs/play/{ALGO}", exist_ok=True)
            create_gif(folder_gif_path, state_name + ".gif", f"logs/play/{ALGO}")
