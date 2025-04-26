import matplotlib.pyplot as plt
import os
import json

ALGORITHMS = ["LRTAStar", "BeamSearch_K_10"]
TESTS = [
    "easy_map1",
    "easy_map2",
    "hard_map1",
    "hard_map2",
    "medium_map1",
    "medium_map2",
    "large_map1",
    "large_map2",
]


def read_data():
    dict_ = {algo_: {} for algo_ in ALGORITHMS}
    for algo in ALGORITHMS:
        folder_path = "logs/" + algo
        for filename in os.listdir(folder_path):
            if filename.endswith(".yaml"):
                file_path = os.path.join(folder_path, filename)
                state_name = file_path.split("/")[-1][:-5]
                with open(file_path, "r") as f:
                    for line in f.readlines():
                        k, v = line.strip().split("=")
                        k = k.strip()
                        v = v.strip()
                        v = eval(v)
                        dict_[algo][state_name] = dict_[algo].get(state_name, {})
                        dict_[algo][state_name][k] = v
    return dict_


def get_values(dict_, key="time"):
    values_ = {algo_: {} for algo_ in ALGORITHMS}
    for algo in ALGORITHMS:
        for test in TESTS:

            values_[algo][test] = dict_[algo].get(test, {}).get(key, None)
    return values_


KEYS = ["time", "moves_no", "expanded"]
TITLES = ["Running Time (seconds)", "Number of Moves (#)", "Expanded States (#)"]


def plot_key(key, title):
    data = read_data()
    vals = get_values(data, key=key)
    print(vals)

    # Create plot
    plt.figure(figsize=(8, 5))
    plt.xticks(rotation=15)

    plt.plot(TESTS, vals[ALGORITHMS[0]].values(), marker="o", label=f"{ALGORITHMS[0]}")
    plt.plot(TESTS, vals[ALGORITHMS[1]].values(), marker="s", label=f"{ALGORITHMS[1]}")

    # Logarithmic y-axis
    plt.yscale("log")

    # Labels, title, grid
    plt.xlabel("Test Case")
    plt.ylabel(title)
    plt.title("Algorithm Performance Comparison (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Save the figure
    plt.savefig(f"graphs/algorithm_comparison_{key}.png", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    for key, ttl in zip(KEYS, TITLES):
        plot_key(key, title=ttl)
