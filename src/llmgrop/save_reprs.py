import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.llmgrop.sim import Sim
from src.llmgrop.constants import LEARN_TASKS


def gen_rgb_from_userjson(json_path, task="Task 1"):
    with open(json_path, "r") as f:
        data = json.load(f)
    task_data = data[task]
    objects = LEARN_TASKS[task]["objects"]
    final_positions = []
    final_orientations = []
    for iobj, obj in enumerate(objects):
        final_positions.append([task_data[obj]["x_slider"], task_data[obj]["y_slider"], task_data[obj]["z_slider"]])
        final_orientations.append([task_data[obj]["roll_slider"], task_data[obj]["pitch_slider"], task_data[obj]["yaw_slider"]])
    rgb, *_ = Sim.quick_simulate(objects, final_positions, final_orientations)
    return rgb


user_demo_data_dir = "test/llmgrop/USER_DEMO_DATA"
all_json_files = os.listdir(user_demo_data_dir)
all_json_files = [f for f in all_json_files if f.endswith(".json")]
all_names = [f.split(".")[0] for f in all_json_files]

for user_name in tqdm(all_names):
    json_path = f"{user_demo_data_dir}/{user_name}.json"
    dir_path = f"{user_demo_data_dir}/representatives/{user_name}"
    os.makedirs(dir_path, exist_ok=True)

    for task in LEARN_TASKS.keys():
        rgb = gen_rgb_from_userjson(json_path, task)
        plt.imsave(f"{dir_path}/{task}.png", rgb)
