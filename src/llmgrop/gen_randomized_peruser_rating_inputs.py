"""
Randomly samples from all generations for each method for presenting them to the user for rating.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import json
from easydict import EasyDict as edict
from tqdm import tqdm
import random
random.seed(0)
from src.llmgrop.constants import ALL_TASKS


PER_BASELINE = 3
PER_OWN = 3
PER_USER = 2


def get_all_image_paths(dirpath):
    all_files = os.listdir(dirpath)
    all_files = [f for f in all_files if f.endswith(".png") or f.endswith(".jpg")]
    all_files = [os.path.join(dirpath, f) for f in all_files]
    return all_files


user_demo_data_dir = "test/llmgrop/USER_DEMO_DATA"
generations_dir = "test/llmgrop/GENERATIONS"
user_ratings_dir = "test/llmgrop/USER_RATINGS"
all_json_files = os.listdir(user_demo_data_dir)
all_json_files = [f for f in all_json_files if f.endswith(".json")]
all_names = [f.split(".")[0] for f in all_json_files]

for user_name in tqdm(all_names, desc="Processing users"):
    user_dict = edict()
    NUM_TOT_RATINGS_PER_USER = 0
    for task in tqdm(ALL_TASKS.keys(), desc="Processing tasks", leave=False):
        task = task.replace(" ", "").lower()
        task_dict = edict()
        task_dict.baselines = edict()
        task_dict.own = edict()
        task_dict.users = edict()
        user_dict[task] = task_dict

        # dir paths
        baselines_generations_dir = f"{generations_dir}/{task}"
        own_generations_dir = f"{generations_dir}/{task}/Synapse/{user_name}"
        users_generations_dir = f"{generations_dir}/{task}/Synapse"

        # own paths
        own_paths = get_all_image_paths(own_generations_dir)
        own_paths = random.sample(own_paths, PER_OWN)
        for o in own_paths:
            user_dict[task].own[o] = None  # placeholder for user ratings
            NUM_TOT_RATINGS_PER_USER += 1

        # baseline paths
        baseline_methods = ['GROP', 'LATP', 'LLM-GROP', 'TPRA']
        for b in baseline_methods:
            baseline_paths = get_all_image_paths(os.path.join(baselines_generations_dir, b))
            baseline_paths = random.sample(baseline_paths, PER_BASELINE)
            for bb in baseline_paths:
                user_dict[task].baselines[bb] = None
                NUM_TOT_RATINGS_PER_USER += 1

        # user paths
        other_users = list(set(all_names) - set([user_name]))
        for ou in other_users:
            ou_paths = get_all_image_paths(os.path.join(users_generations_dir, ou))
            ou_paths = random.sample(ou_paths, PER_USER)
            for u in ou_paths:
                user_dict[task].users[u] = None
                NUM_TOT_RATINGS_PER_USER += 1
    user_dict.NUM_TOT_RATINGS_PER_USER = NUM_TOT_RATINGS_PER_USER

    user_storepath = f"{user_ratings_dir}/{user_name}.json"
    with open(user_storepath, "w") as f:
        json.dump(user_dict, f, indent=4)
