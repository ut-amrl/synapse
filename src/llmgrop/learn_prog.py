"""
Script to learn Object Programs (i.e., a boolean program denoting where a particular object can be placed on the table) from user demonstrations
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import json
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
import re
from tqdm import tqdm
from src.llmgrop.sim import Sim
from src.llmgrop.constants import ALL_OBJECTS, ALL_TASKS, LEARN_TASKS, REFS_THINGS, INFER_TASKS
from src.backend.llm import get_llm_response as gpt4

PREPROMPTS_DIR = "config/llmgrop/llm_preprompts"


def _extract_order(objects: list, desc: str):
    with open(f"{PREPROMPTS_DIR}/base.txt", "r") as f:
        base_pre = f.read()
        base_pre = base_pre.strip()

    with open(f"{PREPROMPTS_DIR}/order.txt", "r") as f:
        order_pre = f.read()
        order_pre = order_pre.strip()
    preprompt = base_pre + "\n\n" + order_pre
    prompt = f"## Input\nobjects: {', '.join(objects)}\ndescription: {desc}\n\n## Output\n"
    text, reason = gpt4(pre_prompt=preprompt,
                        prompt=prompt,
                        model='gpt-4o',
                        temperature=0.0,
                        stop='END',
                        seed=42)
    try:
        match = re.search(r'```python(.*?)```', text, re.DOTALL)
        code = match.group(1).strip()
        namespace = {
            "ordered_objects": [],
        }
        exec(code, namespace)
        order = namespace["ordered_objects"]
        assert len(order) == len(objects)
    except Exception as e:
        print("Exception occured in extracting order: ", e)
        order = objects
    return order, reason


def _learn_object_prog(obj: str, allowed_refs: list, desc: str):
    with open(f"{PREPROMPTS_DIR}/base.txt", "r") as f:
        base_pre = f.read()
        base_pre = base_pre.strip()

    with open(f"{PREPROMPTS_DIR}/generate.txt", "r") as f:
        generate_pre = f.read()
        generate_pre = generate_pre.strip()
    preprompt = base_pre + "\n\n" + generate_pre
    prompt = f"## Input\nobject: {obj}\nallowed_references: {', '.join(allowed_refs)}\ndescription: {desc}\n\n## Output\n"
    text, reason = gpt4(pre_prompt=preprompt,
                        prompt=prompt,
                        model='gpt-4o',
                        temperature=0.0,
                        stop='END',
                        seed=42)
    try:
        match = re.search(r'```python(.*?)```', text, re.DOTALL)
        prog = match.group(1).strip()
        assert len(prog) > 8
    except Exception as e:
        print("Exception occured in extracting order: ", e)
        prog = "def program(loc):\n    return False"
    return prog, reason


def _gen_description(objects1: list, desc1: str, objects2: list, desc2: str, cur_objects: list):
    with open(f"{PREPROMPTS_DIR}/base.txt", "r") as f:
        base_pre = f.read()
        base_pre = base_pre.strip()
    with open(f"{PREPROMPTS_DIR}/gen_desc.txt", "r") as f:
        gen_desc_pre = f.read()
        gen_desc_pre = gen_desc_pre.strip()
    preprompt = base_pre + "\n\n" + gen_desc_pre
    prompt = f"## Input\nobjects1: {', '.join(objects1)}\ndescription1: {desc1}\nobjects2: {', '.join(objects2)}\ndescription2: {desc2}\ncurrent_objects: {', '.join(cur_objects)}\n\n## Output\n"
    text, reason = gpt4(pre_prompt=preprompt,
                        prompt=prompt,
                        model='gpt-4o',
                        temperature=0.0,
                        stop='END',
                        seed=42)
    try:
        match = re.search(r'```python(.*?)```', text, re.DOTALL)
        code = match.group(1).strip()
        namespace = {
            "current_description": "",
        }
        exec(code, namespace)
        cur_desc = namespace["current_description"].strip()
        assert len(cur_desc) > 0
    except Exception as e:
        print("Exception occured in generating description: ", e)
        cur_desc = desc1 + "\n" + desc2
    return cur_desc, reason


def _update_object_prog(obj: str, allowed_refs: list, desc: str, cur_prog: str):
    with open(f"{PREPROMPTS_DIR}/base.txt", "r") as f:
        base_pre = f.read()
        base_pre = base_pre.strip()

    with open(f"{PREPROMPTS_DIR}/update.txt", "r") as f:
        update_pre = f.read()
        update_pre = update_pre.strip()
    preprompt = base_pre + "\n\n" + update_pre
    prompt = f"## Input\nobject: {obj}\nallowed_references: {', '.join(allowed_refs)}\ndescription: {desc}\ncurrent_program:\n```python\n{cur_prog}\n```\n\n## Output\n"
    text, reason = gpt4(pre_prompt=preprompt,
                        prompt=prompt,
                        model='gpt-4o',
                        temperature=0.0,
                        stop='END',
                        seed=42)
    try:
        match = re.search(r'```python(.*?)```', text, re.DOTALL)
        prog = match.group(1).strip()
        assert len(prog) > 8
    except Exception as e:
        print("Exception occured in extracting order: ", e)
        prog = cur_prog
    return prog, reason


def learn_userprog_from_json(json_path):
    # read
    with open(json_path, "r") as f:
        data = json.load(f)
    data = edict(data)
    OBJ_PROGS = edict()
    for obj in ALL_OBJECTS.keys():
        OBJ_PROGS[obj] = edict({"prog": "", "orie_roll": None, "orie_pitch": None, "orie_yaw": None})
        OBJ_PROGS[obj].params = edict({
            "near": None,
            "to_right": None,
            "to_left": None,
            "above": None,
            "below": None,
            "on_top": [None, None, None],
            "inside": None,
            "under": [None, None, None],
        })

    OBJ_PROGS.tasks = edict()
    for task in ALL_TASKS.keys():
        OBJ_PROGS.tasks[task] = edict({"order": None, "description": None})

    for task in tqdm(LEARN_TASKS.keys(), desc="Learn Tasks", leave=False):
        task_data = data[task]
        task_reason = task_data.input_text
        task_objects = LEARN_TASKS[task]["objects"]

        # extract order
        task_order_objects, _ = _extract_order(task_objects, task_reason)
        OBJ_PROGS.tasks[task].order = task_order_objects
        OBJ_PROGS.tasks[task].description = task_reason

        # learn/update object programs
        for iobj, obj in enumerate(task_order_objects):
            # append to REFS_THINGS the objects before obj in the order
            allowed_refs = list(REFS_THINGS.keys()) + task_order_objects[:iobj]
            if OBJ_PROGS[obj].prog == "":
                # learn
                obj_prog, _ = _learn_object_prog(obj, allowed_refs, task_reason)
                OBJ_PROGS[obj].prog = obj_prog
                OBJ_PROGS[obj].orie_roll = float(task_data[obj].roll_slider)
                OBJ_PROGS[obj].orie_pitch = float(task_data[obj].pitch_slider)
                OBJ_PROGS[obj].orie_yaw = float(task_data[obj].yaw_slider)
            else:
                # update
                obj_prog, _ = _update_object_prog(obj, allowed_refs, task_reason, OBJ_PROGS[obj].prog)
                OBJ_PROGS[obj].prog = obj_prog
                # average orientation
                OBJ_PROGS[obj].orie_roll = (OBJ_PROGS[obj].orie_roll + float(task_data[obj].roll_slider)) / 2
                OBJ_PROGS[obj].orie_pitch = (OBJ_PROGS[obj].orie_pitch + float(task_data[obj].pitch_slider)) / 2
                OBJ_PROGS[obj].orie_yaw = (OBJ_PROGS[obj].orie_yaw + float(task_data[obj].yaw_slider)) / 2

    # for infer tasks
    for task in tqdm(INFER_TASKS.keys(), desc="Infer Tasks", leave=False):
        task_objects = INFER_TASKS[task]["objects"]
        task_comes_from = INFER_TASKS[task]["comes_from"]
        task_reason, _ = _gen_description(objects1=LEARN_TASKS[task_comes_from[0]]["objects"],
                                          desc1=OBJ_PROGS.tasks[task_comes_from[0]].description,
                                          objects2=LEARN_TASKS[task_comes_from[1]]["objects"],
                                          desc2=OBJ_PROGS.tasks[task_comes_from[1]].description,
                                          cur_objects=task_objects)

        # extract order
        task_order_objects, _ = _extract_order(task_objects, task_reason)
        OBJ_PROGS.tasks[task].order = task_order_objects
        OBJ_PROGS.tasks[task].description = task_reason

        # HACK: for now, just doing this manually
        # # learn/update object programs
        # for iobj, obj in enumerate(task_order_objects):
        #     # append to REFS_THINGS the objects before obj in the order
        #     allowed_refs = list(REFS_THINGS.keys()) + task_order_objects[:iobj]
        #     if OBJ_PROGS[obj].prog == "":
        #         # learn
        #         obj_prog, _ = _learn_object_prog(obj, allowed_refs, task_reason)
        #         OBJ_PROGS[obj].prog = obj_prog
        #     else:
        #         # update
        #         obj_prog, _ = _update_object_prog(obj, allowed_refs, task_reason, OBJ_PROGS[obj].prog)
        #         OBJ_PROGS[obj].prog = obj_prog

    return OBJ_PROGS


if __name__ == '__main__':
    user_demo_data_dir = "test/llmgrop/USER_DEMO_DATA"
    all_json_files = os.listdir(user_demo_data_dir)
    all_json_files = [f for f in all_json_files if f.endswith(".json")]
    all_names = [f.split(".")[0] for f in all_json_files]

    for user_name in tqdm(all_names):
        json_path = f"{user_demo_data_dir}/{user_name}.json"
        store_path = f"{user_demo_data_dir}/prog_learned/learned_{user_name}.json"
        OBJ_PROGS = learn_userprog_from_json(json_path)
        with open(store_path, "w") as f:
            json.dump(OBJ_PROGS, f, indent=4)
        print(f"Learning successful for {user_name}!")
