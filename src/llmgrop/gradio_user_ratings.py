import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import gradio as gr
import json
from easydict import EasyDict as edict
import numpy as np
from PIL import Image
import argparse
from src.llmgrop.constants import ALL_TASKS

parser = argparse.ArgumentParser()
parser.add_argument('-u', "--user", type=str, required=True, help="user name")
args = parser.parse_args()


TOP_TITLE = "# LLM-GROP: Anonymized User Ratings"

TOP_DESC = """
Please rate the following generated arrangements for respective tasks, based on how well it aligns with your preference. You'll see 8 different tasks: 5 train and 3 test tasks. The ratings will be 1-10 where qualitatively the ratings represent:\n
* 1: missing critical items
* 3: very poor arrangement and major adjustments needed
* 5: okayish arrangement though not necessarily preferred
* 7: pretty good, but can be improved
* 9: very good, well aligned, tiny changes can be made for perfection
These are different generations from various different methods and baselines.\n\n
Note: you can close the link and reopen, your progress will be saved. Also we are showing you your submitted preference as well just as a reminder!
"""

USER_NAME = args.user
user_ratings_dir = "test/llmgrop/USER_RATINGS"
user_demodata_dir = "test/llmgrop/USER_DEMO_DATA"
USER_RATINGS_JSONPATH = f"{user_ratings_dir}/{USER_NAME}.json"
USER_DEMO_JSONDATA = None
with open(f"{user_demodata_dir}/{USER_NAME}.json", "r") as f:
    USER_DEMO_JSONDATA = edict(json.load(f))
assert os.path.exists(USER_RATINGS_JSONPATH), f"User ratings data not found at {USER_RATINGS_JSONPATH}"

USER_RATINGS_DATA_LOADED = None
PROGRESS_TRACKER = edict()
PROGRESS_TRACKER.task1 = None
PROGRESS_TRACKER.task2 = None
PROGRESS_TRACKER.task3 = None
PROGRESS_TRACKER.task4 = None
PROGRESS_TRACKER.task5 = None
PROGRESS_TRACKER.task6 = None
PROGRESS_TRACKER.task7 = None
PROGRESS_TRACKER.task8 = None
PROGRESS_TRACKER.task_count_tot_needed = None
PROGRESS_TRACKER.cur_task = None
PROGRESS_TRACKER.task_paths = None

ALL_TASKS_DONE = False


def _start_recording():
    global TOP_TITLE, TOP_DESC, USER_NAME, USER_RATINGS_JSONPATH, USER_DEMO_JSONDATA, USER_RATINGS_DATA_LOADED, PROGRESS_TRACKER, ALL_TASKS_DONE
    if PROGRESS_TRACKER.task_count_tot_needed is None:
        # not started yet, so start
        with open(USER_RATINGS_JSONPATH, "r") as f:
            USER_RATINGS_DATA_LOADED = edict(json.load(f))

        for tid in range(1, 9):
            task_name = f"task{tid}"
            task_baselines_paths_dict = USER_RATINGS_DATA_LOADED[task_name].baselines
            task_own_paths_dict = USER_RATINGS_DATA_LOADED[task_name].own
            task_users_paths_dict = USER_RATINGS_DATA_LOADED[task_name].users
            task_values = list(task_baselines_paths_dict.values()) + list(task_own_paths_dict.values()) + list(task_users_paths_dict.values())
            PROGRESS_TRACKER[task_name] = len([v for v in task_values if v is not None])

        task_count_tot_needed = USER_RATINGS_DATA_LOADED.NUM_TOT_RATINGS_PER_USER // 8
        PROGRESS_TRACKER.task_count_tot_needed = task_count_tot_needed

        for tid in range(1, 9):
            task_name = f"task{tid}"
            if PROGRESS_TRACKER[task_name] < task_count_tot_needed:
                PROGRESS_TRACKER.cur_task = tid
                PROGRESS_TRACKER.task_paths = sorted(list(USER_RATINGS_DATA_LOADED[task_name].baselines.keys()) + list(USER_RATINGS_DATA_LOADED[task_name].own.keys()) + list(USER_RATINGS_DATA_LOADED[task_name].users.keys()))
                break

        if PROGRESS_TRACKER.cur_task is None:
            ALL_TASKS_DONE = True
            return "All tasks done! You can close the window now.", np.array(Image.open("config/llmgrop/you_are_done_image.png")), \
                "All tasks done! You can close the window now.", np.array(Image.open("config/llmgrop/you_are_done_image.png"))

    task_name = f"task{PROGRESS_TRACKER.cur_task}"
    _task_obj_list = ', '.join([f"{i+1}. {obj}" for i, obj in enumerate(ALL_TASKS[f"Task {PROGRESS_TRACKER.cur_task}"]["objects"])])
    _progress = f"Progress: {PROGRESS_TRACKER[task_name]}/{PROGRESS_TRACKER.task_count_tot_needed} done"
    _task_textbox = f"[Task {PROGRESS_TRACKER.cur_task}]: {_task_obj_list}\n{_progress}"
    if 1 <= PROGRESS_TRACKER.cur_task <= 5:
        # train tasks
        _pref_img = np.array(Image.open(f"{user_demodata_dir}/representatives/{USER_NAME}/Task {PROGRESS_TRACKER.cur_task}.png"))
        _pref_NL = USER_DEMO_JSONDATA[f"Task {PROGRESS_TRACKER.cur_task}"].input_text
    else:
        # test tasks
        _pref_img = np.array(Image.open("config/llmgrop/test_task.png"))
        _pref_NL = "Test task. Based on preference learned from train tasks."
    _gen_img = np.array(Image.open(PROGRESS_TRACKER.task_paths[PROGRESS_TRACKER[task_name]]))
    return _task_textbox, _pref_img, _pref_NL, _gen_img


def _back(rating: str):
    global TOP_TITLE, TOP_DESC, USER_NAME, USER_RATINGS_JSONPATH, USER_DEMO_JSONDATA, USER_RATINGS_DATA_LOADED, PROGRESS_TRACKER, ALL_TASKS_DONE
    if PROGRESS_TRACKER.task_count_tot_needed is None:
        # not started yet
        return "Click Start button.", np.array(Image.open("config/llmgrop/start_button.png")), \
            "Click Start button.", np.array(Image.open("config/llmgrop/start_button.png"))

    if ALL_TASKS_DONE:
        return "All tasks done! You can close the window now.", np.array(Image.open("config/llmgrop/you_are_done_image.png")), \
            "All tasks done! You can close the window now.", np.array(Image.open("config/llmgrop/you_are_done_image.png"))

    task_name = f"task{PROGRESS_TRACKER.cur_task}"
    if PROGRESS_TRACKER[task_name] > 0:
        PROGRESS_TRACKER[task_name] -= 1

    _task_obj_list = ', '.join([f"{i+1}. {obj}" for i, obj in enumerate(ALL_TASKS[f"Task {PROGRESS_TRACKER.cur_task}"]["objects"])])
    _progress = f"Progress: {PROGRESS_TRACKER[task_name]}/{PROGRESS_TRACKER.task_count_tot_needed} done"
    _task_textbox = f"[Task {PROGRESS_TRACKER.cur_task}]: {_task_obj_list}\n{_progress}"
    if 1 <= PROGRESS_TRACKER.cur_task <= 5:
        # train tasks
        _pref_img = np.array(Image.open(f"{user_demodata_dir}/representatives/{USER_NAME}/Task {PROGRESS_TRACKER.cur_task}.png"))
        _pref_NL = USER_DEMO_JSONDATA[f"Task {PROGRESS_TRACKER.cur_task}"].input_text
    else:
        # test tasks
        _pref_img = np.array(Image.open("config/llmgrop/test_task.png"))
        _pref_NL = "Test task. Based on preference learned from train tasks."
    _gen_img = np.array(Image.open(PROGRESS_TRACKER.task_paths[PROGRESS_TRACKER[task_name]]))
    return _task_textbox, _pref_img, _pref_NL, _gen_img


def _next(rating: str):
    global TOP_TITLE, TOP_DESC, USER_NAME, USER_RATINGS_JSONPATH, USER_DEMO_JSONDATA, USER_RATINGS_DATA_LOADED, PROGRESS_TRACKER, ALL_TASKS_DONE
    if PROGRESS_TRACKER.task_count_tot_needed is None:
        # not started yet
        return "Click Start button.", np.array(Image.open("config/llmgrop/start_button.png")), \
            "Click Start button.", np.array(Image.open("config/llmgrop/start_button.png"))
    task_name = f"task{PROGRESS_TRACKER.cur_task}"
    if PROGRESS_TRACKER[task_name] >= PROGRESS_TRACKER.task_count_tot_needed:
        # move to next task
        PROGRESS_TRACKER.cur_task += 1
        if PROGRESS_TRACKER.cur_task > 8:
            ALL_TASKS_DONE = True
        else:
            task_name = f"task{PROGRESS_TRACKER.cur_task}"
            PROGRESS_TRACKER.task_paths = sorted(list(USER_RATINGS_DATA_LOADED[task_name].baselines.keys()) + list(USER_RATINGS_DATA_LOADED[task_name].own.keys()) + list(USER_RATINGS_DATA_LOADED[task_name].users.keys()))

            _task_obj_list = ', '.join([f"{i+1}. {obj}" for i, obj in enumerate(ALL_TASKS[f"Task {PROGRESS_TRACKER.cur_task}"]["objects"])])
            _progress = f"Progress: {PROGRESS_TRACKER[task_name]}/{PROGRESS_TRACKER.task_count_tot_needed} done"
            _task_textbox = f"[Task {PROGRESS_TRACKER.cur_task}]: {_task_obj_list}\n{_progress}"
            if 1 <= PROGRESS_TRACKER.cur_task <= 5:
                # train tasks
                _pref_img = np.array(Image.open(f"{user_demodata_dir}/representatives/{USER_NAME}/Task {PROGRESS_TRACKER.cur_task}.png"))
                _pref_NL = USER_DEMO_JSONDATA[f"Task {PROGRESS_TRACKER.cur_task}"].input_text
            else:
                # test tasks
                _pref_img = np.array(Image.open("config/llmgrop/test_task.png"))
                _pref_NL = "Test task. Based on preference learned from train tasks."
            _gen_img = np.array(Image.open(PROGRESS_TRACKER.task_paths[PROGRESS_TRACKER[task_name]]))
            return _task_textbox, _pref_img, _pref_NL, _gen_img

    if ALL_TASKS_DONE:
        return "All tasks done! You can close the window now.", np.array(Image.open("config/llmgrop/you_are_done_image.png")), \
            "All tasks done! You can close the window now.", np.array(Image.open("config/llmgrop/you_are_done_image.png"))

    if rating == "N/A":
        _task_obj_list = ', '.join([f"{i+1}. {obj}" for i, obj in enumerate(ALL_TASKS[f"Task {PROGRESS_TRACKER.cur_task}"]["objects"])])
        _progress = f"Progress: {PROGRESS_TRACKER[task_name]}/{PROGRESS_TRACKER.task_count_tot_needed} done"
        _task_textbox = f"[Task {PROGRESS_TRACKER.cur_task}]: {_task_obj_list}\n{_progress}\nYou need to rate the generation to proceed."
        if 1 <= PROGRESS_TRACKER.cur_task <= 5:
            # train tasks
            _pref_img = np.array(Image.open(f"{user_demodata_dir}/representatives/{USER_NAME}/Task {PROGRESS_TRACKER.cur_task}.png"))
            _pref_NL = USER_DEMO_JSONDATA[f"Task {PROGRESS_TRACKER.cur_task}"].input_text
        else:
            # test tasks
            _pref_img = np.array(Image.open("config/llmgrop/test_task.png"))
            _pref_NL = "Test task. Based on preference learned from train tasks."
        _gen_img = np.array(Image.open(PROGRESS_TRACKER.task_paths[PROGRESS_TRACKER[task_name]]))
        return _task_textbox, _pref_img, _pref_NL, _gen_img

    rate_int = int(rating)
    if USER_NAME in PROGRESS_TRACKER.task_paths[PROGRESS_TRACKER[task_name]]:
        # own
        USER_RATINGS_DATA_LOADED[task_name].own[PROGRESS_TRACKER.task_paths[PROGRESS_TRACKER[task_name]]] = rate_int
    elif "Synapse" not in PROGRESS_TRACKER.task_paths[PROGRESS_TRACKER[task_name]]:
        # baselines
        USER_RATINGS_DATA_LOADED[task_name].baselines[PROGRESS_TRACKER.task_paths[PROGRESS_TRACKER[task_name]]] = rate_int
    else:
        # users
        USER_RATINGS_DATA_LOADED[task_name].users[PROGRESS_TRACKER.task_paths[PROGRESS_TRACKER[task_name]]] = rate_int

    with open(USER_RATINGS_JSONPATH, "w") as f:
        json.dump(USER_RATINGS_DATA_LOADED, f, indent=4)
    PROGRESS_TRACKER[task_name] += 1

    if PROGRESS_TRACKER[task_name] >= PROGRESS_TRACKER.task_count_tot_needed:
        return "Task done! Click Next button.", np.array(Image.open(f"config/llmgrop/{task_name}_done.png")), \
            "Task done! Click Next button.", np.array(Image.open(f"config/llmgrop/{task_name}_done.png"))

    _task_obj_list = ', '.join([f"{i+1}. {obj}" for i, obj in enumerate(ALL_TASKS[f"Task {PROGRESS_TRACKER.cur_task}"]["objects"])])
    _progress = f"Progress: {PROGRESS_TRACKER[task_name]}/{PROGRESS_TRACKER.task_count_tot_needed} done"
    _task_textbox = f"[Task {PROGRESS_TRACKER.cur_task}]: {_task_obj_list}\n{_progress}"
    if 1 <= PROGRESS_TRACKER.cur_task <= 5:
        # train tasks
        _pref_img = np.array(Image.open(f"{user_demodata_dir}/representatives/{USER_NAME}/Task {PROGRESS_TRACKER.cur_task}.png"))
        _pref_NL = USER_DEMO_JSONDATA[f"Task {PROGRESS_TRACKER.cur_task}"].input_text
    else:
        # test tasks
        _pref_img = np.array(Image.open("config/llmgrop/test_task.png"))
        _pref_NL = "Test task. Based on preference learned from train tasks."
    _gen_img = np.array(Image.open(PROGRESS_TRACKER.task_paths[PROGRESS_TRACKER[task_name]]))
    return _task_textbox, _pref_img, _pref_NL, _gen_img


with gr.Blocks() as interface:
    gr.Markdown(TOP_TITLE)
    gr.Markdown(TOP_DESC)
    pref_NL = gr.Textbox(label="Your Explanation", value="Click Start Button.")
    pref_img = gr.Image(type="numpy", label="Your Original Submission for Preference", value="config/llmgrop/start_button.png")
    task_textbox = gr.Textbox(label="TASK", value="Click Start Button.")
    gen_img = gr.Image(type="numpy", label="Anonymous method generation", value="config/llmgrop/start_button.png")
    ratings_radio = gr.Radio(label="Rating. Press Next to record the rating", choices=[str(i) for i in range(1, 11)] + ["N/A"], value="N/A")

    next_button = gr.Button(value="Next", variant="primary")
    next_button.click(
        fn=_next,
        inputs=[ratings_radio],
        outputs=[task_textbox, pref_img, pref_NL, gen_img],
    )
    back_button = gr.Button(value="Previous", variant="primary")
    back_button.click(
        fn=_back,
        inputs=[ratings_radio],
        outputs=[task_textbox, pref_img, pref_NL, gen_img],
    )
    start_button = gr.Button(value="Start", variant="primary")
    start_button.click(
        fn=_start_recording,
        inputs=[],
        outputs=[task_textbox, pref_img, pref_NL, gen_img],
    )


interface.launch(share=True)
