import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import math
import gradio as gr
import json
from easydict import EasyDict as edict
from src.llmgrop.sim import Sim
from src.llmgrop.constants import LEARN_TASKS


TASKS_TO_QUERY = LEARN_TASKS

SAVE_ROOTDIR = "test/llmgrop/USER_DEMO_DATA"
DEFAULTS = edict({
    "task_selector": "Task 1",
    "obj_selector_slider": 1,
    "x_slider": 1.1,
    "y_slider": -0.22,
    "z_slider": 0.65,
    "roll_slider": 0.0,
    "pitch_slider": 0.0,
    "yaw_slider": 0.0,
})

USER_INPUTS = edict()
for itx, task in enumerate(TASKS_TO_QUERY.keys()):
    USER_INPUTS[task] = edict({"input_text": ""})
    for iobj, obj in enumerate(TASKS_TO_QUERY[task]["objects"]):
        USER_INPUTS[task][obj] = edict({
            "x_slider": DEFAULTS.x_slider,
            "y_slider": DEFAULTS.y_slider,
            "z_slider": DEFAULTS.z_slider,
            "roll_slider": DEFAULTS.roll_slider,
            "pitch_slider": DEFAULTS.pitch_slider,
            "yaw_slider": DEFAULTS.yaw_slider,
        })


def place_objects(task: str, obj_selector: int, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
    global USER_INPUTS
    if obj_selector > len(TASKS_TO_QUERY[task]["objects"]):
        obj_selector = len(TASKS_TO_QUERY[task]["objects"])
    USER_INPUTS[task][TASKS_TO_QUERY[task]["objects"][obj_selector - 1]].x_slider = x
    USER_INPUTS[task][TASKS_TO_QUERY[task]["objects"][obj_selector - 1]].y_slider = y
    USER_INPUTS[task][TASKS_TO_QUERY[task]["objects"][obj_selector - 1]].z_slider = z
    USER_INPUTS[task][TASKS_TO_QUERY[task]["objects"][obj_selector - 1]].roll_slider = roll
    USER_INPUTS[task][TASKS_TO_QUERY[task]["objects"][obj_selector - 1]].pitch_slider = pitch
    USER_INPUTS[task][TASKS_TO_QUERY[task]["objects"][obj_selector - 1]].yaw_slider = yaw
    objects = TASKS_TO_QUERY[task]["objects"]
    positions = []
    orientations = []
    for iobj, obj in enumerate(objects):
        # positions.append([x, y, z])
        # orientations.append([roll, pitch, yaw])
        positions.append([USER_INPUTS[task][obj].x_slider, USER_INPUTS[task][obj].y_slider, USER_INPUTS[task][obj].z_slider])
        orientations.append([USER_INPUTS[task][obj].roll_slider, USER_INPUTS[task][obj].pitch_slider, USER_INPUTS[task][obj].yaw_slider])
    rgb, *_ = Sim.quick_simulate(objects, positions, orientations)
    # objects list mapped to index+1 number
    obj_list = ', '.join([f"{i+1}. {obj}" for i, obj in enumerate(objects)])  # 1. dinner plate, 2. dinner fork, 3. dinner knife
    return rgb, obj_list


def record_task(task: str, input_text: str):
    global USER_INPUTS
    USER_INPUTS[task].input_text = input_text
    return f"Task submission for {task} successful!"


def record_all(name: str):
    global USER_INPUTS
    file_name = f"{name.replace(' ', '_').lower()}.json"
    file_path = os.path.join(SAVE_ROOTDIR, file_name)
    with open(file_path, 'w') as file:
        json.dump(USER_INPUTS, file, indent=4)  # Convert EasyDict to dict before dumping
    return "Submission successful! Thanks a lot for your time!!."


with gr.Blocks() as interface:
    name_textbox = gr.Textbox(label="Full Name", placeholder="e.g. John Doe")
    task_selector = gr.Radio(label="Task Selector", choices=[f"Task {i}" for i in range(1, 6)], value=DEFAULTS.task_selector)
    obj_selector_slider = gr.Slider(label="Object Selector", minimum=1, maximum=3, step=1, value=DEFAULTS.obj_selector_slider)
    x_slider = gr.Slider(label="X", minimum=0.749, maximum=1.251, step=0.001, value=DEFAULTS.x_slider)
    y_slider = gr.Slider(label="Y", minimum=-0.251, maximum=0.251, step=0.001, value=DEFAULTS.y_slider)
    z_slider = gr.Slider(label="Z", minimum=0.64, maximum=1.10, step=0.01, value=DEFAULTS.z_slider)
    roll_slider = gr.Slider(label="Roll", minimum=0, maximum=2 * math.pi, step=math.pi / 4, value=DEFAULTS.roll_slider)
    pitch_slider = gr.Slider(label="Pitch", minimum=0, maximum=2 * math.pi, step=math.pi / 4, value=DEFAULTS.pitch_slider)
    yaw_slider = gr.Slider(label="Yaw", minimum=0, maximum=2 * math.pi, step=math.pi / 4, value=DEFAULTS.yaw_slider)
    gr.Interface(
        fn=place_objects,
        inputs=[task_selector, obj_selector_slider, x_slider, y_slider, z_slider, roll_slider, pitch_slider, yaw_slider],
        outputs=[gr.Image(type="numpy", label="Preferred Arrangement", value="config/llmgrop/background_default.png"),
                 gr.Textbox(label="objects", value="dinner plate, dinner fork, dinner knife")],
        live=True,
        title="LLM-GROP Task User Interface",
        description="Imagine you are at your home sitting at your dining table (near the bottom edge of the table). You have a assistant home robot and you want it to give you the most ideal preferred arrangement of objects on your dinner table. Below you will see 5 different scenarios/tasks and you need to teach the robot by showing (1) your preferred arrangement, and (2) your reason for choosing that arrangement. Please set the arrangement of the objects according to your preference as closely as possible (object locations, orientations, and relative poses matter) and be precise and descriptive in your reason. Please go through EACH scenario and use the 'Task Submit' button AFTER you have set that task's preferred arrangement + your reason in the text box for choosing that arrangement (note: you can resubmit tasks if you need to modify).\n\nOnce done with all tasks, use 'Submit' button for a FINAL submission. Thank you!\n(Note: start with object 1, then 2 and so on for each task, else sliders may mess up your chosen positions)",
        allow_flagging='never',
    )
    input_textbox = gr.Textbox(label="Descriptive Reason", placeholder="e.g. Plate near the edge of table, knife on the right facing the plate, and fork on the left of the plate. Both close to the plate.")
    result_textbox = gr.Textbox(label="Output", visible=True)
    task_submit_button = gr.Button(value="Task Submit")
    submit_button = gr.Button(value="Submit", variant="primary")
    task_submit_button.click(
        fn=record_task,
        inputs=[task_selector, input_textbox],
        outputs=result_textbox,
    )
    submit_button.click(
        fn=record_all,
        inputs=[name_textbox],
        outputs=result_textbox,
    )
interface.launch(share=True)
