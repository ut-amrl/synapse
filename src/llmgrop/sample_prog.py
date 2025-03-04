"""
Script to :
(1) Parameter synthesis of object programs from program sketches (i.e., fill holes)
(2) Sample generations from the learned object programs
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import json
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
import re
from tqdm import tqdm
import shutil
import numpy as np
import ast
import itertools
import argparse
import random
random.seed(0)
from src.llmgrop.sim import Sim
from src.llmgrop.constants import ALL_OBJECTS, LEARN_TASKS, REFS_THINGS, PARAMS_DEFAULTS

# functionv -> vectorized impl version of function
# DSL functions args: k: ref_obj, loc: x,y,z, obj: place_obj


def extract_conditions_from_prog(source_code):
    """
    Extract conditions from the provided function source code (as string).

    Returns a list of conditions found in `if`, `elif`, and `else` blocks in string form.
    """
    # Parse the source code into an AST (Abstract Syntax Tree)
    tree = ast.parse(source_code)

    conditions = []

    def parse_condition(node):
        """
        Recursively parse the condition AST node to reconstruct the condition as a string.
        """
        if isinstance(node, ast.BoolOp):
            # Handle boolean operations (and/or)
            op = ' and ' if isinstance(node.op, ast.And) else ' or '
            return op.join([parse_condition(value) for value in node.values])
        elif isinstance(node, ast.Call):
            # Handle function calls, e.g., near('bottom edge')
            func_name = node.func.id
            arg_values = []
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    # Python 3.8+
                    arg_values.append(repr(arg.value))
                elif isinstance(arg, ast.Str):
                    # Python versions before 3.8
                    arg_values.append(repr(arg.s))
                elif isinstance(arg, ast.Name):
                    arg_values.append(arg.id)
                else:
                    arg_values.append(parse_condition(arg))
            args = ", ".join(arg_values)
            return f"{func_name}({args})"
        elif isinstance(node, ast.Compare):
            # Handle comparison operations
            left = parse_condition(node.left)
            ops = {'Eq': '==', 'NotEq': '!=', 'Lt': '<', 'LtE': '<=', 'Gt': '>', 'GtE': '>='}
            comparisons = []
            for op, comparator in zip(node.ops, node.comparators):
                comp_op = ops[type(op).__name__]
                right = parse_condition(comparator)
                comparisons.append(f"{left} {comp_op} {right}")
            return ' and '.join(comparisons)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Str):
            # For string literals in versions before Python 3.8
            return repr(node.s)
        else:
            # Handle other node types if necessary
            return ast.dump(node)

    # Start parsing from the top-level nodes
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            # We found a function definition; let's look for if statements inside it
            for stmt in node.body:
                if isinstance(stmt, ast.If):
                    current_node = stmt
                    while True:
                        # Extract the condition
                        cond = parse_condition(current_node.test)
                        conditions.append(cond)
                        # Check if there's an elif or else
                        if current_node.orelse:
                            if len(current_node.orelse) == 1 and isinstance(current_node.orelse[0], ast.If):
                                current_node = current_node.orelse[0]
                            else:
                                # No more elif; exit the loop
                                break
                        else:
                            # No else block; exit the loop
                            break

    return conditions


def transform_code(code, loc_arg, obj_arg):
    # Add loc and obj to all functions
    code = re.sub(r'(\w+)\((.*?)\)', r'\1(\2, ' + loc_arg + ', ' + obj_arg + ')', code)
    code = code.replace('program(loc, loc, obj)', 'program(loc, obj)')
    return code


def bbox_corner_coords(xc, yc, xsize, ysize):
    """
    lower left, upper right
    """
    lower_left = (xc - xsize / 2, yc + ysize / 2)
    upper_right = (xc + xsize / 2, yc - ysize / 2)
    return lower_left, upper_right


def do_boxes_overlap(box1, box2):
    """
    box format: (xc, yc, xsize, ysize)
    """
    # blow up size a bit to avoid edge cases
    # box1 = (box1[0], box1[1], box1[2] + 0.02, box1[3] + 0.02)
    # box2 = (box2[0], box2[1], box2[2] + 0.02, box2[3] + 0.02)
    ll1, ur1 = bbox_corner_coords(*box1)
    ll2, ur2 = bbox_corner_coords(*box2)
    x1_min, y1_max = ll1
    x1_max, y1_min = ur1
    x2_min, y2_max = ll2
    x2_max, y2_min = ur2
    # if either box is to the left of the other
    if y1_min > y2_max or y2_min > y1_max:
        return False
    # if either box is above the other
    if x1_max < x2_min or x2_max < x1_min:
        return False
    return True


def bbox_corner_coordsv(xc, yc, xsize, ysize):
    """
    Vectorized version: Computes the lower-left and upper-right corner coordinates
    for given box center coordinates and sizes. Supports scalar or array inputs.

    Parameters:
        xc (float or np.ndarray): X-coordinate(s) of the box center(s).
        yc (float or np.ndarray): Y-coordinate(s) of the box center(s).
        xsize (float or np.ndarray): Width(s) of the box(es).
        ysize (float or np.ndarray): Height(s) of the box(es).

    Returns:
        lower_left (np.ndarray): Array of lower-left corner coordinates.
        upper_right (np.ndarray): Array of upper-right corner coordinates.
    """
    xc = np.asarray(xc)
    yc = np.asarray(yc)
    xsize = np.asarray(xsize)
    ysize = np.asarray(ysize)

    lower_left = np.stack((xc - xsize / 2, yc + ysize / 2), axis=-1)
    upper_right = np.stack((xc + xsize / 2, yc - ysize / 2), axis=-1)
    return lower_left, upper_right


def do_boxes_overlapv(box1, box2):
    """
    Vectorized version: Determines if box1 overlaps with one or more boxes in box2.

    Parameters:
        box1 (tuple or list): Single box defined as (xc, yc, xsize, ysize).
        box2 (np.ndarray): Array of boxes, each defined as (xc, yc, xsize, ysize).

    Returns:
        overlaps (np.ndarray): Boolean array indicating overlap with each box in box2.
    """
    # Unpack box1
    xc1, yc1, xsize1, ysize1 = box1

    # Convert box1 parameters to arrays for broadcasting
    xc1 = np.asarray(xc1)
    yc1 = np.asarray(yc1)
    xsize1 = np.asarray(xsize1)
    ysize1 = np.asarray(ysize1)

    # Ensure box2 is a NumPy array
    box2 = np.asarray(box2)

    # Unpack box2 parameters
    xc2, yc2, xsize2, ysize2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Compute corner coordinates
    ll1, ur1 = bbox_corner_coordsv(xc1, yc1, xsize1, ysize1)
    ll2, ur2 = bbox_corner_coordsv(xc2, yc2, xsize2, ysize2)

    x1_min, y1_max = ll1  # Scalars
    x1_max, y1_min = ur1  # Scalars
    x2_min, y2_max = ll2[:, 0], ll2[:, 1]  # Arrays
    x2_max, y2_min = ur2[:, 0], ur2[:, 1]  # Arrays

    # Check for overlap conditions
    overlaps = ~(
        (x1_max < x2_min) | (x2_max < x1_min) |  # No overlap in x-axis
        (y1_min > y2_max) | (y2_min > y1_max)    # No overlap in y-axis
    )
    return overlaps


def near(k: str, loc: tuple, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER
    if k in REFS_THINGS.keys():
        ref_pt = list(REFS_THINGS[k].pt)
    else:
        ref_pt = OBJ_POS[k]
        if ref_pt[0] is None:
            return False  # ref obj not placed yet
        if do_boxes_overlap(box1=(ref_pt[0], ref_pt[1], ALL_OBJECTS[k]["size"][0], ALL_OBJECTS[k]["size"][1]),
                            box2=(loc[0], loc[1], ALL_OBJECTS[obj]["size"][0], ALL_OBJECTS[obj]["size"][1])):
            return False
    loc = list(loc)
    dist = np.linalg.norm(np.array(ref_pt[:2]) - np.array(loc[:2]))
    if MODE == "learn":
        PARAMS_COUNTER.near += 1
        if PARAMS_COUNTER.near == 1:
            OBJ_PROGS[obj].params.near = dist
        else:
            # OBJ_PROGS[obj].params.near = (OBJ_PROGS[obj].params.near * (PARAMS_COUNTER.near - 1) + dist) / PARAMS_COUNTER.near
            OBJ_PROGS[obj].params.near = max(OBJ_PROGS[obj].params.near, dist)
    return dist <= OBJ_PROGS[obj].params.near + 0.02


def nearv(k: str, loc: np.ndarray, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER
    loc = np.asarray(loc)   # N x 2
    result = np.ones(loc.shape[0], dtype=bool)
    if k in REFS_THINGS.keys():
        ref_pt = list(REFS_THINGS[k].pt)
    else:
        ref_pt = OBJ_POS[k]
        if ref_pt[0] is None:
            return np.zeros(loc.shape[0], dtype=bool)  # ref obj not placed yet
        box1 = (ref_pt[0], ref_pt[1], ALL_OBJECTS[k]["size"][0], ALL_OBJECTS[k]["size"][1])
        obj_size_x = ALL_OBJECTS[obj]["size"][0]
        obj_size_y = ALL_OBJECTS[obj]["size"][1]
        box2_array = np.column_stack((
            loc[:, 0],  # x-coordinates
            loc[:, 1],  # y-coordinates
            np.full(loc.shape[0], obj_size_x),
            np.full(loc.shape[0], obj_size_y)
        ))
        overlaps = do_boxes_overlapv(box1, box2_array)
        result[overlaps] = False
    # Compute distances only for locations where result is still True
    compute_mask = result.copy()
    if np.any(compute_mask):
        locs_to_compute = loc[compute_mask]
        distances = np.linalg.norm(locs_to_compute[:, :2] - ref_pt[:2], axis=1)
        threshold = OBJ_PROGS[obj].params.near + 0.02
        result[compute_mask] = distances <= threshold
    else:
        result[:] = False  # If no locations to compute, set all to False
    return result


def to_right(k: str, loc: tuple, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER
    if k in REFS_THINGS.keys():
        if k in ["left edge", "center"]:
            ref_pt = list(REFS_THINGS[k].pt)
        else:
            # something weird
            print(f"Error: {k} not allowed for to_right")
            ref_pt = list(REFS_THINGS["left edge"].pt)  # default to left edge
    else:
        ref_pt = OBJ_POS[k]
        if ref_pt[0] is None:
            return False  # ref obj not placed yet
        if do_boxes_overlap(box1=(ref_pt[0], ref_pt[1], ALL_OBJECTS[k]["size"][0], ALL_OBJECTS[k]["size"][1]),
                            box2=(loc[0], loc[1], ALL_OBJECTS[obj]["size"][0], ALL_OBJECTS[obj]["size"][1])):
            return False
    loc = list(loc)
    if loc[1] > ref_pt[1]:
        return False
    vert_dist = abs(loc[0] - ref_pt[0])
    if MODE == "learn":
        PARAMS_COUNTER.to_right += 1
        if PARAMS_COUNTER.to_right == 1:
            OBJ_PROGS[obj].params.to_right = vert_dist
        else:
            # OBJ_PROGS[obj].params.to_right = (OBJ_PROGS[obj].params.to_right * (PARAMS_COUNTER.to_right - 1) + vert_dist) / PARAMS_COUNTER.to_right
            OBJ_PROGS[obj].params.to_right = max(OBJ_PROGS[obj].params.to_right, vert_dist)
    return vert_dist <= OBJ_PROGS[obj].params.to_right + 0.02


def to_rightv(k: str, loc: np.ndarray, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER, REFS_THINGS, ALL_OBJECTS
    loc = np.asarray(loc)  # Ensure loc is an N x 2 array
    result = np.ones(loc.shape[0], dtype=bool)

    # Determine the reference point
    if k in REFS_THINGS.keys():
        if k in ["left edge", "center"]:
            ref_pt = np.array(REFS_THINGS[k].pt)
        else:
            print(f"Error: {k} not allowed for to_right")
            ref_pt = np.array(REFS_THINGS["left edge"].pt)  # Default to left edge
    else:
        ref_pt = np.array(OBJ_POS[k])
        if ref_pt[0] is None:
            return np.zeros(loc.shape[0], dtype=bool)  # Reference object not placed yet
        # Check for box overlap
        box1 = (
            ref_pt[0],
            ref_pt[1],
            ALL_OBJECTS[k]["size"][0],
            ALL_OBJECTS[k]["size"][1],
        )
        obj_size_x = ALL_OBJECTS[obj]["size"][0]
        obj_size_y = ALL_OBJECTS[obj]["size"][1]
        box2_array = np.column_stack((
            loc[:, 0],
            loc[:, 1],
            np.full(loc.shape[0], obj_size_x),
            np.full(loc.shape[0], obj_size_y),
        ))
        overlaps = do_boxes_overlapv(box1, box2_array)
        result[overlaps] = False

    # Exclude locations that are above the reference point
    above_ref_mask = loc[:, 1] > ref_pt[1]
    result[above_ref_mask] = False

    # Compute vertical distances for remaining locations
    compute_mask = result.copy()
    if np.any(compute_mask):
        locs_to_compute = loc[compute_mask]
        vert_distances = np.abs(locs_to_compute[:, 0] - ref_pt[0])
        threshold = OBJ_PROGS[obj].params.to_right + 0.02
        result_indices = np.where(compute_mask)[0]
        result[result_indices] = vert_distances <= threshold
    else:
        result[:] = False  # No locations to compute

    return result


def to_left(k: str, loc: tuple, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER
    if k in REFS_THINGS.keys():
        if k in ["right edge", "center"]:
            ref_pt = list(REFS_THINGS[k].pt)
        else:
            # something weird
            print(f"Error: {k} not allowed for to_left")
            ref_pt = list(REFS_THINGS["right edge"].pt)  # default to right edge
    else:
        ref_pt = OBJ_POS[k]
        if ref_pt[0] is None:
            return False  # ref obj not placed yet
        if do_boxes_overlap(box1=(ref_pt[0], ref_pt[1], ALL_OBJECTS[k]["size"][0], ALL_OBJECTS[k]["size"][1]),
                            box2=(loc[0], loc[1], ALL_OBJECTS[obj]["size"][0], ALL_OBJECTS[obj]["size"][1])):
            return False
    loc = list(loc)
    if loc[1] < ref_pt[1]:
        return False
    vert_dist = abs(loc[0] - ref_pt[0])
    if MODE == "learn":
        PARAMS_COUNTER.to_left += 1
        if PARAMS_COUNTER.to_left == 1:
            OBJ_PROGS[obj].params.to_left = vert_dist
        else:
            # OBJ_PROGS[obj].params.to_left = (OBJ_PROGS[obj].params.to_left * (PARAMS_COUNTER.to_left - 1) + vert_dist) / PARAMS_COUNTER.to_left
            OBJ_PROGS[obj].params.to_left = max(OBJ_PROGS[obj].params.to_left, vert_dist)
    return vert_dist <= OBJ_PROGS[obj].params.to_left + 0.02


def to_leftv(k: str, loc: np.ndarray, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER, REFS_THINGS, ALL_OBJECTS
    loc = np.asarray(loc)  # Ensure loc is an N x 2 array
    result = np.ones(loc.shape[0], dtype=bool)

    # Determine the reference point
    if k in REFS_THINGS.keys():
        if k in ["right edge", "center"]:
            ref_pt = np.array(REFS_THINGS[k].pt)
        else:
            print(f"Error: {k} not allowed for to_left")
            ref_pt = np.array(REFS_THINGS["right edge"].pt)  # Default to right edge
    else:
        ref_pt = np.array(OBJ_POS[k])
        if ref_pt[0] is None:
            return np.zeros(loc.shape[0], dtype=bool)  # Reference object not placed yet
        # Check for box overlap
        box1 = (
            ref_pt[0],
            ref_pt[1],
            ALL_OBJECTS[k]["size"][0],
            ALL_OBJECTS[k]["size"][1],
        )
        obj_size_x = ALL_OBJECTS[obj]["size"][0]
        obj_size_y = ALL_OBJECTS[obj]["size"][1]
        box2_array = np.column_stack((
            loc[:, 0],
            loc[:, 1],
            np.full(loc.shape[0], obj_size_x),
            np.full(loc.shape[0], obj_size_y),
        ))
        overlaps = do_boxes_overlapv(box1, box2_array)
        result[overlaps] = False

    # Exclude locations that are below the reference point
    below_ref_mask = loc[:, 1] < ref_pt[1]
    result[below_ref_mask] = False

    # Compute vertical distances for remaining locations
    compute_mask = result.copy()
    if np.any(compute_mask):
        locs_to_compute = loc[compute_mask]
        vert_distances = np.abs(locs_to_compute[:, 0] - ref_pt[0])
        threshold = OBJ_PROGS[obj].params.to_left + 0.02
        result_indices = np.where(compute_mask)[0]
        result[result_indices] = vert_distances <= threshold
    else:
        result[:] = False  # No locations to compute

    return result


def above(k: str, loc: tuple, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER
    if k in REFS_THINGS.keys():
        if k in ["bottom edge", "center"]:
            ref_pt = list(REFS_THINGS[k].pt)
        else:
            # something weird
            print(f"Error: {k} not allowed for above")
            ref_pt = list(REFS_THINGS["bottom edge"].pt)  # default to bottom edge
    else:
        ref_pt = OBJ_POS[k]
        if ref_pt[0] is None:
            return False  # ref obj not placed yet
        if do_boxes_overlap(box1=(ref_pt[0], ref_pt[1], ALL_OBJECTS[k]["size"][0], ALL_OBJECTS[k]["size"][1]),
                            box2=(loc[0], loc[1], ALL_OBJECTS[obj]["size"][0], ALL_OBJECTS[obj]["size"][1])):
            return False
    loc = list(loc)
    if loc[0] < ref_pt[0]:
        return False
    hor_dist = abs(loc[1] - ref_pt[1])
    if MODE == "learn":
        PARAMS_COUNTER.above += 1
        if PARAMS_COUNTER.above == 1:
            OBJ_PROGS[obj].params.above = hor_dist
        else:
            # OBJ_PROGS[obj].params.above = (OBJ_PROGS[obj].params.above * (PARAMS_COUNTER.above - 1) + hor_dist) / PARAMS_COUNTER.above
            OBJ_PROGS[obj].params.above = max(OBJ_PROGS[obj].params.above, hor_dist)
    return hor_dist <= OBJ_PROGS[obj].params.above + 0.02


def below(k: str, loc: tuple, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER
    if k in REFS_THINGS.keys():
        if k in ["top edge", "center"]:
            ref_pt = list(REFS_THINGS[k].pt)
        else:
            # something weird
            print(f"Error: {k} not allowed for below")
            ref_pt = list(REFS_THINGS["top edge"].pt)  # default to top edge
    else:
        ref_pt = OBJ_POS[k]
        if ref_pt[0] is None:
            return False  # ref obj not placed yet
        if do_boxes_overlap(box1=(ref_pt[0], ref_pt[1], ALL_OBJECTS[k]["size"][0], ALL_OBJECTS[k]["size"][1]),
                            box2=(loc[0], loc[1], ALL_OBJECTS[obj]["size"][0], ALL_OBJECTS[obj]["size"][1])):
            return False
    loc = list(loc)
    if loc[0] > ref_pt[0]:
        return False
    hor_dist = abs(loc[1] - ref_pt[1])
    if MODE == "learn":
        PARAMS_COUNTER.below += 1
        if PARAMS_COUNTER.below == 1:
            OBJ_PROGS[obj].params.below = hor_dist
        else:
            # OBJ_PROGS[obj].params.below = (OBJ_PROGS[obj].params.below * (PARAMS_COUNTER.below - 1) + hor_dist) / PARAMS_COUNTER.below
            OBJ_PROGS[obj].params.below = max(OBJ_PROGS[obj].params.below, hor_dist)
    return hor_dist <= OBJ_PROGS[obj].params.below + 0.02


def abovev(k: str, loc: np.ndarray, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, REFS_THINGS, ALL_OBJECTS
    loc = np.asarray(loc)  # Ensure loc is an N x 2 array
    N = loc.shape[0]
    result = np.ones(N, dtype=bool)

    # Determine the reference point
    if k in REFS_THINGS.keys():
        if k in ["bottom edge", "center"]:
            ref_pt = np.array(REFS_THINGS[k].pt)
        else:
            print(f"Error: {k} not allowed for above")
            ref_pt = np.array(REFS_THINGS["bottom edge"].pt)  # Default to bottom edge
    else:
        ref_pt = np.array(OBJ_POS[k])
        if ref_pt[0] is None:
            return np.zeros(N, dtype=bool)  # Reference object not placed yet
        # Check for box overlap
        box1 = (
            ref_pt[0],
            ref_pt[1],
            ALL_OBJECTS[k]["size"][0],
            ALL_OBJECTS[k]["size"][1],
        )
        obj_size_x = ALL_OBJECTS[obj]["size"][0]
        obj_size_y = ALL_OBJECTS[obj]["size"][1]
        box2_array = np.column_stack((
            loc[:, 0],
            loc[:, 1],
            np.full(N, obj_size_x),
            np.full(N, obj_size_y),
        ))
        overlaps = do_boxes_overlapv(box1, box2_array)
        result[overlaps] = False

    # Exclude locations that are below the reference point
    below_ref_mask = loc[:, 0] < ref_pt[0]
    result[below_ref_mask] = False

    # Compute horizontal distances for remaining locations
    compute_mask = result.copy()
    if np.any(compute_mask):
        locs_to_compute = loc[compute_mask]
        hor_distances = np.abs(locs_to_compute[:, 1] - ref_pt[1])
        threshold = OBJ_PROGS[obj].params.above + 0.02
        result_indices = np.where(compute_mask)[0]
        result[result_indices] = hor_distances <= threshold
    else:
        result[:] = False  # No locations to compute

    return result


def belowv(k: str, loc: np.ndarray, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, REFS_THINGS, ALL_OBJECTS
    loc = np.asarray(loc)  # Ensure loc is an N x 2 array
    N = loc.shape[0]
    result = np.ones(N, dtype=bool)

    # Determine the reference point
    if k in REFS_THINGS.keys():
        if k in ["top edge", "center"]:
            ref_pt = np.array(REFS_THINGS[k].pt)
        else:
            print(f"Error: {k} not allowed for below")
            ref_pt = np.array(REFS_THINGS["top edge"].pt)  # Default to top edge
    else:
        ref_pt = np.array(OBJ_POS[k])
        if ref_pt[0] is None:
            return np.zeros(N, dtype=bool)  # Reference object not placed yet
        # Check for box overlap
        box1 = (
            ref_pt[0],
            ref_pt[1],
            ALL_OBJECTS[k]["size"][0],
            ALL_OBJECTS[k]["size"][1],
        )
        obj_size_x = ALL_OBJECTS[obj]["size"][0]
        obj_size_y = ALL_OBJECTS[obj]["size"][1]
        box2_array = np.column_stack((
            loc[:, 0],
            loc[:, 1],
            np.full(N, obj_size_x),
            np.full(N, obj_size_y),
        ))
        overlaps = do_boxes_overlapv(box1, box2_array)
        result[overlaps] = False

    # Exclude locations that are above the reference point
    above_ref_mask = loc[:, 0] > ref_pt[0]
    result[above_ref_mask] = False

    # Compute horizontal distances for remaining locations
    compute_mask = result.copy()
    if np.any(compute_mask):
        locs_to_compute = loc[compute_mask]
        hor_distances = np.abs(locs_to_compute[:, 1] - ref_pt[1])
        threshold = OBJ_PROGS[obj].params.below + 0.02
        result_indices = np.where(compute_mask)[0]
        result[result_indices] = hor_distances <= threshold
    else:
        result[:] = False  # No locations to compute

    return result


def on_top(k: str, loc: tuple, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER
    if k in REFS_THINGS.keys():
        if k in ["table", "center"]:
            ref_pt = list(REFS_THINGS[k].pt)
        else:
            # something weird
            print(f"Error: {k} not allowed for on_top")
            ref_pt = list(REFS_THINGS["table"].pt)  # default to table
    else:
        ref_pt = OBJ_POS[k]
        if ref_pt[0] is None:
            return False  # ref obj not placed yet
    loc = list(loc)
    ref_height = ALL_OBJECTS[k]["size"][2]
    top_z = ref_pt[2] + 0.6 * ref_height
    x_diff = abs(loc[0] - ref_pt[0])
    y_diff = abs(loc[1] - ref_pt[1])
    z_diff = abs(loc[2] - top_z)
    if MODE == "learn":
        PARAMS_COUNTER.on_top += 1
        if PARAMS_COUNTER.on_top == 1:
            OBJ_PROGS[obj].params.on_top = [x_diff, y_diff, z_diff]
        else:
            # OBJ_PROGS[obj].params.on_top = [(OBJ_PROGS[obj].params.on_top[0] * (PARAMS_COUNTER.on_top - 1) + x_diff) / PARAMS_COUNTER.on_top,
            #                                 (OBJ_PROGS[obj].params.on_top[1] * (PARAMS_COUNTER.on_top - 1) + y_diff) / PARAMS_COUNTER.on_top,
            #                                 (OBJ_PROGS[obj].params.on_top[2] * (PARAMS_COUNTER.on_top - 1) + z_diff) / PARAMS_COUNTER.on_top]
            OBJ_PROGS[obj].params.on_top = [max(OBJ_PROGS[obj].params.on_top[0], x_diff),
                                            max(OBJ_PROGS[obj].params.on_top[1], y_diff),
                                            max(OBJ_PROGS[obj].params.on_top[2], z_diff)]
    return x_diff <= OBJ_PROGS[obj].params.on_top[0] + 0.02 and y_diff <= OBJ_PROGS[obj].params.on_top[1] + 0.02 and z_diff <= OBJ_PROGS[obj].params.on_top[2] + 0.02


def inside(k: str, loc: tuple, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER
    if k in REFS_THINGS.keys():
        # something weird
        print(f"Error: {k} not allowed for inside")
        ref_pt = list(REFS_THINGS["table"].pt)  # default to table
    else:
        ref_pt = OBJ_POS[k]
        if ref_pt[0] is None:
            return False  # ref obj not placed yet
    loc = list(loc)
    dist = np.linalg.norm(np.array(ref_pt) - np.array(loc))
    if MODE == "learn":
        PARAMS_COUNTER.inside += 1
        if PARAMS_COUNTER.inside == 1:
            OBJ_PROGS[obj].params.inside = dist
        else:
            # OBJ_PROGS[obj].params.inside = (OBJ_PROGS[obj].params.inside * (PARAMS_COUNTER.inside - 1) + dist) / PARAMS_COUNTER.inside
            OBJ_PROGS[obj].params.inside = max(OBJ_PROGS[obj].params.inside, dist)
    return dist <= OBJ_PROGS[obj].params.inside + 0.02


def under(k: str, loc: tuple, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER
    if k in REFS_THINGS.keys():
        # something weird
        print(f"Error: {k} not allowed for under")
        ref_pt = list(REFS_THINGS["table"].pt)  # default to table
    else:
        ref_pt = OBJ_POS[k]
        if ref_pt[0] is None:
            return False  # ref obj not placed yet
    loc = list(loc)
    ref_height = ALL_OBJECTS[k]["size"][2]
    bot_z = ref_pt[2] - 0.6 * ref_height
    x_diff = abs(loc[0] - ref_pt[0])
    y_diff = abs(loc[1] - ref_pt[1])
    z_diff = abs(loc[2] - bot_z)
    if MODE == "learn":
        PARAMS_COUNTER.under += 1
        if PARAMS_COUNTER.under == 1:
            OBJ_PROGS[obj].params.under = [x_diff, y_diff, z_diff]
        else:
            # OBJ_PROGS[obj].params.under = [(OBJ_PROGS[obj].params.under[0] * (PARAMS_COUNTER.under - 1) + x_diff) / PARAMS_COUNTER.under,
            #                                (OBJ_PROGS[obj].params.under[1] * (PARAMS_COUNTER.under - 1) + y_diff) / PARAMS_COUNTER.under,
            #                                (OBJ_PROGS[obj].params.under[2] * (PARAMS_COUNTER.under - 1) + z_diff) / PARAMS_COUNTER.under]
            OBJ_PROGS[obj].params.under = [max(OBJ_PROGS[obj].params.under[0], x_diff),
                                           max(OBJ_PROGS[obj].params.under[1], y_diff),
                                           max(OBJ_PROGS[obj].params.under[2], z_diff)]
    return x_diff <= OBJ_PROGS[obj].params.under[0] + 0.02 and y_diff <= OBJ_PROGS[obj].params.under[1] + 0.02 and z_diff <= OBJ_PROGS[obj].params.under[2] + 0.02


def on_topv(k: str, loc: np.ndarray, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, REFS_THINGS, ALL_OBJECTS
    loc = np.asarray(loc)  # Ensure loc is an N x 3 array
    N = loc.shape[0]
    result = np.ones(N, dtype=bool)

    # Determine the reference point
    if k in REFS_THINGS.keys():
        if k in ["table", "center"]:
            ref_pt = np.array(REFS_THINGS[k].pt)
        else:
            print(f"Error: {k} not allowed for on_top")
            ref_pt = np.array(REFS_THINGS["table"].pt)  # Default to table
    else:
        ref_pt = np.array(OBJ_POS[k])
        if ref_pt[0] is None:
            return np.zeros(N, dtype=bool)  # Reference object not placed yet

    ref_height = ALL_OBJECTS[k]["size"][2]
    top_z = ref_pt[2] + 0.6 * ref_height

    # Compute differences
    x_diff = np.abs(loc[:, 0] - ref_pt[0])
    y_diff = np.abs(loc[:, 1] - ref_pt[1])
    z_diff = np.abs(loc[:, 2] - top_z)

    # Thresholds from OBJ_PROGS
    x_thresh = OBJ_PROGS[obj].params.on_top[0] + 0.02
    y_thresh = OBJ_PROGS[obj].params.on_top[1] + 0.02
    z_thresh = OBJ_PROGS[obj].params.on_top[2] + 0.02

    # Check if differences are within thresholds
    within_x = x_diff <= x_thresh
    within_y = y_diff <= y_thresh
    within_z = z_diff <= z_thresh

    # Combine the conditions
    result = within_x & within_y & within_z

    return result


def insidev(k: str, loc: np.ndarray, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, REFS_THINGS, ALL_OBJECTS
    loc = np.asarray(loc)  # Ensure loc is an N x 3 array
    N = loc.shape[0]
    result = np.ones(N, dtype=bool)

    if k in REFS_THINGS.keys():
        print(f"Error: {k} not allowed for inside")
        ref_pt = np.array(REFS_THINGS["table"].pt)  # Default to table
    else:
        ref_pt = np.array(OBJ_POS[k])
        if ref_pt[0] is None:
            return np.zeros(N, dtype=bool)  # Reference object not placed yet

    # Compute distances
    distances = np.linalg.norm(loc - ref_pt, axis=1)

    # Threshold from OBJ_PROGS
    threshold = OBJ_PROGS[obj].params.inside + 0.02

    # Check if distances are within threshold
    result = distances <= threshold

    return result


def underv(k: str, loc: np.ndarray, obj: str):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, REFS_THINGS, ALL_OBJECTS
    loc = np.asarray(loc)  # Ensure loc is an N x 3 array
    N = loc.shape[0]
    result = np.ones(N, dtype=bool)

    if k in REFS_THINGS.keys():
        print(f"Error: {k} not allowed for under")
        ref_pt = np.array(REFS_THINGS["table"].pt)  # Default to table
    else:
        ref_pt = np.array(OBJ_POS[k])
        if ref_pt[0] is None:
            return np.zeros(N, dtype=bool)  # Reference object not placed yet

    ref_height = ALL_OBJECTS[k]["size"][2]
    bot_z = ref_pt[2] - 0.6 * ref_height

    # Compute differences
    x_diff = np.abs(loc[:, 0] - ref_pt[0])
    y_diff = np.abs(loc[:, 1] - ref_pt[1])
    z_diff = np.abs(loc[:, 2] - bot_z)

    # Thresholds from OBJ_PROGS
    x_thresh = OBJ_PROGS[obj].params.under[0] + 0.02
    y_thresh = OBJ_PROGS[obj].params.under[1] + 0.02
    z_thresh = OBJ_PROGS[obj].params.under[2] + 0.02

    # Check if differences are within thresholds
    within_x = x_diff <= x_thresh
    within_y = y_diff <= y_thresh
    within_z = z_diff <= z_thresh

    # Combine the conditions
    result = within_x & within_y & within_z

    return result


def current_overlap(obj1: str, obj2: str, loc1=None, loc2=None):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER
    loc1 = OBJ_POS[obj1] if loc1 is None else loc1
    loc2 = OBJ_POS[obj2] if loc2 is None else loc2
    size1 = ALL_OBJECTS[obj1]["size"]
    size2 = ALL_OBJECTS[obj2]["size"]
    if loc1[0] is None or loc2[0] is None:  # one of the objects not placed yet
        return False
    return do_boxes_overlap(box1=(loc1[0], loc1[1], size1[0], size1[1]),
                            box2=(loc2[0], loc2[1], size2[0], size2[1]))


def _task_infer(tobj, xmin, xmax, xstep, ymin, ymax, ystep):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER
    tobj_prog = OBJ_PROGS[tobj].prog
    tobj_prog = transform_code(tobj_prog, "loc", "obj")
    exec(tobj_prog, globals())
    tobj_valid_pos_list = []
    for _x in tqdm(range(xmin, xmax, xstep), desc="x", leave=False):
        for _y in tqdm(range(ymin, ymax, ystep), desc="y", leave=False):
            for _z in tqdm(range(int(ALL_OBJECTS[tobj].z_range[0] * 100), int(ALL_OBJECTS[tobj].z_range[1] * 100 + 1)), desc="z", leave=False):
                ret_val = program(loc=(_x / 100, _y / 100, _z / 100), obj=tobj)
                if ret_val:
                    tobj_valid_pos_list.append([_x / 100, _y / 100, _z / 100])
    random.shuffle(tobj_valid_pos_list)
    return tobj_valid_pos_list


def generate_loc_generator(xmin, xmax, ymin, ymax, zmin, zmax, xstep, ystep, zstep):
    x_values = np.arange(xmin, xmax + xstep, xstep) / 100
    y_values = np.arange(ymin, ymax + ystep, ystep) / 100
    z_values = np.arange(zmin, zmax + zstep, zstep) / 100
    return itertools.product(x_values, y_values, z_values)


def _task_inferv(tobj, xmin, xmax, xstep, ymin, ymax, ystep):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER
    tobj_prog = OBJ_PROGS[tobj].prog
    tobj_prog = transform_code(tobj_prog, "loc", "obj")
    locs_array = np.asarray(list(generate_loc_generator(xmin, xmax, ymin, ymax, int(ALL_OBJECTS[tobj].z_range[0] * 100), int(ALL_OBJECTS[tobj].z_range[1] * 100 + 1), xstep, ystep, 1)))
    conditions_prog = extract_conditions_from_prog(tobj_prog)
    finresult = np.zeros((locs_array.shape[0],), dtype=bool)
    for cond in conditions_prog:
        cond = cond.replace(' and ', ' & ').replace(' or ', ' | ').replace("(", "v(")
        temp_cond_fnstr = f"def _COND(loc, obj):\n    return {cond}"
        exec(temp_cond_fnstr, globals())
        cond_val = _COND(loc=locs_array, obj=tobj)
        finresult |= cond_val
    tobj_valid_pos_list = locs_array[finresult]
    random.shuffle(tobj_valid_pos_list)
    return tobj_valid_pos_list


def infer_main(task, user_full_learned_prog_data, save_dir="src/llmgrop/dump", N_sample=40, step=1):
    # if ret_val and not current_overlap(tobj1, tobj2, tobj1_pos, (_x / 100, _y / 100, _z / 100)):
    # if ret_val and not current_overlap(tobj1, tobj3, tobj1_pos, (_x / 100, _y / 100, _z / 100)) and not current_overlap(tobj2, tobj3, tobj2_pos, (_x / 100, _y / 100, _z / 100)):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER
    _xmin, _xmax, _xstep, _ymin, _ymax, _ystep = 78, 100, step, -20, 20, step
    PARAMS_COUNTER = None
    with open(user_full_learned_prog_data, "r") as f:
        OBJ_PROGS = json.load(f)
    OBJ_PROGS = edict(OBJ_PROGS)
    TASK = task
    TASK_OBJECTS = OBJ_PROGS.tasks[TASK].order  # IMPORTANT: ordered
    OBJ_POS = edict({
        obj: [None, None, None] for obj in ALL_OBJECTS.keys()  # None = unplaced
    })
    OBJ_ORIE = edict({
        obj: [None, None, None] for obj in ALL_OBJECTS.keys()  # None = unplaced
    })
    task_valid_pos_list = []

    tobj1 = TASK_OBJECTS[0]
    tobj1_valid_pos_list = _task_inferv(tobj1, _xmin, _xmax, _xstep, _ymin, _ymax, _ystep)
    outer_tqdm = tqdm(tobj1_valid_pos_list, desc="tobj1", leave=False)
    for tobj1_pos in outer_tqdm:
        cur_len = len(task_valid_pos_list)
        outer_tqdm.set_postfix({"found": str(cur_len if cur_len < 1000 else f"{cur_len/1000:.2f}k")})
        # if cur_len >= 10 * N_sample:
        #     break
        obj1_tally = 0
        if obj1_tally > N_sample // 3:
            continue
        OBJ_POS[tobj1] = tobj1_pos
        tobj2 = TASK_OBJECTS[1]
        tobj2_valid_pos_list = _task_inferv(tobj2, _xmin, _xmax, _xstep, _ymin, _ymax, _ystep)
        for tobj2_pos in tqdm(tobj2_valid_pos_list, desc="tobj2", leave=False):
            obj2_tally = 0
            if obj1_tally > N_sample // 3 or obj2_tally > N_sample // 3:
                continue
            OBJ_POS[tobj2] = tobj2_pos
            tobj3 = TASK_OBJECTS[2]
            tobj3_valid_pos_list = _task_inferv(tobj3, _xmin, _xmax, _xstep, _ymin, _ymax, _ystep)
            for tobj3_pos in tqdm(tobj3_valid_pos_list, desc="tobj3", leave=False):
                obj3_tally = 0
                if obj1_tally > N_sample // 3 or obj2_tally > N_sample // 3 or obj3_tally > N_sample // 3:
                    continue
                OBJ_POS[tobj3] = tobj3_pos
                if len(TASK_OBJECTS) == 3:
                    obj1_tally += 1
                    obj2_tally += 1
                    obj3_tally += 1
                    task_valid_pos_list.append([tobj1_pos, tobj2_pos, tobj3_pos])
                else:
                    tobj4 = TASK_OBJECTS[3]
                    tobj4_valid_pos_list = _task_inferv(tobj4, _xmin, _xmax, _xstep, _ymin, _ymax, _ystep)
                    for tobj4_pos in tqdm(tobj4_valid_pos_list, desc="tobj4", leave=False):
                        obj4_tally = 0
                        if obj1_tally > N_sample // 3 or obj2_tally > N_sample // 3 or obj3_tally > N_sample // 3 or obj4_tally > N_sample // 3:
                            continue
                        OBJ_POS[tobj4] = tobj4_pos
                        if len(TASK_OBJECTS) == 4:
                            obj1_tally += 1
                            obj2_tally += 1
                            obj3_tally += 1
                            obj4_tally += 1
                            task_valid_pos_list.append([tobj1_pos, tobj2_pos, tobj3_pos, tobj4_pos])
                        else:
                            tobj5 = TASK_OBJECTS[4]
                            tobj5_valid_pos_list = _task_inferv(tobj5, _xmin, _xmax, _xstep, _ymin, _ymax, _ystep)
                            for tobj5_pos in tqdm(tobj5_valid_pos_list, desc="tobj5", leave=False):
                                obj5_tally = 0
                                if obj1_tally > N_sample // 3 or obj2_tally > N_sample // 3 or obj3_tally > N_sample // 3 or obj4_tally > N_sample // 3 or obj5_tally > N_sample // 3:
                                    continue
                                OBJ_POS[tobj5] = tobj5_pos
                                if len(TASK_OBJECTS) == 5:
                                    obj1_tally += 1
                                    obj2_tally += 1
                                    obj3_tally += 1
                                    obj4_tally += 1
                                    obj5_tally += 1
                                    task_valid_pos_list.append([tobj1_pos, tobj2_pos, tobj3_pos, tobj4_pos, tobj5_pos])
                                else:
                                    raise ValueError("More than 5 objects not supported")
    # show
    print(f"\n{len(task_valid_pos_list)} valid positions")
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    if len(task_valid_pos_list) < N_sample:
        N_sample = len(task_valid_pos_list)
    task_valid_pos_list = random.sample(task_valid_pos_list, N_sample)
    final_ori = []
    for obj in TASK_OBJECTS:
        final_ori.append([OBJ_PROGS[obj].orie_roll, OBJ_PROGS[obj].orie_pitch, OBJ_PROGS[obj].orie_yaw])
    for ipos, final_pos in enumerate(tqdm(task_valid_pos_list, desc="saving", leave=False)):
        rgb, *_ = Sim.quick_simulate(objects=TASK_OBJECTS, final_positions=final_pos, final_orie_eulers=final_ori)
        plt.imsave(os.path.join(save_dir, f"{ipos:06}.png"), rgb)


def learn_main(user_raw_data_path, user_prog_data_path, full_learned_store_prog_data_path):
    global TASK, TASK_OBJECTS, OBJ_POS, OBJ_ORIE, OBJ_PROGS, MODE, PARAMS_COUNTER

    with open(user_raw_data_path, "r") as f:
        raw_data = json.load(f)
    raw_data = edict(raw_data)
    with open(user_prog_data_path, "r") as f:
        OBJ_PROGS = json.load(f)
    OBJ_PROGS = edict(OBJ_PROGS)

    # learning parameters of the object programs
    for task in tqdm(LEARN_TASKS.keys(), desc="learn tasks", leave=False):
        TASK = task
        TASK_OBJECTS = OBJ_PROGS.tasks[task].order  # IMPORTANT: ordered
        OBJ_POS = edict({
            obj: [None, None, None] for obj in ALL_OBJECTS.keys()  # None = unplaced
        })
        OBJ_ORIE = edict({
            obj: [None, None, None] for obj in ALL_OBJECTS.keys()  # None = unplaced
        })
        task_data = raw_data[task]
        for tobj in TASK_OBJECTS:
            PARAMS_COUNTER = edict({
                param: 0 for param in PARAMS_DEFAULTS.keys()
            })
            tobj_data = task_data[tobj]
            tobj_prog = OBJ_PROGS[tobj].prog
            tobj_prog = transform_code(tobj_prog, "loc", "obj")
            exec(tobj_prog, globals())
            try:
                ret_val = program(loc=(tobj_data.x_slider, tobj_data.y_slider, tobj_data.z_slider), obj=tobj)  # TODO: wrap it inside a while ret_val
            except Exception as e:
                print(f"\nError: {e}")
            # place the object
            OBJ_POS[tobj] = [tobj_data.x_slider, tobj_data.y_slider, tobj_data.z_slider]
            OBJ_ORIE[tobj] = [tobj_data.roll_slider, tobj_data.pitch_slider, tobj_data.yaw_slider]

    # if params None for objects, then use default values
    for obj in tqdm(ALL_OBJECTS.keys(), desc="default params", leave=False):
        for fn in PARAMS_DEFAULTS.keys():
            if fn in ["on_top", "under"]:
                if OBJ_PROGS[obj].params[fn][0] is None:
                    OBJ_PROGS[obj].params[fn] = PARAMS_DEFAULTS[fn]
            else:
                if OBJ_PROGS[obj].params[fn] is None:
                    OBJ_PROGS[obj].params[fn] = PARAMS_DEFAULTS[fn]

    # store
    with open(full_learned_store_prog_data_path, "w") as f:
        json.dump(OBJ_PROGS, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="learn or infer")
    parser.add_argument("--task", type=int, default=1, help="task number: 1 to 8")
    parser.add_argument("--step", type=int, default=1, help="step size")
    parser.add_argument("--unum", type=int, required=True, help="user num: 0 to 10, -1 for all")
    args = parser.parse_args()

    MODE = args.mode
    user_demo_data_dir = "test/llmgrop/USER_DEMO_DATA"
    generations_dir = "test/llmgrop/GENERATIONS"
    all_json_files = os.listdir(user_demo_data_dir)
    all_json_files = [f for f in all_json_files if f.endswith(".json")]
    all_names = sorted([f.split(".")[0] for f in all_json_files])

    # Parameter Synthesis
    if MODE == "learn":
        if args.unum == -1:
            for user_name in tqdm(all_names, desc="users"):
                TASK = None
                TASK_OBJECTS = None
                OBJ_POS = None
                OBJ_ORIE = None
                OBJ_PROGS = None
                PARAMS_COUNTER = None
                user_raw_data_path = f"{user_demo_data_dir}/{user_name}.json"
                user_prog_data_path = f"{user_demo_data_dir}/prog_learned/learned_{user_name}.json"
                full_learned_store_prog_data_path = f"{user_demo_data_dir}/full_learned/full_learned_{user_name}.json"
                learn_main(user_raw_data_path, user_prog_data_path, full_learned_store_prog_data_path)
        else:
            TASK = None
            TASK_OBJECTS = None
            OBJ_POS = None
            OBJ_ORIE = None
            OBJ_PROGS = None
            PARAMS_COUNTER = None
            user_name = all_names[args.unum]
            user_raw_data_path = f"{user_demo_data_dir}/{user_name}.json"
            user_prog_data_path = f"{user_demo_data_dir}/prog_learned/learned_{user_name}.json"
            full_learned_store_prog_data_path = f"{user_demo_data_dir}/full_learned/full_learned_{user_name}.json"
            learn_main(user_raw_data_path, user_prog_data_path, full_learned_store_prog_data_path)
    # Sample generations
    elif MODE == "infer":
        NSAMPLE = 200
        if args.unum == -1:
            for user_name in tqdm(all_names, desc="users"):
                TASK = None
                TASK_OBJECTS = None
                OBJ_POS = None
                OBJ_ORIE = None
                OBJ_PROGS = None
                PARAMS_COUNTER = None
                task = f"Task {args.task}"
                infer_main(task=task,
                           user_full_learned_prog_data=f"{user_demo_data_dir}/full_learned/full_learned_{user_name}.json",
                           save_dir=f"{generations_dir}/task{args.task}/Synapse/{user_name}",
                           N_sample=NSAMPLE,
                           step=args.step)
        else:
            TASK = None
            TASK_OBJECTS = None
            OBJ_POS = None
            OBJ_ORIE = None
            OBJ_PROGS = None
            PARAMS_COUNTER = None
            user_name = all_names[args.unum]
            task = f"Task {args.task}"
            infer_main(task=task,
                       user_full_learned_prog_data=f"{user_demo_data_dir}/full_learned/full_learned_{user_name}.json",
                       save_dir=f"{generations_dir}/task{args.task}/Synapse/{user_name}",
                       N_sample=NSAMPLE,
                       step=args.step)
