from easydict import EasyDict as edict

ALL_OBJECTS = edict({
    'dinner plate': {"size": (0.16, 0.16, 0.03), "z_range": (0.64, 0.64)},      # 1, 5, 6, 7, 8
    'dinner fork': {"size": (0.2, 0.05, 0.03), "z_range": (0.64, 0.64)},        # 1, 6, 7, 8
    'dinner knife': {"size": (0.22, 0.05, 0.03), "z_range": (0.64, 0.64)},      # 1, 7, 8
    'bread plate': {"size": (0.14, 0.14, 0.03), "z_range": (0.64, 0.64)},       # 2, 3
    'water cup': {"size": (0.09, 0.09, 0.13), "z_range": (0.64, 0.70)},         # 2
    'bread': {"size": (0.1, 0.1, 0.06), "z_range": (0.62, 0.67)},               # 2
    'teacup': {"size": (0.15, 0.1, 0.1), "z_range": (0.65, 0.70)},              # 3, 4, 5, 6, 8
    'place mat': {"size": (0.12, 0.12, 0.03), "z_range": (0.62, 0.66)},         # 3
    'fruit bowl': {"size": (0.15, 0.15, 0.07), "z_range": (0.62, 0.69)},        # 4
    'strawberry': {"size": (0.1, 0.1, 0.05), "z_range": (0.64, 0.69)},  # * 4, 7
    'teacup lid': {"size": (0.06, 0.06, 0.03), "z_range": (0.64, 0.73)},        # 5, 6, 8
})

ALL_TASKS = edict({
    "Task 1": {  # Dinner Plate, Dinner Fork, Dinner Knife
        "objects": ['dinner plate', 'dinner fork', 'dinner knife'],
    },
    "Task 2": {  # Bread Plate, Water Cup, Bread
        "objects": ['bread plate', 'water cup', 'bread'],
    },
    "Task 3": {  # Mug, Bread Plate, Mug Mat
        "objects": ['teacup', 'bread plate', 'place mat'],
    },
    "Task 4": {  # Fruit Bowl, Mug, Strawberry
        "objects": ['fruit bowl', 'teacup', 'strawberry'],
    },
    "Task 5": {  # Mug, Dinner plate, Mug Lid
        "objects": ['teacup', 'dinner plate', 'teacup lid'],
    },
    "Task 6": {  # Dinner Plate, Dinner Fork, Mug, Mug Lid
        "objects": ['dinner plate', 'dinner fork', 'teacup', 'teacup lid'],
    },
    "Task 7": {  # Dinner Plate, Dinner Fork, Dinner Knife, Strawberry
        "objects": ['dinner plate', 'dinner fork', 'dinner knife', 'strawberry'],
    },
    "Task 8": {  # Dinner Plate, Dinner Fork, Dinner Knife, Mug, Mug Lid
        "objects": ['dinner plate', 'dinner fork', 'dinner knife', 'teacup', 'teacup lid'],
    },
})

LEARN_TASKS = edict({
    "Task 1": {  # Dinner Plate, Dinner Fork, Dinner Knife
        "objects": ['dinner plate', 'dinner fork', 'dinner knife'],
    },
    "Task 2": {  # Bread Plate, Water Cup, Bread
        "objects": ['bread plate', 'water cup', 'bread'],
    },
    "Task 3": {  # Mug, Bread Plate, Mug Mat
        "objects": ['teacup', 'bread plate', 'place mat'],
    },
    "Task 4": {  # Fruit Bowl, Mug, Strawberry
        "objects": ['fruit bowl', 'teacup', 'strawberry'],
    },
    "Task 5": {  # Mug, Dinner plate, Mug Lid
        "objects": ['teacup', 'dinner plate', 'teacup lid'],
    },
})

INFER_TASKS = edict({
    "Task 6": {  # Dinner Plate, Dinner Fork, Mug, Mug Lid
        "objects": ['dinner plate', 'dinner fork', 'teacup', 'teacup lid'],
        "comes_from": ['Task 1', 'Task 5'],
    },
    "Task 7": {  # Dinner Plate, Dinner Fork, Dinner Knife, Strawberry
        "objects": ['dinner plate', 'dinner fork', 'dinner knife', 'strawberry'],
        "comes_from": ['Task 1', 'Task 4'],
    },
    "Task 8": {  # Dinner Plate, Dinner Fork, Dinner Knife, Mug, Mug Lid
        "objects": ['dinner plate', 'dinner fork', 'dinner knife', 'teacup', 'teacup lid'],
        "comes_from": ['Task 1', 'Task 5'],
    },
})

REFS_THINGS = edict({
    'table': {"pt": (0.9, 0.0, 0.65)},
    'bottom edge': {"pt": (0.749, 0.0, 0.65)},
    'top edge': {"pt": (1.251, 0.0, 0.65)},
    'left edge': {"pt": (1.0, 0.251, 0.65)},
    'right edge': {"pt": (1.0, -0.251, 0.65)},
    'center': {"pt": (1.0, 0.0, 0.65)},
})

PARAMS_DEFAULTS = edict({
    "near": 0.2,
    "to_right": 0.2,
    "to_left": 0.2,
    "above": 0.2,
    "below": 0.2,
    "on_top": [0.06, 0.06, 0.06],
    "inside": 0.08,
    "under": [0.06, 0.06, 0.06],
})
