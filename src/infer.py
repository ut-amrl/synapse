import os
import numpy as np
import cv2
import sys
from copy import deepcopy
from simple_colors import red, green
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.ALL_TERRAINS_MAPS import NSLABELS_TWOWAY_NSINT, DATASETINTstr_TO_DATASETLABELS, DATASETLABELS_TO_NSLABELS, NSLABELS_TRAVERSABLE_TERRAINS, NSLABELS_NON_TRAVERSABLE_TERRAINS
from src.backend.terrain_infer import TerrainSegFormer
from src.backend.utils.std_utils import json_reader
from src.datagen import NSInferObjDet, NSInferTerrainSeg


class FasterImageInference:
    WAY_PARAM = 1.7
    SLOPE_SCALE = 0.0
    OBJECT_THRESHOLDS = {  # 3-tuple of (box_threshold, text_threshold, nms_threshold)
        "barricade": (0.5, 0.5, 0.3),
        "board": (0.3, 0.3, 0.5),
        "bush": (0.4, 0.4, 0.4),
        "car": (0.3, 0.3, 0.3),
        "entrance": (0.3, 0.3, 0.2),
        "person": (0.25, 0.25, 0.6),
        "pole": (0.4, 0.4, 0.5),
        "staircase": (0.25, 0.25, 0.4),
        "tree": (0.4, 0.4, 0.45),
        "wall": (0.5, 0.5, 0.4)
    }

    def __init__(self, domain):
        print("Initializing FasterImageInference...")
        self.ns_infer_objdet = NSInferObjDet()
        self.ns_infer_terrainseg = NSInferTerrainSeg()
        self.domain = domain
        self.predefined_terrains = NSLABELS_TWOWAY_NSINT
        self.predefined_terrains["dunno"] = 1111
        self.predefined_terrains[1111] = "dunno"
        self.cur_noext_name = None
        self.cur_img_bgr = None
        self.cur_pc_xyz = None
        self.cur_main_terrain_output = None
        self.cur_main_in_the_way_output = None
        self.cur_main_slope_output = None
        self.cur_distance_to_output = None  # dict of objects to outputs
        self.cur_frontal_distance_output = None  # dict of objects to outputs

    def set_state(self, noext_name, img_bgr, pc_xyz):
        self.noext_name = noext_name
        self.cur_img_bgr = img_bgr
        self.cur_pc_xyz = pc_xyz
        self.cur_main_terrain_output = None
        self.cur_main_in_the_way_output = None
        self.cur_main_slope_output = None
        self.cur_distance_to_output = None  # dict of objects to outputs
        self.cur_frontal_distance_output = None  # dict of objects to outputs

    def __getattr__(self, name):
        if name.startswith("_distance_to_"):
            def dynamic_method(arg):
                target = name[len("_distance_to_"):]
                return self._distance_to(arg, target)
            return dynamic_method
        elif name.startswith("_frontal_distance_"):
            def dynamic_method(arg):
                target = name[len("_frontal_distance_"):]
                return self._frontal_distance(arg, target)
            return dynamic_method
        raise AttributeError(f"{name} not found")

    def _terrain(self, pixel_loc):
        """
        Returns the terrain idx at the given pixel location
        """
        if self.cur_main_terrain_output is None:
            self.cur_main_terrain_output = self.terrain(self.cur_img_bgr, self.cur_pc_xyz)
        return int(self.cur_main_terrain_output[pixel_loc[1], pixel_loc[0]])

    def _in_the_way(self, pixel_loc):
        """
        Returns whether the given pixel location is in the way
        """
        if self.cur_main_in_the_way_output is None:
            self.cur_main_in_the_way_output = self.in_the_way(self.cur_img_bgr, self.cur_pc_xyz)
        return bool(self.cur_main_in_the_way_output[pixel_loc[1], pixel_loc[0]])

    def _slope(self, pixel_loc):
        """
        Returns the slope at the given pixel location
        """
        if self.cur_main_slope_output is None:
            self.cur_main_slope_output = self.slope(self.cur_img_bgr, self.cur_pc_xyz)
        return float(self.cur_main_slope_output[pixel_loc[1], pixel_loc[0]])

    def _distance_to(self, pixel_loc, obj_name):
        """
        Returns the distance to the given object at the given pixel location
        """
        if self.cur_distance_to_output is None:
            self.cur_distance_to_output = {}
        if obj_name not in self.cur_distance_to_output.keys():
            self.cur_distance_to_output[obj_name] = self.distance_to_obj(self.cur_img_bgr, self.cur_pc_xyz, obj_name)
        return float(self.cur_distance_to_output[obj_name][pixel_loc[1], pixel_loc[0]])

    def _frontal_distance(self, pixel_loc, obj_name):
        """
        Returns the frontal distance to the given object at the given pixel location
        """
        if self.cur_frontal_distance_output is None:
            self.cur_frontal_distance_output = {}
        if obj_name not in self.cur_frontal_distance_output.keys():
            self.cur_frontal_distance_output[obj_name] = self.frontal_distance_to_obj(self.cur_img_bgr, self.cur_pc_xyz, obj_name)
        return float(self.cur_frontal_distance_output[obj_name][pixel_loc[1], pixel_loc[0]])

    def terrain(self, img_bgr, pc_xyz):
        cur_main_terrain_output = self.ns_infer_terrainseg.main_terrain(img_bgr, self.domain["terrains"])
        pred_seg, new_terrains = cur_main_terrain_output
        new_terrains_array = np.array(new_terrains)
        result = new_terrains_array[pred_seg]
        map_to_ldips = np.vectorize(lambda x: self.predefined_terrains.get(x, 1111))
        ret_out = map_to_ldips(result).squeeze().reshape((img_bgr.shape[0], img_bgr.shape[1]))
        return ret_out

    def in_the_way(self, img_bgr, pc_xyz):
        cur_main_in_the_way_output = self.ns_infer_terrainseg.main_in_the_way(img_bgr, pc_xyz, self.domain["terrains"], param=self.WAY_PARAM)
        ret_out = cur_main_in_the_way_output.reshape((img_bgr.shape[0], img_bgr.shape[1]))
        return ret_out

    def slope(self, img_bgr, pc_xyz):
        cur_main_slope_output = self.ns_infer_terrainseg.main_slope(img_bgr, pc_xyz, scale=self.SLOPE_SCALE)
        ret_out = cur_main_slope_output.reshape((img_bgr.shape[0], img_bgr.shape[1]))
        return ret_out

    def distance_to_obj(self, img_bgr, pc_xyz, obj_name):
        cur_distance_to_obj_output = self.ns_infer_objdet.main_distance_to(img_bgr, pc_xyz, obj_name,
                                                                           box_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[0],
                                                                           text_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[1],
                                                                           nms_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[2])
        dist_arr, found = cur_distance_to_obj_output
        ret_out = dist_arr.reshape((img_bgr.shape[0], img_bgr.shape[1])).astype(np.float32)
        return ret_out

    def frontal_distance_to_obj(self, img_bgr, pc_xyz, obj_name):
        cur_frontal_distance_to_obj_output = self.ns_infer_objdet.main_frontal_distance(img_bgr, pc_xyz, obj_name,
                                                                                        box_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[0],
                                                                                        text_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[1],
                                                                                        nms_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[2])
        dist_arr, found = cur_frontal_distance_to_obj_output
        ret_out = dist_arr.reshape((img_bgr.shape[0], img_bgr.shape[1])).astype(np.float32)
        return ret_out


def bind_method(instance, name):
    def method(arg):
        dynamic_method = getattr(instance, name)
        return dynamic_method(arg)
    return method


if __name__ == "__main__":
    hitl_llm_state = json_reader("test/demonstrations/state.json")
    noext_name = "000000"
    CUR_IMG_BGR = cv2.imread(f"test/{noext_name}.png")
    CUR_PC_XYZ = np.fromfile(f"test/{noext_name}.bin", dtype=np.float32).reshape((-1, 4))[:, :3]
    DOMAIN = hitl_llm_state["domain"]
    filled_lfps_sketch = json_reader("test/demonstrations/seqn_filled_lfps_sketches.json")[str(3)]
    fi = FasterImageInference(DOMAIN)
    terrain = fi._terrain
    in_the_way = fi._in_the_way
    slope = fi._slope
    for method_name in ['distance_to_' + obj for obj in DOMAIN["objects"]]:
        globals()[method_name] = bind_method(fi, f"_{method_name}")
    for method_name in ['frontal_distance_' + obj for obj in DOMAIN["objects"]]:
        globals()[method_name] = bind_method(fi, f"_{method_name}")
    exec(filled_lfps_sketch)
    fi.set_state(noext_name=noext_name,
                 img_bgr=CUR_IMG_BGR,
                 pc_xyz=CUR_PC_XYZ)
    is_safe_mask = np.zeros((CUR_IMG_BGR.shape[0], CUR_IMG_BGR.shape[1]), dtype=np.uint8)
    for k in range(CUR_IMG_BGR.shape[0]):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Processing row: {k}/{CUR_IMG_BGR.shape[0]}")
        for j in range(CUR_IMG_BGR.shape[1]):
            print(f"Processing column: {j}/{CUR_IMG_BGR.shape[1]}", end='\r')
            try:
                is_safe_mask[k, j] = is_safe((j, k))
            except:
                print(red(f"\nError in processing pixel: {j}/{CUR_IMG_BGR.shape[1]}\n", "bold"))
                is_safe_mask[k, j] = False
        print("\033[F\033[K", end="")  # Move up and clear the line
    is_safe_mask[is_safe_mask == 0] = 2
    overlay = TerrainSegFormer.get_seg_overlay(CUR_IMG_BGR, is_safe_mask, alpha=0.24)
    cv2.imshow("overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
