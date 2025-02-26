import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import torch
import PIL
from PIL import Image
import cv2
from pykdtree.kdtree import KDTree
import open3d as o3d
import matplotlib.pyplot as plt
torch.backends.cuda.matmul.allow_tf32 = True
from src.backend.gsam import GSAM as FastGSAM
from src.backend.depth2 import DepthAny2
from src.backend.terrain_infer import TerrainSegFormer
from third_party.jackal_calib import JackalLidarCamCalibration


class FastModels:
    @torch.inference_mode()
    def __init__(self):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            # Terrain model
            self.terrain_model = TerrainSegFormer()
            self.terrain_model.load_model_inference()

            # Grounded SAM object detection model
            self.gsam = FastGSAM(box_threshold=0.25, text_threshold=0.25, nms_threshold=0.6)
            self.gsam_object_dict = {
                "person": 0,
                "bush": 1,
                "car": 2,
                "pole": 3,
                "entrance": 4,
                "staircase": 5
            }

            # Lidar to camera calibration
            self.lidar_cam_calib = JackalLidarCamCalibration(ros_flag=False)

            # Metric Depth model
            self.metric_depth_model = DepthAny2()

    @torch.inference_mode()
    def terrain_pred(self, image: PIL.Image.Image):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            pred_img, pred_seg = self.terrain_model.predict_new(image)
        return pred_img, pred_seg.squeeze().cpu().numpy()

    @torch.inference_mode()
    def depth_metric_pred(self, image: PIL.Image.Image):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            pil_img_np = np.asarray(image.convert('RGB'))
            cv2_img = cv2.cvtColor(pil_img_np, cv2.COLOR_RGB2BGR)
            depth_arr = self.metric_depth_model.predict(cv2_img)
            pred = depth_arr.squeeze()
            resized_pred = Image.fromarray(pred).resize((self.lidar_cam_calib.img_width, self.lidar_cam_calib.img_height), Image.NEAREST)
        return np.asarray(resized_pred)

    @torch.inference_mode()
    def get_pc_from_depth(self, image: PIL.Image.Image):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            z = self.depth_metric_pred(image)
            image = image.convert('RGB')
            FX = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix'][0, 0]
            FY = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix'][1, 1]
            CX = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix'][0, 2]
            CY = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix'][1, 2]
            K = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix']
            d = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['dist_coeffs']
            R = np.eye(3)
            x, y = np.meshgrid(np.arange(self.lidar_cam_calib.img_width), np.arange(self.lidar_cam_calib.img_height))
            # undistort pixel coordinates
            pcs_coords = np.stack((x.flatten(), y.flatten()), axis=-1).astype(np.float64)
            undistorted_pcs_coords = cv2.undistortPoints(pcs_coords.reshape(1, -1, 2), K, d, R=R, P=K)
            undistorted_pcs_coords = np.swapaxes(undistorted_pcs_coords, 0, 1).squeeze().reshape((-1, 2))
            x, y = np.split(undistorted_pcs_coords, 2, axis=1)
            x = x.reshape(self.lidar_cam_calib.img_height, self.lidar_cam_calib.img_width)
            y = y.reshape(self.lidar_cam_calib.img_height, self.lidar_cam_calib.img_width)
            # back project (along the camera ray) the pixel coordinates to 3D using the depth
            """
            For understanding, ignore distortions. According to pinhole model, pinhole is at (0,0,0) in CCS and image plane at -f in z axis, and equivalent image plane at +f in z axis which is what we actually use for calculations.
            So similarity of triangles gives that, for (X, Y, Z) in CCS, we get the point mapped to (fxX/Z, fyY/Z, f) on image plane.
            Dropping z=f, we get pixel coordinates upon shifting origin to top-left corner, (cx + fxX/Z, cy + fyY/Z).
            So, HERE below, what we are basically doing is the inverse process of this, i.e., given the pixel coords x y z and the CCS depth Z:
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy
            """
            x = (x - CX) / FX
            y = (y - CY) / FY
            points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            colors = np.asarray(image).reshape(-1, 3) / 255.0
            # convert to open3d point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([pcd])  # visualize the point cloud
            # pcd = o3d.io.read_point_cloud("pointcloud.ply")  # read the point cloud
            # o3d.io.write_point_cloud("pointcloud.ply", pcd)  # save the point cloud
        return z, pcd

    @torch.inference_mode()
    def gsam_pred(self, image: cv2.imread):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            ann_img, dets, per_class_mask = self.gsam.predict_and_segment_on_image(image, list(self.gsam_object_dict.keys()))
        return ann_img, dets, per_class_mask

    @torch.inference_mode()
    def depth_true(self, pc_xyz: np.ndarray):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            vlp_points = np.asarray(pc_xyz).astype(np.float64)
            wcs_coords = self.lidar_cam_calib.projectVLPtoWCS(vlp_points)
            ccs_coords = self.lidar_cam_calib.jackal_cam_calib.projectWCStoCCS(wcs_coords)
            corresponding_pcs_coords, mask = self.lidar_cam_calib.jackal_cam_calib.projectCCStoPCS(ccs_coords, mode='skip')
            ccs_coords = ccs_coords[mask]
            corresponding_ccs_depths = ccs_coords[:, 2].reshape((-1, 1))
            all_ys, all_xs = np.meshgrid(np.arange(self.lidar_cam_calib.img_height), np.arange(self.lidar_cam_calib.img_width))
            all_pixel_locs = np.stack((all_xs.flatten(), all_ys.flatten()), axis=-1)
            all_ccs_depths, _ = self.lidar_cam_calib.double_interp(a=corresponding_pcs_coords, b=corresponding_ccs_depths, x=all_pixel_locs, do_nearest=True, firstmethod="linear")
        return all_pixel_locs, all_ccs_depths

    def process(self, image: PIL.Image.Image, terrain_results, depth_results, gsam_results):
        terrain_pred_seg = terrain_results[1].squeeze().copy()
        terrain_pred_seg[terrain_pred_seg == 0] = 20
        terrain_pred_seg[(terrain_pred_seg == 2) | (terrain_pred_seg == 6) | (terrain_pred_seg == 9)] = 0
        terrain_pred_seg[terrain_pred_seg != 0] = 1
        terrain_pred_seg_bool = terrain_pred_seg.astype(bool)
        inv_terrain_pred_seg_bool = ~terrain_pred_seg_bool

        per_class_mask = gsam_results[2]
        pcd = depth_results[1]
        all_pc_coords = np.asarray(pcd.points).reshape((-1, 3))
        x, y = np.meshgrid(np.arange(self.lidar_cam_calib.img_width), np.arange(self.lidar_cam_calib.img_height))
        all_pixel_locs = np.stack((x.flatten(), y.flatten()), axis=-1)
        terrain_pixel_locs = all_pixel_locs[inv_terrain_pred_seg_bool.flatten()]
        terrain_pc_coords = all_pc_coords[inv_terrain_pred_seg_bool.flatten()]

        distance_to_arrays = {}
        for cidx in range(per_class_mask.shape[0]):
            dist_arr = np.ones((image.height, image.width)) * (-1.0)
            class_mask = per_class_mask[cidx].squeeze()
            all_mask_values = class_mask[all_pixel_locs[:, 1], all_pixel_locs[:, 0]]
            class_pixel_locs = all_pixel_locs[all_mask_values]
            class_pc_coords = all_pc_coords[all_mask_values]
            if class_pc_coords is None or class_pc_coords.shape[0] == 0:
                dist_arr[terrain_pixel_locs[:, 1], terrain_pixel_locs[:, 0]] = np.inf
            else:
                kdtree = KDTree(class_pc_coords)
                distances, _ = kdtree.query(terrain_pc_coords)
                dist_arr[terrain_pixel_locs[:, 1], terrain_pixel_locs[:, 0]] = distances
            distance_to_arrays[cidx] = dist_arr

        in_the_way_mask = np.ones((image.height, image.width), dtype=np.uint8) * (-1)
        tpred_seg = terrain_results[1].squeeze()
        non_traversable_seg = ((tpred_seg == 0) | (tpred_seg == 1) | (tpred_seg == 3) | (tpred_seg == 13)).astype(bool)
        linearized_non_traversable_seg = non_traversable_seg.flatten()
        non_traversable_pc_coords = all_pc_coords[linearized_non_traversable_seg]
        # terrain_pc_xy = terrain_pc_coords[:, :2]
        # non_traversable_pc_xy = non_traversable_pc_coords[:, :2]
        tree_nontraversable = KDTree(non_traversable_pc_coords)
        distances, _ = tree_nontraversable.query(terrain_pc_coords)
        too_far_mask = (distances > 2.5).astype(np.uint8)
        in_the_way_mask[terrain_pixel_locs[:, 1], terrain_pixel_locs[:, 0]] = too_far_mask
        return terrain_pred_seg, distance_to_arrays, in_the_way_mask

    def _terrain(self, image: PIL.Image.Image, accumulated_results):
        return accumulated_results[0]

    def _in_the_way(self, image: PIL.Image.Image, accumulated_results):
        return accumulated_results[2]

    def _distance_to(self, image: PIL.Image.Image, class_name: str, accumulated_results):
        return accumulated_results[1][self.gsam_object_dict[class_name]]

    def _frontal_distance(self, image: PIL.Image.Image, class_name: str, accumulated_results):
        return self._distance_to(image, class_name, accumulated_results)

    @torch.inference_mode()
    def predict_new(self, image: PIL.Image.Image, do_car=True):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            terrain_results = self.terrain_pred(image)
            depth_results = self.get_pc_from_depth(image)
            gsam_results = self.gsam_pred(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
            accumulated_results = self.process(image, terrain_results, depth_results, gsam_results)
        if do_car:
            return ((self._terrain(image, accumulated_results) == 0) &
                    (self._distance_to(image, "person", accumulated_results) > 2.0) &
                    (self._distance_to(image, "pole", accumulated_results) > 0.0) &
                    (self._in_the_way(image, accumulated_results) == 0) &
                    (self._frontal_distance(image, "entrance", accumulated_results) > 3.5) &
                    (self._distance_to(image, "bush", accumulated_results) > 0.0) &
                    (self._frontal_distance(image, "staircase", accumulated_results) > 3.5) &
                    (self._distance_to(image, "car", accumulated_results) > 8.0)
                    ).astype(np.uint8), accumulated_results, terrain_results, depth_results, gsam_results
        return ((self._terrain(image, accumulated_results) == 0) &
                (self._distance_to(image, "person", accumulated_results) > 2.0) &
                (self._distance_to(image, "pole", accumulated_results) > 0.0) &
                (self._in_the_way(image, accumulated_results) == 0)
                # (self._frontal_distance(image, "entrance", accumulated_results) > 3.5) &
                # (self._distance_to(image, "bush", accumulated_results) > 0.0) &
                # (self._frontal_distance(image, "staircase", accumulated_results) > 3.5) &
                # (self._distance_to(image, "car", accumulated_results) > 8.0)
                ).astype(np.uint8), accumulated_results, terrain_results, depth_results, gsam_results


if __name__ == "__main__":
    test_sample_image_path = "test/000000.png"
    fm = FastModels()
    pil_img = Image.open(test_sample_image_path)
    pred, *_ = fm.predict_new(pil_img)
    plt.imshow(pred * 255)
    plt.show()
