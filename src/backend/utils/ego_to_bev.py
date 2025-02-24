import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from third_party.jackal_calib import JackalLidarCamCalibration
from PIL import Image
import matplotlib.pyplot as plt
import cv2


class EgoToBEV:
    def __init__(self):
        self.lidar_cam = JackalLidarCamCalibration(ros_flag=False)
        self.bev_lidar_cam = JackalLidarCamCalibration(ros_flag=False, cam_extrinsics_filepath="config/bev_params/baselink_to_zed_left_extrinsics.yaml")

    def get_3d_bev(self, pil_img, pc_np, using="pc", inpaint=True):
        """
        Uses the lidar pointcloud to calculate BEV image.
        pil_img: A x B PIL Image object (RGB)
        pc_np: N x ? numpy array of point cloud (x, y, z, whatever)
        using: "pc" or "z" to use the pointcloud OR Z values of the pointcloud (+ camera homography) respectively
        Returns: A x B Numpy array of BEV image (RGB)        
        """
        pc_np = pc_np[:, :3]
        np_pil_img = np.array(pil_img)
        np_cv2_img = cv2.cvtColor(np_pil_img, cv2.COLOR_RGB2BGR)
        all_ys, all_xs = np.meshgrid(np.arange(pil_img.height), np.arange(pil_img.width))
        all_pixel_locs = np.stack((all_xs.flatten(), all_ys.flatten()), axis=-1)  # K x 2
        if using == "z":
            _, _, _, all_vlp_zs, _ = self.lidar_cam.projectPCtoImageFull(pc_np, np_cv2_img)
            all_wcs_coords, mask = self.lidar_cam.projectPCStoWCSusingZ(all_pixel_locs, all_vlp_zs)
            all_pixel_locs = all_pixel_locs[mask]
            bev_np_pil_img = np.zeros_like(np_pil_img)
            bev_pixel_locs, bev_mask = self.bev_lidar_cam.jackal_cam_calib.projectWCStoPCS(all_wcs_coords)
        elif using == "pc":
            _, all_vlp_coords, _, _, interp_mask = self.lidar_cam.projectPCtoImageFull(pc_np, np_cv2_img, do_nearest=False)
            all_pixel_locs = all_pixel_locs[interp_mask]
            bev_np_pil_img = np.zeros_like(np_pil_img)
            bev_pixel_locs, bev_mask, _ = self.bev_lidar_cam.projectVLPtoPCS(all_vlp_coords)
        else:
            raise ValueError("using must be 'pc' or 'z'")
        all_pixel_locs = all_pixel_locs[bev_mask]
        rows_bev, cols_bev = bev_pixel_locs[:, 1], bev_pixel_locs[:, 0]
        rows_all, cols_all = all_pixel_locs[:, 1], all_pixel_locs[:, 0]
        bev_np_pil_img[rows_bev, cols_bev] = np_pil_img[rows_all, cols_all]
        if inpaint:
            inpaint_mask = np.all(bev_np_pil_img == [0, 0, 0], axis=-1).astype(np.uint8)
            polygon_mask = np.zeros((bev_np_pil_img.shape[0], bev_np_pil_img.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(polygon_mask, bev_pixel_locs, 1)
            combined_mask = cv2.bitwise_and(inpaint_mask, polygon_mask)
            bev_np_pil_img = cv2.inpaint(bev_np_pil_img, combined_mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
        return bev_np_pil_img


if __name__ == "__main__":
    raw_pil_img = Image.open("test/000000.png")
    pc_np = np.fromfile("test/000000.bin", dtype=np.float32).reshape((-1, 4))
    ego_to_bev = EgoToBEV()

    f, axs = plt.subplots(1, 2)
    f.set_figheight(30)
    f.set_figwidth(50)
    axs[0].set_title("Raw", {'fontsize': 40})
    axs[0].imshow(raw_pil_img)
    axs[1].set_title("BEV", {'fontsize': 40})
    axs[1].imshow(ego_to_bev.get_3d_bev(raw_pil_img, pc_np, using="pc"))
    plt.show()
