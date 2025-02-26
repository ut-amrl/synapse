import os
import cv2
import rosbag
from PIL import Image
import open3d as o3d
import PIL
import shutil
import numpy as np
from cv_bridge import CvBridge
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.backend.terrain_infer import TerrainSegFormer
from src.viz.fast_infer import FastModels
from simple_colors import green
from tqdm.auto import tqdm


def save_pcd(pointcloud, filename="pointcloud.pcd"):
    """Save an Open3D point cloud to a PCD file."""
    o3d.io.write_point_cloud(filename, pointcloud)


def load_pcd(filename="pointcloud.pcd"):
    """Load a PCD file as an Open3D point cloud."""
    return o3d.io.read_point_cloud(filename)


def pil_img_to_pointcloud(image: PIL.Image.Image):
    full_pred_seg, accumulated_results, terrain_results, depth_results, gsam_results = fm.predict_new(image)
    cv2_image_np = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    cv2_pred_overlay = TerrainSegFormer.get_seg_overlay(cv2_image_np, full_pred_seg)
    pilnp_pred_overlay = cv2.cvtColor(cv2_pred_overlay, cv2.COLOR_BGR2RGB)
    pcd = depth_results[1]
    colors = np.asarray(pilnp_pred_overlay).reshape(-1, 3) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def read_images_from_bag(bag_file, topic):
    bridge = CvBridge()
    bag = rosbag.Bag(bag_file)
    images = []
    for topic, msg, t in bag.read_messages(topics=[topic]):
        # Using cv_bridge to convert compressed image message to cv2 format
        cv2_frame_np = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        images.append(cv2_frame_np)
    bag.close()
    return images


def save_bag_to_pcds(bag_file, image_topic, pcd_dir, tempdir="test/tempdir"):
    os.makedirs(tempdir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)

    # Read images from the bag file
    print(green(f"Reading images from the bag file: {bag_file}", ["bold"]))
    images = read_images_from_bag(bag_file, image_topic)
    print(green(f"Number of images read: {len(images)}", ["bold"]))

    for i, image in enumerate(tqdm(images, desc="Saving pcds", total=len(images))):
        # Convert image to point cloud
        pcd = pil_img_to_pointcloud(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        save_pcd(pcd, os.path.join(pcd_dir, f"{i:06}.pcd"))

    # Clean up
    shutil.rmtree(tempdir)


if __name__ == "__main__":
    fm = FastModels()
    save_bag_to_pcds(bag_file="test/demonstrations/2.bag",
                     image_topic="/zed2i/zed_node/left/image_rect_color/compressed",
                     pcd_dir="test/demonstrations/pcds")
