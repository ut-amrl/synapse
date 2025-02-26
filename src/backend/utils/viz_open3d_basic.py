import numpy as np
import open3d as o3d
from simple_colors import green
import os
import cv2
from tqdm import tqdm
import shutil


def images_to_video(dir_path, video_path, skip_rate=1):
    images = sorted([img for img in os.listdir(dir_path) if img.endswith(".png")])
    if not images:
        print("No images found in the directory.")
        return
    first_image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10, (width, height), True)
    for i, image in enumerate(tqdm(images, desc="Writing Video", total=len(images))):
        img_path = os.path.join(dir_path, image)
        frame = cv2.imread(img_path)
        if i % skip_rate != 0:
            out.write(save_frame)
        else:
            save_frame = frame
            out.write(save_frame)
    out.release()
    print(green(f"Video saved at {video_path}", ["bold"]))


def images_from_pcds(pcd_dir, output_dir, camera_params_json):
    os.makedirs(output_dir, exist_ok=True)
    pcds = [o3d.io.read_point_cloud(f"{pcd_dir}/{i:06}.pcd") for i in range(len(os.listdir(pcd_dir)))]
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.point_size = 0.85  # Smaller point size
    param = o3d.io.read_pinhole_camera_parameters(camera_params_json)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)
    for i, pcd in enumerate(tqdm(pcds, desc="pcds", total=len(pcds))):
        vis.clear_geometries()
        vis.add_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.update_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"{output_dir}/{i:06}.png")
    vis.destroy_window()


def visualize_point_cloud_norgb(pc_np):
    pcd = o3d.geometry.PointCloud()
    xyz = pc_np[:, :3]
    pcd.points = o3d.utility.Vector3dVector(xyz)
    orange_color = [1, 0.647, 0]  # RGB for orange
    pcd.colors = o3d.utility.Vector3dVector(np.tile(orange_color, (len(xyz), 1)))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 0.75  # Smaller point size
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    vis.run()
    vis.destroy_window()


def write_o3d_camera_params(vis, jsonpath):
    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(jsonpath, camera_params)


def visualize_pcd(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.point_size = 0.85  # Smaller point size
    param = o3d.io.read_pinhole_camera_parameters("config/open3d_cameraview_params.json")
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.clear_geometries()
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.update_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    # write_o3d_camera_params(vis, "camera_params.json")
    vis.destroy_window()


if __name__ == "__main__":
    # pc_np = np.fromfile("test/000000.bin", dtype=np.float32).reshape((-1, 4))
    # visualize_point_cloud_norgb(pc_np)

    images_from_pcds(pcd_dir="test/demonstrations/pcds",
                     camera_params_json="config/open3d_cameraview_params.json",
                     output_dir="test/demonstrations/pcd_imgs")

    # visualize_pcd("test/demonstrations/pcds/000000.pcd")
