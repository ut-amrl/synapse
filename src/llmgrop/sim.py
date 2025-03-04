import os
import sys
import numpy as np
import math
import re
import pybullet
import pybullet_data
import imageio_ffmpeg
from tqdm import tqdm


class Client():
    def __init__(self):
        pybullet.connect(pybullet.DIRECT)  # pybullet.GUI for local GUI.
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(0, 0, -9.8)

        # reset robot
        self.plane_id = pybullet.loadURDF("plane.urdf")
        self.robot_id = pybullet.loadURDF("kuka_iiwa/model_vr_limits.urdf", basePosition=[1.55, 0.0, 0.6], baseOrientation=[0.0, 0.0, 0.0, 1.0])
        jointPositions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for jointIndex in range(pybullet.getNumJoints(self.robot_id)):
            pybullet.resetJointState(self.robot_id, jointIndex, jointPositions[jointIndex])
            pybullet.setJointMotorControl2(self.robot_id, jointIndex, pybullet.POSITION_CONTROL, jointPositions[jointIndex], 0)

        # camera width and height
        self.cam_width = 480
        self.cam_height = 480

        # create a list to record utensil id
        self.utensil_id = {}
        self.gripper_id = None

    def render_image(self):
        # camera parameters
        cam_target_pos = [1.0, 0.0, 0.5]
        # cam_distance = 1.5
        cam_distance = 0.7
        cam_yaw, cam_pitch, cam_roll = -90, -90, 0
        cam_up, cam_up_axis_idx, cam_near_plane, cam_far_plane, cam_fov = [0, 0, 1], 2, 0.01, 100, 60
        cam_view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(cam_target_pos, cam_distance, cam_yaw, cam_pitch, cam_roll, cam_up_axis_idx)
        cam_projection_matrix = pybullet.computeProjectionMatrixFOV(cam_fov, self.cam_width * 1. / self.cam_height, cam_near_plane, cam_far_plane)
        znear, zfar = 0.01, 10.

        # get raw data
        _, _, color, depth, segment = pybullet.getCameraImage(
            width=self.cam_width,
            height=self.cam_height,
            viewMatrix=cam_view_matrix,
            projectionMatrix=cam_projection_matrix,
            shadow=1,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        # get color image.
        color_image_size = (self.cam_width, self.cam_height, 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel

        # get depth image.
        depth_image_size = (self.cam_width, self.cam_height)
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
        depth = (2 * znear * zfar) / depth

        # get segment image.
        segment = np.reshape(segment, [self.cam_width, self.cam_height]) * 1. / 255.
        return color, depth, segment

    def reset_video(self, video_path='video.mp4'):
        video = imageio_ffmpeg.write_frames(video_path, (self.cam_width, self.cam_height), fps=60)
        video.send(None)  # seed the video writer with a blank frame
        return video

    def render_video(self, video, image):
        video.send(np.ascontiguousarray(image))

    def add_table(self):
        flags = pybullet.URDF_USE_INERTIA_FROM_FILE
        path = 'urdf_models/'

        if not os.path.exists(path):
            print('!Error: cannot find urdf_models/!')
            sys.exit()

        # add table
        table_id = pybullet.loadURDF("urdf_models/furniture_table_square/table.urdf", basePosition=[1.0, 0.0, 0.0], baseOrientation=[0, 0, 0.7071, 0.7071])
        self.utensil_id['table'] = table_id

    def add_objects(self, utensil_name, utensil_init_pose):
        flags = pybullet.URDF_USE_INERTIA_FROM_FILE
        path = 'urdf_models/'

        if not os.path.exists(path):
            print('!Error: cannot find urdf_models/!')
            sys.exit()

        # add objects according to utensil_name
        color = 'blue'
        color2 = 'red'
        # plate & bowl category
        if 'bread plate' in utensil_name:
            self.utensil_id['bread plate'] = pybullet.loadURDF(path + 'utensil_plate_circle_' + color + '_small' + '/model.urdf', basePosition=utensil_init_pose['bread plate'][0], baseOrientation=utensil_init_pose['bread plate'][1], flags=flags)
        if 'dinner plate' in utensil_name:
            self.utensil_id['dinner plate'] = pybullet.loadURDF(path + 'utensil_plate_circle_' + color + '/model.urdf', basePosition=utensil_init_pose['dinner plate'][0], baseOrientation=utensil_init_pose['dinner plate'][1], flags=flags)
        if 'fruit bowl' in utensil_name:
            self.utensil_id['fruit bowl'] = pybullet.loadURDF(path + 'utensil_bowl_' + color + '_big' + '/model.urdf', basePosition=utensil_init_pose['fruit bowl'][0], baseOrientation=utensil_init_pose['fruit bowl'][1], flags=flags)

        # fork category
        if 'dinner fork' in utensil_name:
            self.utensil_id['dinner fork'] = pybullet.loadURDF(path + 'utensil_fork_' + color + '/model.urdf', basePosition=utensil_init_pose['dinner fork'][0], baseOrientation=utensil_init_pose['dinner fork'][1], flags=flags)
        if 'salad fork' in utensil_name:
            self.utensil_id['salad fork'] = pybullet.loadURDF(path + 'utensil_fork_' + color + '/model.urdf', basePosition=utensil_init_pose['salad fork'][0], baseOrientation=utensil_init_pose['salad fork'][1], flags=flags)
        if 'dessert fork' in utensil_name:
            self.utensil_id['dessert fork'] = pybullet.loadURDF(path + 'utensil_fork_' + color + '_small' + '/model.urdf', basePosition=utensil_init_pose['dessert fork'][0], baseOrientation=utensil_init_pose['dessert fork'][1], flags=flags)
        if 'seafood fork' in utensil_name:
            self.utensil_id['seafood fork'] = pybullet.loadURDF(path + 'utensil_fork_' + color + '/model.urdf', basePosition=utensil_init_pose['seafood fork'][0], baseOrientation=utensil_init_pose['seafood fork'][1], flags=flags)
        if 'fork' in utensil_name:
            self.utensil_id['fork'] = pybullet.loadURDF(path + 'utensil_fork_' + color + '/model.urdf', basePosition=utensil_init_pose['fork'][0], baseOrientation=utensil_init_pose['fork'][1], flags=flags)

        # knife category
        if 'butter knife' in utensil_name:
            self.utensil_id['butter knife'] = pybullet.loadURDF(path + 'utensil_knife_' + color + '_small' + '/model.urdf', basePosition=utensil_init_pose['butter knife'][0], baseOrientation=utensil_init_pose['butter knife'][1], flags=flags)
        if 'dinner knife' in utensil_name:
            self.utensil_id['dinner knife'] = pybullet.loadURDF(path + 'utensil_knife_' + color + '/model.urdf', basePosition=utensil_init_pose['dinner knife'][0], baseOrientation=utensil_init_pose['dinner knife'][1], flags=flags)
        if 'fish knife' in utensil_name:
            self.utensil_id['fish knife'] = pybullet.loadURDF(path + 'utensil_knife_' + color + '/model.urdf', basePosition=utensil_init_pose['fish knife'][0], baseOrientation=utensil_init_pose['fish knife'][1], flags=flags)

        # spoon category
        if 'soup spoon' in utensil_name:
            self.utensil_id['soup spoon'] = pybullet.loadURDF(path + 'utensil_spoon_' + color + '/model.urdf', basePosition=utensil_init_pose['soup spoon'][0], baseOrientation=utensil_init_pose['soup spoon'][1], flags=flags)
        if 'dessert spoon' in utensil_name:
            self.utensil_id['dessert spoon'] = pybullet.loadURDF(path + 'utensil_spoon_' + color + '_small' + '/model.urdf', basePosition=utensil_init_pose['dessert spoon'][0], baseOrientation=utensil_init_pose['dessert spoon'][1], flags=flags)
        if 'tea spoon' in utensil_name:
            self.utensil_id['tea spoon'] = pybullet.loadURDF(path + 'utensil_spoon_' + color + '_small' + '/model.urdf', basePosition=utensil_init_pose['tea spoon'][0], baseOrientation=utensil_init_pose['tea spoon'][1], flags=flags)

        # cup & glass category
        if 'water cup' in utensil_name:
            self.utensil_id['water cup'] = pybullet.loadURDF(path + 'utensil_cup_' + color + '_small' + '/model.urdf', basePosition=utensil_init_pose['water cup'][0], baseOrientation=utensil_init_pose['water cup'][1], flags=flags)
        if 'wine glass' in utensil_name:
            self.utensil_id['wine glass'] = pybullet.loadURDF(path + 'utensil_glass_' + color + '/model.urdf', basePosition=utensil_init_pose['wine glass'][0], baseOrientation=utensil_init_pose['wine glass'][1], flags=flags)
        if 'teacup' in utensil_name:
            utensil_init_pose['teacup'][1] = [0.0, 0.0, 0.707, 0.707]
            self.utensil_id['teacup'] = pybullet.loadURDF(path + 'utensil_teacup_' + color + '_big' + '/model.urdf', basePosition=utensil_init_pose['teacup'][0], baseOrientation=utensil_init_pose['teacup'][1], flags=flags)

        # food category
        if 'bread' in utensil_name:
            self.utensil_id['bread'] = pybullet.loadURDF(path + 'food_bread' + '/model.urdf', basePosition=utensil_init_pose['bread'][0], baseOrientation=utensil_init_pose['bread'][1], flags=flags)
        if 'strawberry' in utensil_name:
            self.utensil_id['strawberry'] = pybullet.loadURDF(path + 'food_strawberry' + '/model.urdf', basePosition=utensil_init_pose['strawberry'][0], baseOrientation=utensil_init_pose['strawberry'][1], flags=flags)

        # others
        if 'napkin' in utensil_name:
            self.utensil_id['napkin'] = pybullet.loadURDF(path + 'utensil_napkin_' + color + '/model.urdf', basePosition=utensil_init_pose['napkin'][0], baseOrientation=utensil_init_pose['napkin'][1], flags=flags)
        if 'place mat' in utensil_name:
            self.utensil_id['place mat'] = pybullet.loadURDF(path + 'utensil_mat_' + color2 + '_small' + '/model.urdf', basePosition=utensil_init_pose['place mat'][0], baseOrientation=utensil_init_pose['place mat'][1], flags=flags)
        if 'salt shaker' in utensil_name:
            self.utensil_id['salt shaker'] = pybullet.loadURDF(path + 'utensil_shaker_' + color + '/model.urdf', basePosition=utensil_init_pose['salt shaker'][0], baseOrientation=utensil_init_pose['salt shaker'][1], flags=flags)
        if 'pepper shaker' in utensil_name:
            self.utensil_id['pepper shaker'] = pybullet.loadURDF(path + 'utensil_shaker_' + color + '/model.urdf', basePosition=utensil_init_pose['pepper shaker'][0], baseOrientation=utensil_init_pose['pepper shaker'][1], flags=flags)
        if 'teacup mat' in utensil_name:
            self.utensil_id['teacup mat'] = pybullet.loadURDF(path + 'utensil_mat_' + color + '_small' + '/model.urdf', basePosition=utensil_init_pose['teacup mat'][0], baseOrientation=utensil_init_pose['teacup mat'][1], flags=flags)
        if 'teacup lid' in utensil_name:  # TBD
            self.utensil_id['teacup lid'] = pybullet.loadURDF(path + 'utensil_mat_' + color + '_small' + '/model.urdf', basePosition=utensil_init_pose['teacup lid'][0], baseOrientation=utensil_init_pose['teacup lid'][1], flags=flags)

        return self.utensil_id

    def get_bounding_box(self, obj_id):
        (min_x, min_y, min_z), (max_x, max_y, max_z) = pybullet.getAABB(obj_id)
        return [min_x, min_y, min_z], [max_x, max_y, max_z]

    def home_joints(self):
        jointPositions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for jointIndex in range(pybullet.getNumJoints(self.robot_id)):
            pybullet.resetJointState(self.robot_id, jointIndex, jointPositions[jointIndex])
            pybullet.setJointMotorControl2(self.robot_id, jointIndex, pybullet.POSITION_CONTROL, jointPositions[jointIndex], 0)

    def pick_place(self, object_name, object_id, object_position_init, object_position_end, video, target_orie_euler=None):
        num_joints = pybullet.getNumJoints(self.robot_id)
        end_effector_index = 6
        # target_position = [0.9, -0.6, 0.65]
        target_position = [object_position_init[0], object_position_init[1], object_position_init[2] + 0.1]

        for step in tqdm(range(1000)):
            if step % 4 == 0:  # PyBullet default simulation time step is 240fps, but we want to record video at 60fps.
                rgb, depth, mask = self.render_image()
                self.render_video(video, np.ascontiguousarray(rgb))

            if target_orie_euler is not None:
                target_orientation = pybullet.getQuaternionFromEuler(target_orie_euler)
            else:
                if 'teacup' not in object_name:
                    target_orientation = pybullet.getQuaternionFromEuler([0, 1.01 * math.pi, 0])
                else:
                    target_orientation = pybullet.getQuaternionFromEuler([0, 1.01 * math.pi, 0.5 * math.pi])

            gripper_status = {'ungrasp': 0, 'grasp': 1}
            gripper_value = gripper_status['ungrasp']
            if step >= 150 and step < 250:
                target_position = [object_position_init[0], object_position_init[1], object_position_init[2] + 0.1]  # grab object
                gripper_value = gripper_status['grasp']
            elif step >= 250 and step < 400:
                # target_position = [0.85, -0.2, 0.7 + 0.2*(step-250)/150.] # move up after picking object
                target_position = [object_position_init[0], object_position_init[1], object_position_init[2] + 0.3]
                gripper_value = gripper_status['grasp']
            elif step >= 400 and step < 600:
                # target_position = [0.85, -0.2 + 0.4*(step-400)/200., 0.9] # move to target position
                target_position = [object_position_init[0] + (object_position_end[0] - object_position_init[0]) * (step - 400) / 200, object_position_init[1] + (object_position_end[1] - object_position_init[1]) * (step - 400) / 200, object_position_init[2] + 0.3]
                gripper_value = gripper_status['grasp']
            elif step >= 600 and step < 700:
                target_position = [object_position_end[0], object_position_end[1], object_position_end[2] + 0.2]  # stop at target position
                gripper_value = gripper_status['grasp']
            elif step >= 700:
                target_position = [object_position_end[0], object_position_end[1], object_position_end[2] + 0.2]  # drop object
                # print('object_name:{}, target_position:{}'.format(object_name, target_position))
                gripper_value = gripper_status['ungrasp']

            joint_poses = pybullet.calculateInverseKinematics(self.robot_id, end_effector_index, target_position, target_orientation)
            for joint_index in range(num_joints):
                pybullet.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=joint_index, controlMode=pybullet.POSITION_CONTROL, targetPosition=joint_poses[joint_index])

            if gripper_value == 0 and self.gripper_id != None:
                pybullet.removeConstraint(self.gripper_id)
                self.gripper_id = None
            if gripper_value == 1 and self.gripper_id == None:
                cube_orn = pybullet.getQuaternionFromEuler([0, math.pi, 0])
                self.gripper_id = pybullet.createConstraint(self.robot_id, end_effector_index, object_id, -1, pybullet.JOINT_FIXED, [0, 0, 0], [0, 0, 0.05], [0, 0, 0], childFrameOrientation=cube_orn)

            pybullet.stepSimulation()

    def disconnect(self):
        pybullet.disconnect()


class Sim:
    """
    Sizes in meters
    """
    # Plate & Bowl Category
    bread_plate_width_length_height = "0.14, 0.14, 0.03"
    dinner_plate_width_length_height = "0.16, 0.16, 0.03"
    fruit_bowl_width_length_height = "0.15, 0.15, 0.07"

    # Fork Category
    dinner_fork_width_length_height = "0.2, 0.05, 0.03"
    salad_fork_width_length_height = "0.2, 0.05, 0.03"
    dessert_fork_width_length_height = "0.17, 0.05, 0.03"
    seafood_fork_width_length_height = "0.2, 0.05, 0.03"
    fork_width_length_height = "0.2, 0.05, 0.03"

    # Knife Category
    butter_knife_width_length_height = "0.19, 0.05, 0.03"
    dinner_knife_width_length_height = "0.22, 0.05, 0.03"
    fish_knife_width_length_height = "0.22, 0.05, 0.03"

    # Spoon Category
    dessert_spoon_width_length_height = "0.17, 0.05, 0.03"
    tea_spoon_width_length_height = "0.17, 0.05, 0.03"
    soup_spoon_width_length_height = "0.2, 0.05, 0.03"

    # Cup & Glass Category
    water_cup_width_length_height = "0.09, 0.09, 0.13"
    wine_glass_width_length_height = "0.07, 0.07, 0.17"
    teacup_width_length_height = "0.15, 0.1, 0.1"

    # Food Category
    bread_width_length_height = "0.1, 0.1, 0.06"
    strawberry_width_length_height = "0.1, 0.1, 0.05"

    # Others
    napkin_width_length_height = "0.1, 0.25, 0.03"
    place_mat_width_length_height = "0.12, 0.12, 0.03"
    salt_shaker_width_length_height = "0.06, 0.06, 0.1"
    pepper_shaker_width_length_height = "0.06, 0.06, 0.1"
    teacup_mat_width_length_height = "0.12, 0.12, 0.03"
    teacup_lid_width_length_height = "0.06, 0.06, 0.03"

    # Table information (in meters)
    boundary_min_x = 0.75
    boundary_max_x = 1.25
    boundary_min_y = 0.25
    boundary_max_y = -0.25
    beta = 1.0

    # Adjust boundaries based on beta
    boundary_min_x *= beta
    boundary_max_x *= beta
    boundary_min_y *= beta
    boundary_max_y *= beta

    # Calculate table position and size
    table_x, table_y = (boundary_min_x + boundary_max_x) / 2, (boundary_min_y + boundary_max_y) / 2
    table_width = abs(boundary_max_x - boundary_min_x)
    table_length = abs(boundary_max_y - boundary_min_y)

    def __init__(self, objects: list):
        self.utensils = objects
        self.utensil_size = {}
        pattern = "([0-9]*\.[0-9]*)"
        alpha = 1.05  # expand object size
        for utensil in objects:
            variable_name = utensil.replace(" ", "_")
            size_name = variable_name + "_width_length_height"
            matches = re.findall(pattern, self.__getattribute__(size_name))
            self.utensil_size[utensil] = [float(matches[0]) * alpha, float(matches[1]) * alpha, float(matches[2]) * alpha]

        # intialize utensils' pose
        self.utensil_init_pose = {}
        for utensil in objects:
            self.utensil_init_pose[utensil] = [[1.15, -0.18, 0.65 + self.utensil_size[utensil][2]], [0, 0, 0, 1]]

    def simulate_video(self, utensil_goal_pose, video_path="video.mp4"):
        demo = Client()
        demo.add_table()
        video = demo.reset_video(video_path)
        for item_name in self.utensils:
            print('item_name: {}'.format(item_name))
            # add objects
            demo.add_objects([item_name], self.utensil_init_pose)
            [min_x, min_y, min_z], [max_x, max_y, max_z] = demo.get_bounding_box(demo.utensil_id[item_name])
            # pick-place utensils
            object_position_init = self.utensil_init_pose[item_name][0]
            object_position_end = [utensil_goal_pose[item_name]['x'], utensil_goal_pose[item_name]['y'], utensil_goal_pose[item_name]['z']]
            demo.pick_place(item_name, demo.utensil_id[item_name], object_position_init, object_position_end, video, target_orie_euler=None)
        demo.home_joints()
        rgb, depth, mask = demo.render_image()
        demo.render_video(video, np.ascontiguousarray(rgb))
        video.close()
        demo.disconnect()

    @staticmethod
    def quick_simulate(objects: list, final_positions: list, final_orie_eulers: list):
        """
        positions in meters
        orie_eulers in radians
        """
        demo = Client()
        demo.add_table()
        pose = {}
        for i, obj in enumerate(objects):
            pose[obj] = [final_positions[i], pybullet.getQuaternionFromEuler(final_orie_eulers[i])]
        demo.add_objects(objects, pose)
        rgb, depth, segment = demo.render_image()
        # plt.imshow(rgb)
        # plt.show()
        demo.disconnect()
        return rgb, depth, segment
