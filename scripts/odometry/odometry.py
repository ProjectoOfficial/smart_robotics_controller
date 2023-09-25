#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import time
from odometry.depth import get_stereo_depth_manual, decomposition, get_stereo_maps, get_stereo_depth
from odometry.features import feature_extractor, feature_matching, get_detector, get_matcher
from odometry.motion import motion_estimation
from filterpy.kalman import KalmanFilter

from imu.imu import IMU
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm


class Odometry:
    def __init__(self, left_camera_info: dict, right_camera_info: dict, config:dict) -> None:
        self.left_camera_info = left_camera_info
        self.right_camera_info = right_camera_info
        self.config = config
        
        num_channels = 3 if config["rgb"] else 1
        block_size = 1
        self.stereo_config=dict(
            numDisparities=128,
            minDisparity=0,
            blockSize=block_size,
            P1=8 * num_channels * block_size ** 2,
            P2=32 * num_channels * block_size ** 2,
            mode=3
            )
        
        self.stereo_matcher = cv2.StereoSGBM.create(**self.stereo_config)
        self.left_stereo_map_x, self.left_stereo_map_y, self.right_stereo_map_x, self.right_stereo_map_y = \
        get_stereo_maps(self.left_camera_info, self.right_camera_info)
        
        self.detector = get_detector(config["detector"])
        self.keypoints_matcher = get_matcher(config["matcher"])
        
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            flags=cv2.MOTION_AFFINE,
            minEigThreshold=None,
        )
        
        self.cur_pose = np.eye(4)
        self.pose_visual = self.cur_pose
        self.pose_imu = self.cur_pose
        self.prev_frame_l, self.prev_frame_r, self.prev_depth = None, None, None

        self.imu = IMU(1)
        
    
    def _sanity_check(self, image: np.ndarray):
        assert image is not None, "Image is None!"
        assert image.size > 0, "Image is an empty tensor!"
    
    
    def clip(self, array:np.ndarray, threshold:float=1e-3):
        mask1 = array > -threshold
        mask2 = array < threshold
        array[mask1 & mask2] = 0
        return array
    
    
    def angular_velocity_to_rotation_matrix(self, angular_velocity:np.ndarray, delta_t:np.ndarray) -> np.ndarray:
        wx, wy, wz = angular_velocity
        
        skew_symmetric = np.array([
            [ 0, -wz,  wy],
            [ wz,  0, -wx],
            [-wy,  wx,  0]])
        
        rotation = expm(skew_symmetric * delta_t)

        return rotation
    
    
    def process(self, l_image: np.ndarray, r_image: np.ndarray, depth: np.ndarray, debug:bool=False) -> (float, float):
        self._sanity_check(l_image)
        self._sanity_check(r_image)
        
        if not self.config["rgb"]:
            l_image = cv2.cvtColor(l_image, cv2.COLOR_RGB2GRAY)
            r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2GRAY)
        mask = None
        
        if self.prev_frame_l is None:
            self.prev_frame_l = l_image
        if self.prev_frame_r is None:
            self.prev_frame_r = r_image
        if self.prev_depth is None:
            self.prev_depth = depth
        
        start = time.perf_counter()
        
        if self.config["stereo_depth"]:
            # stereo_depth = get_stereo_depth_manual(self.stereo_matcher, self.prev_frame_l, self.prev_frame_r, P0=self.left_camera_info["P"], P1=self.right_camera_info["P"]) 
            stereo_depth = get_stereo_depth(self.left_stereo_map_x, self.left_stereo_map_y, self.right_stereo_map_x, self.right_stereo_map_y, self.stereo_matcher, l_image, r_image)            
            nan_mask = np.isnan(depth)
            ri, ci = np.where(nan_mask)
            depth[ri, ci] = stereo_depth[ri, ci]
            
            if debug:
                cv2.imshow("stereo depth", stereo_depth / np.nanmax(stereo_depth))
                cv2.waitKey(0)
            
        keypoint_left_first, descriptor_left_first = feature_extractor(self.prev_frame_l, self.detector, mask)
        keypoint_left_next, descriptor_left_next = feature_extractor(l_image, self.detector, mask)
        
        matches = feature_matching(self.keypoints_matcher,
                                   descriptor_left_first,
                                   descriptor_left_next,
                                   distance_threshold=self.config["distance_threshold"])
        
        left_instrinsic_matrix, _, _ = decomposition(self.left_camera_info["P"])
        rotation_matrix, translation_vector, _, _ = motion_estimation(
            matches, keypoint_left_first, keypoint_left_next, left_instrinsic_matrix, self.left_camera_info["D"], depth, self.config)
        
        l_acc, a_vel, t, orient = self.imu.get_avg_values()
        
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = self.clip(rotation_matrix)
        transformation_matrix[:3, 3] = self.clip(translation_vector.T)
        self.pose_visual = self.pose_visual @ np.linalg.inv(transformation_matrix)
        
        if l_acc is not None and a_vel is not None and t is not None and orient is not None:
            delta_t = np.diff(t)
            
            # delta_rotation = R.from_euler('xyz', a_vel * delta_t, degrees=False).as_matrix()
            delta_rotation = self.clip(self.angular_velocity_to_rotation_matrix(a_vel, delta_t))
            # delta_rotation = delta_rotation @ self.cur_pose[:3, :3].T

            delta_translation = self.clip(delta_rotation @ (l_acc * delta_t**2) / 2)

            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = delta_rotation
            transformation_matrix[:3, 3] = delta_translation
            self.pose_imu = self.pose_imu @ np.linalg.inv(transformation_matrix)
        
            self.pose = (self.pose_visual + self.pose_imu) / 2
            print(R.from_matrix(self.pose[:3,:3]).as_euler("xyz", True))
        else:
            self.pose = self.pose_visual
            
        self.pose_visual = self.pose_imu = self.pose
        
        if debug:
            rospy.loginfo(f"pose: {self.pose}")
            image = np.zeros_like(self.prev_frame_l)
            image = cv2.drawKeypoints(self.prev_frame_l, keypoint_left_first, image, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            image = cv2.drawKeypoints(image, keypoint_left_next, image, (255, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("left image", image)
            cv2.imshow("depth", (depth / depth.max()) * 255 )
            cv2.waitKey(1000//60)
        
        self.prev_frame_l = l_image
        self.prev_frame_r = r_image
        self.prev_depth = depth
        
        end = time.perf_counter()
        
        xs = self.pose[0, 3]
        ys = self.pose[1, 3]
        zs = self.pose[2, 3]
        
        # print(f"Current pose: \n{(xs, ys, zs)}")
        print(f"FPS: {1/(end - start)}")
        return (zs, xs, ys), R.from_matrix(self.pose[:3, :3]).as_quat()