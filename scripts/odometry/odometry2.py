#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import time
from scipy.optimize import least_squares
from utils.imu import IMU

class Odometry:
    def __init__(self, left_camera_info: dict, right_camera_info: dict, config:dict) -> None:
        self.left_camera_info = left_camera_info
        self.right_camera_info = right_camera_info
        
        num_channels = 3 if config["rgb"] else 1
        block_size = 7
        self.stereo_config=dict(
            numDisparities=6*16,
            minDisparity=0,
            blockSize=block_size,
            P1=8 * num_channels * block_size ** 2,
            P2=32 * num_channels * block_size ** 2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        
        self.disparity = cv2.StereoSGBM.create(self.stereo_config)
        self.kpts_detector = cv2.ORB.create() if config["detector"] == "orb" else \
            cv2.SIFT.create(edgeThreshold=50, contrastThreshold=0.2, enable_precise_upscale=True)
        
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            flags=cv2.MOTION_AFFINE,
            minEigThreshold=None,
        )
        
        self.feature_params = dict(maxCorners=100,
                                    qualityLevel=0.3,
                                    minDistance=7,
                                    blockSize=7)
        
        self.gt_path = []
        self.estimated_path = []
        self.camera_pose_list = []
        
        self.cur_pose = np.eye(4)
        self.prev_frame_l, self.prev_frame_r, self.prev_depth = None, None, None
        
        self.imu = IMU()
        
        
    def create_transformation(self, rotation:np.ndarray, traslation:np.ndarray) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = traslation
        return transform
        
        
    def get_keypoints(self, img):
        keypoints = self.kpts_detector.detect(img)
        if len(keypoints) > 10:
            keypoints = sorted(keypoints, key=lambda x: -x.response)
            keypoints = keypoints[:10]
        
        good_features = cv2.goodFeaturesToTrack(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask=None, **self.feature_params).squeeze(1)
        good_features = [cv2.KeyPoint(kp[0], kp[1], 20) for kp in good_features]
        return np.concatenate([keypoints])
    
    
    def track_keypoints(self, img1, img2, kp1, max_error=4):
        trackpoints1 = np.expand_dims(cv2.KeyPoint.convert(kp1), axis=1)
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, None, None, **self.lk_params)
        trackable = st.astype(bool)

        under_thresh = np.where(err[trackable] < max_error, True, False)

        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        h, w, c = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2
    
    
    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        in_bounds = np.logical_and(mask1, mask2)
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        
        return q1_l, q1_r, q2_l, q2_r
    
    
    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        r = dof[:3]
        rotation, _ = cv2.Rodrigues(r)
        traslation = dof[3:]
        transf = self.create_transformation(rotation, traslation)

        f_projection = np.matmul(self.left_camera_info["P"], transf)
        b_projection = np.matmul(self.left_camera_info["P"], np.linalg.inv(transf))

        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        q1_pred = Q2.dot(f_projection.T)
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        q2_pred = Q1.dot(b_projection.T)
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals
    
    
    def calc_3d(self, q1_l, q1_r, q2_l, q2_r, depth_image: np.ndarray, use_real_depth:bool=False):
        Q1 = cv2.triangulatePoints(self.left_camera_info["P"], self.right_camera_info["P"], q1_l.T, q1_r.T)
        Q1 = np.transpose(Q1[:3] / Q1[3])

        Q2 = cv2.triangulatePoints(self.left_camera_info["P"], self.right_camera_info["P"], q2_l.T, q2_r.T)
        Q2 = np.transpose(Q2[:3] / Q2[3])
        
        # Q1 = linear_LS_triangulation(q1_l, self.left_camera_info["P"], q1_r, self.right_camera_info["P"])[0]
        # Q2 = linear_LS_triangulation(q2_l, self.left_camera_info["P"], q2_r, self.right_camera_info["P"])[0]
            
        return Q1, Q2
    
    
    def ransac_pose_estimation(self, q1, q2, Q1, Q2, max_iter=100):
        early_termination_threshold = 5

        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            in_guess = np.zeros(6)
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200, args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                break

        r = out_pose[:3]
        rotation, _ = cv2.Rodrigues(r)
        traslation = out_pose[3:]
        transformation_matrix = self.create_transformation(rotation, traslation)
        return transformation_matrix

        
        
    def _sanity_check(self, image: np.ndarray):
        assert image is not None, "Image is None!"
        assert image.size > 0, "Image is an empty tensor!"
        
        
    def process(self, l_image: np.ndarray, r_image: np.ndarray, depth: np.ndarray, debug:bool=False) -> (float, float):        
        self._sanity_check(l_image)
        self._sanity_check(r_image)
        
        l_image = cv2.cvtColor(l_image, cv2.COLOR_RGB2BGR)
        r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
        
        if self.prev_frame_l is None:
            self.prev_frame_l = l_image
        if self.prev_frame_r is None:
            self.prev_frame_r = r_image
        if self.prev_depth is None:
            self.prev_depth = depth
        
        start = time.perf_counter()
        
        prev_l_image_kpts = self.get_keypoints(self.prev_frame_l)
        prev_l_image_track, l_image_track =  self.track_keypoints(self.prev_frame_l, l_image, prev_l_image_kpts)
        
        prev_disp = np.divide(self.disparity.compute(self.prev_frame_l, self.prev_frame_r).astype(np.float32), 16)
        new_disp = np.divide(self.disparity.compute(l_image, r_image).astype(np.float32), 16)
        
        prev_l_image_track, prev_r_image_track, l_image_track, r_image_track = self.calculate_right_qs(prev_l_image_track, l_image_track, prev_disp, new_disp)
        
        transformation = np.concatenate((np.concatenate((np.eye(3), np.zeros((3,1))), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
        if prev_l_image_track.size != 0:
            q1, q2 = self.calc_3d(prev_l_image_track, prev_r_image_track, l_image_track, r_image_track, depth, use_real_depth=False)
            transformation = self.ransac_pose_estimation(prev_l_image_track, l_image_track, q1, q2)
            
        if debug:
            rospy.loginfo(f"pose: {self.cur_pose @ transformation}")
            image = np.zeros_like(self.prev_frame_l)
            image = cv2.drawKeypoints(self.prev_frame_l, prev_l_image_kpts, image, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            image = cv2.drawKeypoints(image, [cv2.KeyPoint(kp[0], kp[1], 10) for kp in l_image_track], image, (255, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("left image", image)
            cv2.waitKey(1000//30)
        
        self.cur_pose = self.cur_pose @ transformation
        
        hom_array = np.array([[0, 0, 0, 1]])
        hom_camera_pose = np.concatenate((self.cur_pose, hom_array), axis=0)
        self.camera_pose_list.append(hom_camera_pose)
        
        estimated_camera_pose_x, estimated_camera_pose_y = self.cur_pose[0, 3], self.cur_pose[2, 3]
        self.estimated_path.append((estimated_camera_pose_x, estimated_camera_pose_y))
        
        self.prev_frame_l = l_image
        self.prev_frame_r = r_image
        self.prev_depth = depth
        
        end = time.perf_counter()
        
        print(f"Current pose: \n{(estimated_camera_pose_x, estimated_camera_pose_y)}")
        print(f"FPS: {1/(end - start)}")
        return estimated_camera_pose_x, estimated_camera_pose_y
        
        