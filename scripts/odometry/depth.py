import numpy as np
import cv2


def disparity_mapping(matcher:cv2.StereoSGBM, left_image:np.ndarray, right_image:np.ndarray, rgb:bool=False):
    '''
    Takes a stereo pair of images from the sequence and
    computes the disparity map for the left image.

    :params left_image: image from left camera
    :params right_image: image from right camera

    '''
    # Disparity map
    left_image_disparity_map = matcher.compute(left_image, right_image).astype(np.float32) / 16

    return left_image_disparity_map


def depth_mapping(left_disparity_map, left_intrinsic, left_translation, right_translation, rectified:bool=True):
    '''

    :params left_disparity_map: disparity map of left camera
    :params left_intrinsic: intrinsic matrix for left camera
    :params left_translation: translation vector for left camera
    :params right_translation: translation vector for right camera

    '''
    # Focal length of x axis for left camera
    focal_length = left_intrinsic[0][0]

    # Calculate baseline of stereo pair
    if rectified:
        baseline = right_translation[0] - left_translation[0]
    else:
        baseline = left_translation[0] - right_translation[0]

    # Avoid instability and division by zero
    left_disparity_map[left_disparity_map == 0.0] = 0.1
    left_disparity_map[left_disparity_map == -1.0] = 0.1

    # depth_map = f * b/d
    depth_map = np.ones(left_disparity_map.shape)
    depth_map = (focal_length * baseline) / left_disparity_map

    return depth_map


# Decompose camera projection Matrix
def decomposition(p):
    '''
    :params p: camera projection matrix

    '''
    # Decomposing the projection matrix
    intrinsic_matrix, rotation_matrix, translation_vector, _, _, _, _ = cv2.decomposeProjectionMatrix(p)

    # Scaling and removing the homogenous coordinates
    translation_vector = (translation_vector / translation_vector[3])[:3]

    return intrinsic_matrix, rotation_matrix, translation_vector


def get_stereo_depth_manual(matcher:cv2.StereoSGBM,left_image, right_image, P0, P1, rgb:bool=False):
    '''
    Takes stereo pair of images and returns a depth map for the left camera. 

    :params left_image: image from left camera
    :params right_image: image from right camera
    :params P0: Projection matrix for the left camera
    :params P1: Projection matrix for the right camera

    '''
    # First we compute the disparity map
    disp_map = disparity_mapping(matcher, left_image, right_image, rgb=rgb)

    # Then decompose the left and right camera projection matrices
    l_intrinsic, l_rotation, l_translation = decomposition(P0)
    r_intrinsic, r_rotation, r_translation = decomposition(P1)

    # Calculate depth map for left camera
    depth = depth_mapping(disp_map, l_intrinsic, l_translation, r_translation)

    return depth


def get_stereo_maps(left_camera_info, right_camera_info):
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(left_camera_info["K"], left_camera_info["D"], left_camera_info["R"], left_camera_info["P"][:3,:3], left_camera_info["shape"], cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(right_camera_info["K"], right_camera_info["D"], right_camera_info["R"], right_camera_info["P"][:3,:3], right_camera_info["shape"], cv2.CV_32FC1)
    
    return leftMapX, leftMapY, rightMapX, rightMapY


def get_stereo_depth(leftMapX, leftMapY, rightMapX, rightMapY, matcher, left_frame, right_frame):
    fixedLeft = cv2.remap(left_frame, leftMapX, leftMapY, cv2.INTER_LINEAR)
    fixedRight = cv2.remap(right_frame, rightMapX, rightMapY, cv2.INTER_LINEAR)

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    depth = matcher.compute(grayLeft, grayRight).astype(np.float32)
    return depth