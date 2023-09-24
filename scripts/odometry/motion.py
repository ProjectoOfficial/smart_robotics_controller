import numpy as np
import cv2

def get_3D(image_points, depth, max_depth, intrinsic_matrix):
    points_3D = np.zeros((0, 3))
    outliers = []
    
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    # Extract depth information to build 3D positions
    for indices, (u, v) in enumerate(image_points):
        z = depth[int(v), int(u)]

        # We will not consider depth greater than max_depth
        if z > max_depth:
            outliers.append(indices)
            continue

        # Using z we can find the x,y points in 3D coordinate using the formula
        x = z*(u-cx)/fx
        y = z*(v-cy)/fy

        # Stacking all the 3D (x,y,z) points
        points_3D = np.vstack([points_3D, np.array([x, y, z])])
    return points_3D, outliers


def motion_estimation(matches, firstImage_keypoints, secondImage_keypoints, intrinsic_matrix, distortion_coeff, depth, config):
    """
    Estimating motion of the left camera from sequential imgaes 

    """
    rotation_matrix = np.eye(3)
    translation_vector = np.zeros((3, 1))

    # Only considering keypoints that are matched for two sequential frames
    image1_points = np.float32([firstImage_keypoints[m.queryIdx].pt for m in matches])
    image2_points = np.float32([secondImage_keypoints[m.trainIdx].pt for m in matches])

    points1_3D, outliers1 = get_3D(image1_points, depth, config["max_depth"], intrinsic_matrix)
    
    # Deleting the false depth points
    image1_points = np.delete(image1_points, outliers1, 0)
    image2_points = np.delete(image2_points, outliers1, 0)

    # Apply Ransac Algorithm to remove outliers
    _, rvec, translation_vector, _ = cv2.solvePnPRansac(points1_3D, image2_points, intrinsic_matrix, distortion_coeff)
    if config["motion_refine"]:
        rvec, translation_vector = cv2.solvePnPRefineLM(points1_3D, image2_points, intrinsic_matrix, distortion_coeff, rvec, translation_vector)

    rotation_matrix = cv2.Rodrigues(rvec)[0]

    return rotation_matrix, translation_vector, image1_points, image2_points

