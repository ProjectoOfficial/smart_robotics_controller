import numpy as np
import cv2

ARUCO_SIZE = 0.05

def detect_aruco(img):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(img)

    markers = dict()
    if markerIds is not None and markerIds.size > 0:
        keys = [marker.item() for marker in markerIds]
        values = [corner for corner in markerCorners]
        for i, key in enumerate(keys):
            markers[key] = values[i]
    
    return markers


def draw_arucopose(image: np.ndarray, markers: dict) -> np.ndarray:
    if len(markers) < 0:
        return image
    
    for (markerID, markerCorner) in markers.items():
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        
        cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
    
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
        cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.circle(image, (image.shape[1]//2, image.shape[0]//2), 4, (0, 0, 255), -1)
    return image


def get3dpoint(marker: np.ndarray, camera_matrix: np.ndarray, dist_coeff: np.ndarray, verbose: bool=False, id: int=0) -> np.ndarray:
    point_2d_pixel = np.mean(marker[0], axis=0)
    point_2d_meters = point_2d_pixel
    point_2d_meters[0] = (point_2d_pixel[0] - camera_matrix[0, 2]) * ARUCO_SIZE / camera_matrix[0, 0]
    point_2d_meters[1] = (point_2d_pixel[1] - camera_matrix[1, 2]) * ARUCO_SIZE / camera_matrix[1, 1]
    
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker, ARUCO_SIZE, camera_matrix, dist_coeff)
    rot_mat, _ = cv2.Rodrigues(rvec)
    translation_mat = np.zeros((4, 4))
    translation_mat[:3, :3] = rot_mat
    translation_mat[:3, 3] = tvec.squeeze()
    translation_mat[3, 3] = 1.0

    inv_translation_mat = np.linalg.inv(translation_mat)

    point_3d_homogeneous = np.array([point_2d_meters[0], point_2d_meters[1], 1, 1])
    point_3d_homogeneous = point_3d_homogeneous.reshape((4, 1))
    point_3d = np.dot(inv_translation_mat, point_3d_homogeneous)[:3, 0]
    
    if verbose:
        print(f"{id}: - center 3d coordinates: {point_3d} - center 2d coordinates: {point_2d_pixel}")
    return point_3d