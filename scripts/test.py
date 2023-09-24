#!/usr/bin/env python3
import rospy
import cv2
import os
from navigation.navigation import Navigation
from scipy.spatial.transform import Rotation as R
import numpy as np

def test_aruco():
        
    ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}
    
    path = os.path.join("/home", "daniel", "Pictures", "2.png")
    img = cv2.imread(path)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    for key in ARUCO_DICT.keys():
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[key])
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(img)
        
        if rejectedCandidates is not None and len(rejectedCandidates):
            print(f"rejected: {key}")
        
        if (markerIds is not None and len(markerIds)) or (markerCorners is not None and len(markerCorners)):
            print("found")
            print(key)
            break

def test_navigation():
    cur_traslation = np.array([0,0,0])
    cur_orientation = np.array([90,0,90])
    cur_orientation = R.from_euler("xyz", cur_orientation, True).as_matrix()
    goal = np.array([5,5])
    
    pose = np.eye(4)
    pose[:3,:3] = cur_orientation
    pose[:3 ,3] = cur_traslation
    
    navigator = Navigation()
    navigator.update_cur_pose(pose)
    rot, tras = navigator.get_deltas(goal)
    print(R.from_matrix(rot).as_euler('xyz', degrees=True))
    print(tras)
    

if __name__ == '__main__':    
    test_navigation()