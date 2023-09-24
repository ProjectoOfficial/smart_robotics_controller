#!/usr/bin/env python3

from copy import deepcopy
import rospy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry as OdometryMSG
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import cv2
import threading, signal

from aruco.aruco_utils import detect_aruco, draw_arucopose
from odometry.odometry import Odometry
from aruco.aruco_detector import ArucoDetector


class VisionController:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.data = dict()
        
        self.left_image_topic = "/zed2/left_raw/image_raw_color"
        self.right_image_topic = "/zed2/right_raw/image_raw_color"
        self.depth_image_topic = "/zed2/depth/depth_registered"
        
        self.left_camera_info_topic = "/zed2/left/camera_info"
        self.right_camera_info_topic = "/zed2/right/camera_info"
        self.left_camera_info = dict()
        self.right_camera_info = dict()
        
        rospy.Subscriber(self.left_image_topic, Image, self.image_callback)
        rospy.Subscriber(self.right_image_topic, Image, self.image_callback)
        rospy.Subscriber(self.depth_image_topic, Image, self.image_callback)
        
        rospy.Subscriber(self.left_camera_info_topic, CameraInfo, self.parameters_callback)
        rospy.Subscriber(self.right_camera_info_topic, CameraInfo, self.parameters_callback)
        
        self.publisher = rospy.Publisher('/odometry', OdometryMSG, queue_size=1)
        
        self.processing_thread = threading.Thread(target=self._process)
        self.stop_event = threading.Event()
        
        self.odometry = None
        self.landmark_detector = ArucoDetector()
        self.path = []
        

    def parameters_callback(self, msg: CameraInfo) -> None:
        parameters = dict()
        
        try:
            parameters["D"] = np.array(msg.D)
            parameters["K"] = np.array(msg.K).reshape(3,3)
            parameters["P"] = np.array(msg.P).reshape(3,4)
            parameters["R"] = np.array(msg.R).reshape(3,3)
            parameters["shape"] = (msg.width, msg.height)
            if msg._connection_header["topic"] == self.left_camera_info_topic:
                self.left_camera_info = parameters
            elif msg._connection_header["topic"] == self.right_camera_info_topic:
                self.right_camera_info = parameters
            
        except Exception as e:
            rospy.logerr(e)
            
            
    def image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if self.left_image_topic == msg._connection_header["topic"]:
                self.data["zed2_image_left"] = image
            elif self.right_image_topic == msg._connection_header["topic"]:
                self.data["zed2_image_right"] = image
            elif self.depth_image_topic == msg._connection_header["topic"]:
                self.data["zed2_image_depth"] = image
            
        except CvBridgeError as e:
            rospy.logerr(e)
        
        
    def is_image(self, data: np.ndarray):
        return isinstance(data, np.ndarray) and data.size > 0
    
    
    def _show(self):
        if "zed2_image_left" in self.data and self.is_image(self.data["zed2_image_left"]):
            markers = detect_aruco(self.data["zed2_image_left"])
            img = draw_arucopose(self.data["zed2_image_left"], markers)
            cv2.imshow("left", img)        
        
        if "zed2_image_right" in self.data and self.is_image(self.data["zed2_image_right"]):
            cv2.imshow("right", self.data["zed2_image_right"])       
        
        if "zed2_image_depth" in self.data and self.is_image(self.data["zed2_image_depth"]):
            cv2.imshow("depth", self.data["zed2_image_depth"])
                           
        cv2.waitKey(1000//30)
    
    
    def _process(self):
        while not self.stop_event.is_set():
            if self.data is not None and self.data != {}:
                if self.odometry is None:
                    self.create_odometry()
                else:
                    if "zed2_image_left" in self.data and self.is_image(self.data["zed2_image_left"]) and \
                        "zed2_image_right" in self.data and self.is_image(self.data["zed2_image_right"]) and \
                        "zed2_image_depth" in self.data and self.is_image(self.data["zed2_image_depth"]):
                        data = deepcopy(self.data)
                        
                        # objs = self.landmark_detector.process(data["zed2_image_left"], data["zed2_image_right"], \
                        #     data["zed2_image_depth"], debug=False)
                        
                        position, orientation = self.odometry.process(data["zed2_image_left"], data["zed2_image_right"], \
                            data["zed2_image_depth"], debug=False)
                        
                        try:
                            odometry_msg = OdometryMSG()
                            odometry_msg.header.stamp = rospy.Time.now() 
                            odometry_msg.header.frame_id = "odom"
                            odometry_msg.child_frame_id = "base_link"
                            odometry_msg.pose.pose.position.x = position[0]
                            odometry_msg.pose.pose.position.y = position[1]
                            odometry_msg.pose.pose.position.z = position[1]
                            odometry_msg.pose.pose.orientation.x = orientation[0]
                            odometry_msg.pose.pose.orientation.y = orientation[1]
                            odometry_msg.pose.pose.orientation.z = orientation[2]
                            odometry_msg.pose.pose.orientation.w = orientation[3]
                            self.publisher.publish(odometry_msg)
                            
                        except Exception as e:
                            print(e)
                        
                        
    def start_processing(self) -> None:
        self.processing_thread.start()
                
                
    def stop_processing(self) -> None:
        self.stop_event.set()
        self.processing_thread.join()
        
        
    def create_odometry(self) -> None:
        while len(self.left_camera_info) == 0 or len(self.right_camera_info) == 0:
            pass
        
        config=dict({
            "rgb": True,
            "rectified": True,
            "detector": 'sift',
            "distance_threshold": 0.38,
            "max_depth": 100,
            "stereo_depth": True,
            "matcher": "flann",
            "motion_refine": False
        })
        
        self.odometry = Odometry(self.left_camera_info, self.right_camera_info, config)


if __name__ == '__main__':
    rospy.init_node("process_image_node")
    rospy.loginfo("STARTING ZED 2 CAMERA MANAGER NODE ...")
    rospy.Rate(1000)
    
    camera_manager = VisionController()
    camera_manager.start_processing()
    signal.signal(signal.SIGINT, camera_manager.stop_processing)
    
    while not rospy.is_shutdown():
        rospy.spin()
        