import numpy as np
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Twist

class Navigation:
    def __init__(self) -> None:
        self.cur_goal = None
        self.cur_pose = None
        
        self.linear_speed = 0.8
        self.angular_speed = 0.8
        
        
    def get_rotation_z(self, theta:float):
        R_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
        return R_z

    
    def get_deltas(self, goal:np.ndarray):
        """find the angular and linear displacement in order to move from the current
        orientation and position to the final position (x,y) in space

        Args:
            goal (np.ndarray): (x, y) position expressing the goal to reach 
            
        Output:
            delta_orientation: a rotation matrix that express how much the robot has to be rotated in order to face the goal in front
            delta_traslation: traslation in order to get to the goal
        """
        delta_orientation = np.eye(3)
        delta_translation = np.zeros(3)
        
        if self.cur_pose is not None:
            linear_displacement = goal - self.cur_pose[:2, 3]
            delta_translation = np.array([linear_displacement[0], linear_displacement[1], 0])

            target_orientation = np.arctan2(linear_displacement[1], linear_displacement[0])
            delta_orientation = self.get_rotation_z(target_orientation)
            delta_orientation = delta_orientation @ np.linalg.inv(self.cur_pose[:3, :3])

        return delta_orientation, delta_translation
    
    
    def set_goal(self, goal:np.ndarray):
        delta_o, delta_t = self.get_deltas(goal)
        
        delta_T = np.eye(4)
        delta_T[:3, :3] = delta_o
        delta_T[:3, 3] = delta_t
        self.cur_goal = delta_T
        
        
    def reset_goal(self):
        self.cur_goal = None
        
    
    def reset_pose(self):
        self.cur_pose = np.eye(4)

    
    def update_cur_pose(self, pose:np.ndarray):
        self.cur_pose = pose
        
        
    def update_speed(self, vl:float=0.8, va:float=0.8):
        self.linear_speed = vl
        self.angular_speed = va
        
        
    def get_twist(self, lx:float, ly:float, lz:float, ax:float, ay:float, az:float) -> Twist:
        twist = Twist()
        twist.linear.x = lx * self.linear_speed
        twist.linear.y = ly * self.linear_speed
        twist.linear.z = lz * self.linear_speed
        twist.angular.x = ax * self.angular_speed
        twist.angular.y = ay * self.angular_speed
        twist.angular.z = az * self.angular_speed
        
        return twist
    
    
        