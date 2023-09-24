import rospy

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError

from navigation.gmapper import GMapping
from navigation.navigation import Navigation
from scipy.spatial.transform import Rotation as R


class Navigator:
    def __init__(self, x_range=10, y_range=10):
        self.x_range = x_range
        self.y_range = y_range
        self.fig, self.ax = plt.subplots()

        self.ax.set_xlim(-self.x_range // 4, self.x_range // 4)
        self.ax.set_ylim(-self.y_range // 4, self.y_range // 4)
        self.position_scatter = self.ax.scatter([], [], c='b', marker='o')
        self.obstacle_scatter = self.ax.scatter([], [], c='r', marker='o')

        self.x_data, self.y_data = [], []
        self.x_data_obstacle, self.y_data_obstacle = [], []
        

        self.custom_odometry_topic = "/odometry"
        self.gazebo_odometry_topic = "/gazebo/controllers/diff_drive/odom"
        self.depth_image_topic = "/zed2/depth/depth_registered"

        rospy.Subscriber(self.custom_odometry_topic, Odometry, self.odometry_callback)
        rospy.Subscriber(self.depth_image_topic, Image, self.image_callback)

        self.depth = None
        self.bridge = CvBridge()
        self.gmapper = GMapping(x_size=x_range, y_size=y_range, resolution=0.2)
        self.navigation = Navigation()

        self.cur_pose = np.eye(4)
        self.is_cur_pose_updated = False
        self.cur_objective = None

        self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size=1)


    def update(self) -> None:
        if len(self.x_data) and len(self.x_data) == len(self.y_data):
            self.position_scatter.set_offsets(list(zip(self.x_data, self.y_data)))
        if len(self.x_data_obstacle) and len(self.x_data_obstacle) == len(self.y_data_obstacle):
            self.obstacle_scatter.set_offsets(list(zip(self.x_data_obstacle, self.y_data_obstacle)))
        plt.pause(0.01)


    def plot_point(self, pt:np.ndarray, color:str):
        plt.scatter(pt[0], pt[1], color=color, marker='o', label='Punto')
        plt.pause(0.01)


    def show(self) -> None:
        plt.show(block=False)


    def image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth = image
            
            threshold = 0.4
            obstacles = self.gmapper.find_obstacle(self.cur_pose, self.depth, threshold, False)
            if len(obstacles):
                self.x_data_obstacle.append(self.cur_pose[0, 3] + threshold)
                self.y_data_obstacle.append(self.cur_pose[1, 3])
            
        except CvBridgeError as e:
            rospy.logerr(e)


    def read_odometry(self, msg: Odometry):
        position = msg.pose.pose.position
        position = np.array([position.x, position.y, position.z])
        orientation = msg.pose.pose.orientation
        orientation = np.array([orientation.x, orientation.y, orientation.z, orientation.w])

        return position, orientation


    def odometry_callback(self, msg: Odometry) -> None:
        try:
            position, orientation = self.read_odometry(msg)

            if len(position) and len(orientation):
                orientation = R.from_quat(orientation).as_matrix()

                pose = np.eye(4)
                pose[:3, 3] = np.array(position)
                pose[:3, :3] = orientation

                x, y, z = position
                self.cur_pose = pose
                self.is_cur_pose_updated = True

                self.x_data.append(x)
                self.y_data.append(y)

                if self.gmapper.is_valid_position(x, y):
                    self.gmapper.update_position(x, y, z, True, True, False)

        except Exception as e:
            rospy.logerr("Error parsing odometry data: %s", str(e))


    def move_update(self, x, y, z, th):
        self.x = x
        self.y = y
        self.z = z
        self.th = th


    def stop(self) -> None:
        twist = self.navigation.get_twist(0,0,0,0,0,0)
        self.publisher.publish(twist)


    def rotate(self, delta_o:np.ndarray, orientation_threshold:float=5):
        if self.is_cur_pose_updated:
            self.is_cur_pose_updated = False

            self.navigation.update_cur_pose(self.cur_pose)

            theta = R.from_matrix(delta_o[:3, :3]).as_euler("xyz", True)[-1]
            print(theta)

            speed = np.abs(theta)/50 if theta > 10 else 0.5
            self.navigation.update_speed(va=speed)

            direction = 1
            if -theta < 0:
                direction = -1

            twist = self.navigation.get_twist(0,0,0,0,0,direction)
            self.publisher.publish(twist)
            self.update()
            rospy.sleep(0.65)

            return np.abs(theta) < orientation_threshold
        return False


    def translate(self, delta_t:np.ndarray, goal:np.ndarray, translation_episodes:int):
        twist = self.navigation.get_twist(5,0,0,0,0,0)
        self.publisher.publish(twist)
        self.update()
        translation_episodes += 1

        return np.all(np.linalg.norm(delta_t) < 0.1), translation_episodes


    def move(self, goal):
        self.navigation.update_cur_pose(self.cur_pose)

        or_ok = False
        tr_ok = False
        translation_episodes = 0
        orientation_threshold = 5
        while np.linalg.norm(self.cur_pose[:2, 3] - goal) > 0.3:
            delta_o, delta_t = self.navigation.get_deltas(goal)
            if translation_episodes == 30:
                translation_episodes = 0
                or_ok = False

            if not or_ok:
                or_ok = self.rotate(delta_o, orientation_threshold)
            if or_ok:
                tr_ok, translation_episodes = self.translate(delta_t, goal, translation_episodes)
            if tr_ok and or_ok:
                break


    def wait_for_pose(self):
        while (self.cur_pose is None or np.all(self.cur_pose == np.eye(4))) and not self.is_cur_pose_updated:
            pass

        self.navigation.cur_pose = self.cur_pose
        self.plot_point(self.cur_pose[:2, 3], "green")


if __name__ == '__main__':
    rospy.init_node("navigator_node")
    rospy.loginfo("NAVIGATOR NODE ...")
    rospy.Rate(1000)

    try:
        goal = np.array([-2,1.5])
        navigator = Navigator(x_range=30, y_range=30)
        navigator.update()
        navigator.wait_for_pose()
        navigator.plot_point(goal, "orange")
        navigator.show()

        navigator.move(goal)
        while not rospy.is_shutdown():
            navigator.update()
            rospy.spin()
    except Exception as e:
        print(e)
    finally:
        plt.close("all")
