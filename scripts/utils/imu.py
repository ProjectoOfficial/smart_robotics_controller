#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Imu
from collections import deque
from filterpy.kalman import KalmanFilter

from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

class IMU:
    def __init__(self, memory_size:int=100, filter_window_size=5) -> None:
        self._n = memory_size
        self._g = 9.81 # m/s2
        self.filter_window_size = filter_window_size
        
        self.times = deque(maxlen=memory_size + 1)
        self.linear_accelerations = deque(maxlen=memory_size)
        self.orientations = deque(maxlen=memory_size)
        self.angular_velocities = deque(maxlen=memory_size)
        self.velocity = np.zeros((3, ))
        self.velocity_history = deque(maxlen=filter_window_size)
        
        self.mutex = False

        rospy.Subscriber("/imu/data_raw", Imu, self.imu_callback)
        
        self.kf = KalmanFilter(dim_x=3, dim_z=3)     
        self.kf.F = np.eye(3)
        self.kf.Q = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])
        self.kf.H = np.eye(3)
        self.kf.R = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
        self.kf.x = np.zeros(3)
        self.kf.P = np.array([[0.0001, 0.0, 0.0], [0.0, 0.0001, 0.0], [0.0, 0.0, 0.0001]])


    def imu_callback(self, data: Imu):
        try:
            if len(self.times) and not self.mutex:
                lin_acc = data.linear_acceleration
                orient = data.orientation
                ang_acc = data.angular_velocity

                self.linear_accelerations.append([lin_acc.x, lin_acc.y, lin_acc.z]) # x, y, z
                self.orientations.append([orient.x, orient.y, orient.z, orient.w]) # x y z w
                self.angular_velocities.append([ang_acc.x, ang_acc.y, ang_acc.z])
                
            self.times.append(rospy.get_time())
        except Exception as e:
            rospy.logerr(e)
            
            
    def reset(self) -> None:
        self.linear_accelerations.clear()
        self.orientations.clear()
        self.times.clear()
            
            
    def quaternion_to_matrix(self, quaternion:tuple) -> np.ndarray:
        return R.from_quat(quaternion).as_matrix()
    
    
    def _calc_gravity_compensation(self, accelerations: deque, orientations: deque):
        rotation_matrices = np.array([self.quaternion_to_matrix(q) for q in orientations])
        
        for i in range(len(accelerations)):
            # accelerations[i] = np.array([(rotation_matrices[i] @ accelerations[i])])
            # self.kf.predict()
            # self.kf.update(accelerations[i] + np.array([0, 0, -self._g])) # sum because we consider -g
            accelerations[i] = accelerations[i] + np.array([0, 0, -self._g]) #self.kf.x 
        return accelerations
    
    
    def get_avg_values(self):
        self.mutex = True
        
        linear_accelerations, angular_velocities, times, orientations = None, None, None, None
        if len(self.times) == len(self.linear_accelerations) + 1:
            times = deepcopy(self.times)
            linear_accelerations = np.array(deepcopy(self.linear_accelerations))
            orientations = np.array(deepcopy(self.orientations))
            angular_velocities = np.array(deepcopy(self.angular_velocities))
            
            linear_accelerations = np.expand_dims(linear_accelerations.mean(axis=0), axis=0)
            orientations = np.expand_dims(orientations.mean(axis=0), axis=0)
            angular_velocities = np.expand_dims(angular_velocities.mean(axis=0), axis=0)
            times = np.array([times[0], times[-1]])
            
            linear_accelerations = self._calc_gravity_compensation(linear_accelerations, orientations)
            # self.reset()
        self.mutex = False
                
        return linear_accelerations.squeeze(0), angular_velocities.squeeze(0), times, orientations.squeeze(0)
                

    def get_velocity(self, filter: bool=False) -> np.ndarray:
        self.mutex = True
        if len(self.times) == len(self.linear_accelerations) + 1:
            accelerations = deepcopy(self.linear_accelerations)
            orientations = deepcopy(self.orientations)
            times = deepcopy(self.times)
            
            time_deltas = np.expand_dims(np.diff(np.array(times)), 1)
            accelerations = self._calc_gravity_compensation(accelerations, orientations)
            self.velocity += (accelerations * time_deltas).sum(0)
            
            self.velocity_history.append(self.velocity.copy())
            if filter and len(self.velocity_history) >= self.filter_window_size:
                self.velocity = np.mean(np.array(self.velocity_history), axis=0)
            
        self.mutex = False
            
        return np.round(accelerations[-1], 3)
    

if __name__ == "__main__":
    rospy.init_node("imu_node")
    rospy.loginfo("STARTING IMU NODE ...")
    rospy.Rate(1000)
    
    imu = IMU(1)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(8, 6))
    xs = []
    vx = []
    vy = []
    vz = []
    
    limit = 10  
        
    # This function is called periodically from FuncAnimation
    def animate(i, xs, yx, yy, yz):
        v = imu.get_velocity()

        xs.append(xs[-1] + 1 if len(xs) else 0)
        yx.append(v[0])
        yy.append(v[1])
        yz.append(v[2])

        xs = xs[-20:]
        yx = yx[-20:]
        yy = yy[-20:]
        yz = yz[-20:]

        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        
        ax1.set_ylim(-limit, limit)
        ax2.set_ylim(-limit, limit)
        ax3.set_ylim(-limit, limit)
        ax4.set_ylim(-limit, limit)
        
        ax1.plot(xs, yx, label='Velocity X')
        ax2.plot(xs, yy, label='Velocity Y')
        ax3.plot(xs, yz, label='Velocity Z')
        ax4.plot(xs, np.sqrt(np.array(yx)**2 + np.array(yy)**2 + np.array(yz)**2), label='Velocity vector')

        ax1.set_ylabel('Velocity (m/s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax4.set_ylabel('Velocity (m/s)')
        
        ax1.set_title('Velocity along X-axis')
        ax2.set_title('Velocity along Y-axis')
        ax3.set_title('Velocity along Z-axis')
        ax4.set_title('Velocity norm')

    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig, animate, fargs=(xs, vx, vy, vz), interval=0.1)
    plt.show()

    
    while not rospy.is_shutdown():
        rospy.spin()