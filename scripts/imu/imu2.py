import rospy
from sensor_msgs.msg import Imu
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
from utils import *
from ahrs import AHRS

class IMU:
    def __init__(self, config:dict) -> None:
        self.config = config
        
        self.linear_acceleration = np.zeros(3)
        self.orientation = np.zeros(4)
        self.angular_velocity = np.zeros(3)
        self.linear_velocity = np.zeros_like(self.linear_acceleration)
        
        self.current_time = 0
        self.previous_time = 0
        self.real_sample_time = 0
        
        self.mutex = False
        self.data_available = False
        
        rospy.Subscriber("/imu/data_raw", Imu, self.imu_callback)
        self.AHRS = AHRS(SamplePeriod=self.config["SamplePeriod"], Kp=self.config["Kp"], KpInit=self.config["KpInit"])
    
    
    def imu_callback(self, data: Imu):
        try:
            if not self.mutex:
                self.linear_acceleration = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z])
                self.orientation = np.array([data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w])
                self.angular_velocity = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])
                self.previous_time = self.current_time
                self.current_time = data.header.stamp.nsecs / 1e9
                self.data_available = True
                
        except Exception as e:
            rospy.logerr(e)
    
    
    def lock(self):
        self.mutex = True
        
        
    def unlock(self):
        self.mutex = False
    
    
    def process(self):
        self.lock()
        if self.data_available:
            self.data_available = False
            
            acc_mag = [np.linalg.norm(self.linear_acceleration)]
            
            b, a = butter(1, (2 * self.config["hp_filter_cutoff"]) / (1 / self.config["SamplePeriod"]), 'high')
            acc_mag_filt = filtfilt(b, a, acc_mag, padlen=0)
            
            acc_mag_filt = np.abs(acc_mag_filt)
            b, a = butter(1, (2 * self.config["lp_filter_cutoff"]) / (1 / self.config["SamplePeriod"]), 'low')
            acc_mag_filt = filtfilt(b, a, acc_mag_filt, padlen=0)
            
            stationary = acc_mag_filt < self.config["stationary_threshold"]
            
            if stationary:
                self.AHRS.Kp = 0.5
            else:
                self.AHRS.Kp = 0
                
            orientation_euler = R.from_quat(self.orientation).as_euler("xyz", False)
            self.AHRS.UpdateIMU(orientation_euler.tolist(), self.linear_acceleration.tolist())
            
            acc = quaternRotate(np.expand_dims(self.linear_acceleration, 0), quaternConj(np.expand_dims(self.AHRS.Quaternion, 0)))
            self.linear_acceleration = np.squeeze(acc[:, 2] - 9.81, 0)
            
            if stationary:
                self.linear_velocity = np.zeros(3)
            else:
                self.linear_velocity += self.linear_acceleration * (self.current_time - self.previous_time)
            print(acc_mag_filt)
            
        self.unlock()


if __name__ == "__main__":
    config = {
            "SamplePeriod": 1/1000,
            "hp_filter_cutoff": 0.001,
            "lp_filter_cutoff": 5,
            "stationary_threshold": 0.05,
            "Kp": 1,
            "KpInit": 1,
            }
    
    rospy.init_node("imutest_node")
    rospy.loginfo("STARTING IMU TEST NODE ...")
    rospy.Rate(1/config["SamplePeriod"])
    
    imu = IMU(config)
    
    while not rospy.is_shutdown():
        imu.process()
    
    