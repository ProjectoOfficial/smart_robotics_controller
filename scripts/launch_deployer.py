import rospy
from probe.deployer import Deployer

if __name__ == "__main__":
    rospy.init_node("deployer_node")
    rospy.loginfo("STARTING DEPLOYER NODE ...")
    rospy.Rate(1000)
    
    deployer = Deployer()
    
    while not rospy.is_shutdown():
        rospy.spin()