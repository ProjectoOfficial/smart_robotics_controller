import rospy
from std_msgs.msg import Empty, UInt8, Bool

class Deployer:
    def __init__(self) -> None:
        self.deploy = False
        self.probes_deployed = 0
        
        self.num_probes_deployed_topic = "/probe_deployment_unit/probes_dropped"
        self.probe_deploy_topic = "/probe_deployment_unit/drop"
        self.probe_deploy_command_topic = "/probe_deploy_command"
        
        rospy.Subscriber(self.num_probes_deployed_topic, UInt8, self.num_probes_callback)
        rospy.Subscriber(self.probe_deploy_command_topic, Bool, self.command_callback)
        self.pub = rospy.Publisher(self.probe_deploy_topic, Empty, queue_size=10)  # Crea un publisher per il topic


    def command_callback(self, msg):
        try:
            print(msg)
        except Exception as e:
            print(e)
            
            
    def num_probes_callback(self, msg):
        try:
            self.probes_deployed = msg.data
            self.deploy_probe()
        except Exception as e:
            print(e)
            
            
    def deploy_probe(self):
        msg = Empty()
        self.pub.publish(msg)
        rospy.loginfo(f"Deployed probe {self.probes_deployed + 1}")


