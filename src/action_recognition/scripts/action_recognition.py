#!/usr/bin/env python

from keras.models import model_from_json
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int16

import rospkg
import rospy

class ActionRecognition(object):

    def __init__(self, web_cam_topic, action_detected_topic):
        self.bridge = CvBridge()
        self.frames = []
        self.count = 0
        self.height = 406
        self.width = 306
        self.depth = 10

        # retrieve the action recognition package path
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('action_recognition')

        # load the model architecture and the model weights
        json_file = open(package_path+'/resources/action_3dcnnmodel.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(package_path+"/resources/action_3dcnnmodel-gpu.hd5")

        print("Loaded model from disk")

        # define the subscriber and publisher
        rospy.Subscriber(web_cam_topic, Image, self.action_recognition_callback)
        self.detected_action_pub = rospy.Publisher(action_detected_topic, Int16, queue_size=10)

    def action_recognition_callback(self, data):
        data = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        data = cv2.resize(data, (self.height, self.width))
        self.frames.append(data)
        if self.count < self.depth - 1:
            self.count += 1
            self.detected_action_pub.publish(-1)
        else:
            action_class = self.model.predict_classes(self.frames)
            self.count = 0
            self.frames = []
            self.detected_action_pub.publish(action_class)

if __name__ == '__main__':
    rospy.init_node('action_recognition_node')
    camera_topic = rospy.get_param('~camera_topic')
    action_detected_topic = rospy.get_param('~action_detected_topic')
    action_recognition = ActionRecognition(camera_topic, action_detected_topic)
    rospy.spin()
