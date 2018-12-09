#!/usr/bin/env python

from keras.models import model_from_json
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospkg

# import h5py
import rospy

class ActionRecognition(object):

    def __init__(self, web_cam_topic):
        self.bridge = CvBridge()
        self.frames = []
        self.count = 0
        self.height = 406
        self.width = 306
        self.depth = 10

        rospack = rospkg.RosPack()
        package_path = rospack.get_path('action_recognition')
        print(package_path)
        json_file = open(package_path+'/resources/action_3dcnnmodel.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(package_path+"/resources/action_3dcnnmodel-gpu.hd5")
        print("Loaded model from disk")
        #self.model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.003), metrics=['accuracy'])
        rospy.Subscriber(web_cam_topic, Image, self.action_recognition_callback)

    def action_recognition_callback(data):
        data = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        data = cv2.resize(data, (self.height, self.width))
        self.frames.append(data)
        if self.count < self.depth - 1:
            self.count += 1
            return -1
        else:
            action_class = self.model.predict_classes(self.frames)
            self.count = 0
            self.frames = []
            return action_class

if __name__ == '__main__':
    rospy.init_node('action_recognition_node')
    camera_topic = rospy.get_param('~camera_topic')
    action_recognition = ActionRecognition(camera_topic)
    rospy.spin()
