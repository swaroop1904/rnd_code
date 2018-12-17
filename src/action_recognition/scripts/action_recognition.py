#!/usr/bin/env python

from keras.models import model_from_json
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int16
import tensorflow as tf
import cv2

import rospkg
import rospy

import numpy as np

import threading

class ActionRecognition(object):

    def __init__(self, web_cam_topic, action_detected_topic, action_validation_topic, is_learning):
        self.bridge = CvBridge()
        self.frames = []
        self.count = 0
        self.height = 406
        self.width = 306
        self.depth = 10
        self.learning = is_learning
        # retrieve the action recognition package path
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('action_recognition')

        # load the model architecture and the model weights
        json_file = open(package_path+'/resources/action_3dcnnmodel.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(package_path+"/resources/action_3dcnnmodel-gpu.hd5")
        self.model._make_predict_function()
        print("Loaded model from disk")
        self.graph = tf.get_default_graph()
        # define the subscriber and publisher
        rospy.Subscriber(web_cam_topic, Image, self.action_recognition_callback)

        self.frames = np.zeros((self.width, self.height, self.depth, 3))
        self.recognizing_action = False
        self.recognition_thread = None

        if self.learning:
            self.detected_action_pub = rospy.Publisher(action_detected_topic, Int16, queue_size=10)
        else:
            self.detected_action_pub = rospy.Publisher(action_validation_topic, Int16, queue_size=10)

    def action_recognition_callback(self, data):
        # if self.learning:
        #     array_1 = np.array([0,1,2,3,4,5,6,7,-1])
        #     array_2 = np.array([2,1,0,3,5,6,4,7,-1])
        #     array_3 = np.array([4,5,0,1,2,3,6,7,-1])
        #     array_4 = np.array([4,0,1,2,3,6,5,7,-1])
        #     for val in array_1:
        #         self.detected_action_pub.publish(val)
        #     for val in array_2:
        #         self.detected_action_pub.publish(val)
        #     for val in array_3:
        #         self.detected_action_pub.publish(val)
        #     for val in array_4:
        #         self.detected_action_pub.publish(val)
        # else:
        #     array_5 = np.array([7,3,1,2,0,4,5,6,-1])
        #     for val in array_5:
        #         self.detected_action_pub.publish(val)
        if self.recognizing_action:
            return
        data = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        data = cv2.resize(data, (self.height, self.width))
        data = np.asarray(data)
        data = data.astype(float)
        self.frames[:,:,self.count,:] = data
        if self.count < self.depth - 1:
            print "generating frames"
            self.count += 1
        else:
            self.recognizing_action = True
            self.recognition_thread = threading.Thread(target=self.action_recognition,
                                                       args=(np.array(self.frames),))
            self.recognition_thread.start()
            self.count = 0

    def action_recognition(self, frames):
        clip = frames[np.newaxis]
        print "calling action recognition"
        with self.graph.as_default():
            action_class = self.model.predict_classes(clip)
            print action_class
        self.detected_action_pub.publish(action_class)
        self.recognizing_action = False
        self.recognition_thread = None


if __name__ == '__main__':
    rospy.init_node('action_recognition_node')
    camera_topic = rospy.get_param('~camera_topic')
    action_detected_topic = rospy.get_param('~action_detected_topic')
    action_validation_topic = rospy.get_param('~action_validation_topic')
    is_learning = rospy.get_param('~learning_topic')
    action_recognition = ActionRecognition(camera_topic, action_detected_topic, action_validation_topic, is_learning)
    rospy.spin()
