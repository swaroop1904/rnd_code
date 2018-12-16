#!/usr/bin/env python

from cv_bridge import CvBridge
from action_recognition.msg import Matrix
from std_msgs.msg import Int16
from sensor_msgs.msg import Image

import cv2
import rospy
import numpy as np
import time

class ImageQueuing(object):
    def __init__(self, web_cam_topic, image_sequence_topic):
        self.bridge = CvBridge()
        self.count = 0
        self.height = 406
        self.width = 306
        self.depth = 10
        self.flag = True
        self.frames = np.zeros((self.width, self.height, self.depth, 3))

        rospy.Subscriber(web_cam_topic, Image, self.image_queuing_callback)

        rospy.Subscriber(action_validation_topic, Int16, self.action_detected)
        rospy.Subscriber(action_recognition_topic, Int16, self.action_detected)

        self.image_sequence = rospy.Publisher(image_sequence_topic, Matrix, queue_size=5)


    def image_queuing_callback(self, msg):
        data = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        data = cv2.resize(data, (self.height, self.width))
        data = np.asarray(data)
        data = data.astype(float)
        self.frames[:,:,self.count,:] = data
        if self.count < self.depth - 1 and self.flag:
            print "generating clip"
            self.count += 1
        elif self.flag:
            print "invoking action recognition"
            self.count = 0
            self.flag = False
            start = time.time()
            compressed_image_sequence = np.squeeze(self.frames.reshape([1, self.height*self.width*self.depth*3]))
            self.image_sequence.publish(compressed_image_sequence)
            print time.time() - start
        else:
            print "waiting for action recognition to complete"
            # time.sleep(5)
            # self.count = 0
            return

    def action_detected(self, msg):
        self.flag = True

if __name__ == "__main__":
    rospy.init_node('image_queuing_node')
    web_cam_topic = rospy.get_param('~camera_topic')
    image_sequence_topic = rospy.get_param('~queued_image_topic')
    action_recognition_topic = rospy.get_param('~action_detected_topic')
    action_validation_topic = rospy.get_param('~action_validation_topic')

    image_queuing = ImageQueuing(web_cam_topic, image_sequence_topic)
    rospy.spin()
