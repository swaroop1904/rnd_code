#!/usr/bin/env python

from keras.models import model_from_json
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import rospy

class ActionRecognition(object):

    def __init__(self, web_cam_topic):
        self.bridge = CvBridge()
        self.frames = []
        self.count = 0
        self.height = 406
        self.width = 306
        self.depth = 10
        json_file = open('action_3dcnnmodel.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("action_3dcnnmodel-gpu.hd5")
        print("Loaded model from disk")
        self.model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.003), metrics=['accuracy'])
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
