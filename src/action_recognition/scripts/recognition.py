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

from heapq import heappop
from heapq import heapify
import csv
import copy

class task_data:
    def __init__(self, start_frame, end_frame, task_number):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.task_number = task_number
    def __str__(self):
        return (str(self.start_frame) + ' ' + str(self.end_frame) + ' ' + str(self.task_number))
    def __gt__(self, other):
        return self.start_frame > other.start_frame
    def __lt__(self, other):
        return self.start_frame < other.start_frame
    def __eq__(self, other):
        return self.start_frame == other.start_frame

class ActionRecognition(object):

    def __init__(self, web_cam_topic, action_detected_topic, action_validation_topic, is_learning):
        # define cv bridge to perform conversion from ros image to cv2 image
        self.bridge = CvBridge()

        # define the size of the video (height, width, depth)
        self.count = 0
        self.height = 406
        self.width = 306
        self.depth = 10
        self.frames = np.zeros((self.width, self.height, self.depth, 3))

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

        # used for multithreading purpose
        self.recognizing_action = False
        self.recognition_thread = None

        # depending on whether it is performing learning or validation, publish
        # the detected action to relevant topic
        self.learning = is_learning

        if self.learning:
            self.detected_action_pub = rospy.Publisher(action_detected_topic, Int16, queue_size=10)
        else:
            self.detected_action_pub = rospy.Publisher(action_validation_topic, Int16, queue_size=10)

        # read the csv files to get the action labels for validation the action representation model
        # with the MP-II dataset
        #self.csv_path = package_path+"/use_case/use_case_2/main_files/usecase_2_complete.csv"
        self.csv_path = package_path+"/use_case/use_case_5/main_files/usecase_5_complete.csv"
        #self.csv_path = package_path+"/use_case/use_case_4/main_files/usecase_4_complete.csv"
        #self.csv_path = package_path+"/use_case/use_case_5/main_files/usecase_5_complete.csv"

    def action_recognition_callback(self, data):

        ## data from camera feed

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

        # data from MP-II dataset
        #self.publish_action_labels()


    def action_recognition(self, frames):
        clip = frames[np.newaxis]
        print "calling action recognition"
        with self.graph.as_default():
            action_class = self.model.predict_classes(clip)
            print action_class
        self.detected_action_pub.publish(action_class)
        self.recognizing_action = False
        self.recognition_thread = None

    def publish_action_labels(self):
        with open(self.csv_path) as f_obj:
            reader = csv.reader(f_obj)
            for row in reader:
                self.detected_action_pub.publish(int(row[0]))

if __name__ == '__main__':
    rospy.init_node('action_recognition_node')
    camera_topic = rospy.get_param('~camera_topic')
    action_detected_topic = rospy.get_param('~action_detected_topic')
    action_validation_topic = rospy.get_param('~action_validation_topic')
    is_learning = rospy.get_param('~learning_topic')
    action_recognition = ActionRecognition(camera_topic, action_detected_topic, action_validation_topic, is_learning)
    rospy.spin()
