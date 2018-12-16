#!/usr/bin/env python

from keras.models import model_from_json
from std_msgs.msg import Int16
from action_recognition.msg import Matrix
import tensorflow as tf
import numpy as np
import rospkg
import rospy

class ActionRecognition(object):

    def __init__(self, queued_image_topic, action_detected_topic, action_validation_topic, is_learning):

        # define image height, width and number of frames.
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
        self.model._make_predict_function()
        print("Loaded model from disk")
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

        rospy.Subscriber(queued_image_topic, Matrix, self.action_recognition_callback)

        self.learning = is_learning

        if self.learning:
            self.detected_action_pub = rospy.Publisher(action_detected_topic, Int16, queue_size=10)
        else:
            self.detected_action_pub = rospy.Publisher(action_validation_topic, Int16, queue_size=10)

    def action_recognition_callback(self, msg):
        frames = np.array(msg.data).reshape((self.width, self.height, self.depth, 3))
        clip = frames[np.newaxis]
        with self.graph.as_default():
            action_class = self.model.predict_classes(clip, batch_size=1)
            print action_class
        self.detected_action_pub.publish(action_class)

if __name__ == '__main__':
    rospy.init_node('action_recognition_node')
    queued_image_topic = rospy.get_param('~queued_image_topic')
    action_detected_topic = rospy.get_param('~action_detected_topic')
    action_validation_topic = rospy.get_param('~action_validation_topic')
    is_learning = rospy.get_param('~learning_topic')
    action_recognition = ActionRecognition(queued_image_topic, action_detected_topic, action_validation_topic, is_learning)
    rospy.spin()
