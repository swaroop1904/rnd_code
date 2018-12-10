#!/usr/bin/env python

import rospy

from std_msgs.msg import Int16
import numpy as np
from action_recognition.msg import Matrix

class ActionRepresentationLearning(object):

    def __init__(self, action_recognition_topic):
        self.constraints = np.zeros([8,8])
        self.action_recognized_cb = rospy.Subscriber(action_recognition_topic, Int16, self.action_recognized_cb)
        self.action_sequence = []
        self.first_demo = True
        self.current_demo_constraints = np.zeros([8,8])

        self.constraint_publisher = rospy.Publisher("constraint_topic", Matrix, queue_size=10)


    def action_recognized_cb(self, action):
        # -1 is sent only when the demonstration is complete.
        if action.data == -1:
            self.update_constraints()
            self.action_sequence = []
            self.current_demo_constraints = np.zeros([8,8])
            self.first_demo = False
            print self.constraints
            one_dimension_constraints = np.squeeze(self.constraints.reshape([1,64]))
            self.constraint_publisher.publish(one_dimension_constraints)

        else:
            if self.first_demo:
                for previous_actions in self.action_sequence:
                    self.constraints[action.data, previous_actions] = 1
            else:
                for previous_actions in self.action_sequence:
                    self.current_demo_constraints[action.data, previous_actions] = 1
            self.action_sequence.append(action.data)

    def update_constraints(self):
        if not self.first_demo:
            conflicting_constraints = self.constraints - self.current_demo_constraints
            conflicting_constraints_index = np.where(np.abs(conflicting_constraints) == 1)
            for diff_index in zip(conflicting_constraints_index[0], conflicting_constraints_index[1]):
                self.constraints[diff_index] = 0

if __name__ == '__main__':
    rospy.init_node('action_representation_learning_node')
    action_recognition_topic = rospy.get_param('~action_detected_topic')
    action_representation = ActionRepresentationLearning(action_recognition_topic)
    rospy.spin()
