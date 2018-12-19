#!/usr/bin/env python

import rospy

from std_msgs.msg import Int16
import numpy as np
from action_recognition.msg import Matrix

class ActionRepresentationLearning(object):

    def __init__(self, action_recognition_topic):
        self.action_recognized_cb = rospy.Subscriber(action_recognition_topic, Int16, self.action_recognized_cb)
        self.action_sequence = []
        self.first_demo = True

        #self.constraints = np.zeros([8,8])
        self.constraints = np.zeros([1,1])
        #self.current_demo_constraints = np.zeros([8,8])
        self.current_demo_constraints = np.zeros([1,1])
        self.action_map = {}
        self.current_demo_action_map = {}
        self.current_demo_action_count = {}

        self.constraint_publisher = rospy.Publisher("constraint_topic", Matrix, queue_size=10)


    def action_recognized_cb(self, action):
        if action.data == -1:
            if self.first_demo:
                self.constraints = self.build_constraint_matrix()
            else:
                self.current_demo_constraints = self.build_constraint_matrix()
                self.update_constraints()

            self.first_demo = False
            self.action_sequence = []
            self.current_demo_action_map = {}
            self.current_demo_action_count = {}
            print self.constraints
            self.constraint_publisher.publish(self.constraints.flatten())
        else:
            action_name = str(action.data)
            if self.first_demo:
                if action.data in self.current_demo_action_count:
                    action_count = self.current_demo_action_count[action.data]
                    action_name = '{0}_{1}'.format(action.data, action_count)
                    self.action_map[len(self.action_sequence)] = action.data
                    self.current_demo_action_count[action.data] += 1
                else:
                    self.action_map[len(self.action_sequence)] = action_name
                    self.current_demo_action_count[action.data] = 1
            else:
                if action.data in self.current_demo_action_count:
                    action_count = self.current_demo_action_count[action.data]
                    action_name = '{0}_{1}'.format(action.data, action_count)
                    self.current_demo_action_map[len(self.action_sequence)] = action_name
                    self.current_demo_action_count[action.data] += 1
                else:
                    self.current_demo_action_map[len(self.action_sequence)] = action_name
                    self.current_demo_action_count[action.data] = 1
            self.action_sequence.append(action_name)

    def build_constraint_matrix(self):
        constraint_matrix = np.zeros([len(self.action_sequence), len(self.action_sequence)])
        for i in range(len(self.action_sequence)):
            constraint_matrix[i,:i] = 1
        return constraint_matrix

        # -1 is sent only when the demonstration is complete.
        # if action.data == -1:
        #     self.update_constraints()
        #     self.action_sequence = []
        #     self.current_demo_constraints = np.zeros([8,8])
        #     self.first_demo = False
        #     print self.constraints
        #     one_dimension_constraints = np.squeeze(self.constraints.reshape([1,64]))
        #     self.constraint_publisher.publish(one_dimension_constraints)
        #
        # else:
        #     if self.first_demo:
        #         constraint_new = np.zeros([self.constraints.shape + 1])
        #         for previous_actions in self.action_sequence:
        #             constraints_new[action.data, previous_actions] = 1
        #         self.action_map[len(self.action_sequence)] = action.data
        #     else:
        #         for previous_actions in self.action_sequence:
        #             self.current_demo_constraints[action.data, previous_actions] = 1
        #     self.action_sequence.append(action.data)

    def update_constraints(self):
        # print "**************************************************"
        new_action_list = [action for action in self.action_sequence
                           if action not in self.action_map.values()]

        prev_action_count = len(self.action_map.values())
        for action in new_action_list:
            self.action_map[prev_action_count] = action
            prev_action_count += 1

        new_action_count = len(self.action_map.values())
        if new_action_count > 0:
            new_constraint_matrix = np.zeros((new_action_count, new_action_count))
            new_constraint_matrix[0:self.constraints.shape[0], 0:self.constraints.shape[1]] = self.constraints
            self.constraints = new_constraint_matrix

        for i in range(self.current_demo_constraints.shape[0]):
            action1 = self.current_demo_action_map[i]
            for j in range(self.current_demo_constraints.shape[1]):
                action2 = self.current_demo_action_map[j]

                action1_idx = self.get_action_idx(action1)
                action2_idx = self.get_action_idx(action2)
                prev_constraint = self.constraints[action1_idx, action2_idx]
                diff = abs(prev_constraint - self.current_demo_constraints[i,j])
                if diff > 0:
                    if action1 in new_action_list:
                        self.constraints[self.action_map[action1], self.action_map[action2]] = 1
                    else:
                        self.constraints[self.action_map[action1], self.action_map[action2]] = 0

        # conflicting_constraints = self.constraints - self.current_demo_constraints
        # # print self.constraints
        # # print self.current_demo_constraints
        # conflicting_constraints_index = np.where(np.abs(conflicting_constraints) == 1)
        # for diff_index in zip(conflicting_constraints_index[0], conflicting_constraints_index[1]):
        #     self.constraints[diff_index] = 0
        # # print self.constraints

    def get_action_idx(self, action_name):
        for action_idx, action in self.current_demo_action_map.items():
            if action == action_name:
                return action_idx

if __name__ == '__main__':
    rospy.init_node('action_representation_learning_node')
    action_recognition_topic = rospy.get_param('~action_detected_topic')
    action_representation = ActionRepresentationLearning(action_recognition_topic)
    rospy.spin()
