#!/usr/bin/env python

import rospy

from std_msgs.msg import Int16
import numpy as np
from action_recognition.msg import Matrix
import csv
import rospkg


class ActionRepresentationLearning(object):

    def __init__(self, action_recognition_topic):
        self.action_recognized_cb = rospy.Subscriber(action_recognition_topic, Int16, self.action_recognized_cb)
        self.action_sequence = []
        self.first_demo = True

        self.constraints = np.zeros([1,1])
        self.current_demo_constraints = np.zeros([1,1])
        self.action_map = {}
        self.current_demo_action_map = {}
        self.current_demo_action_count = {}

        self.constraint_publisher = rospy.Publisher("constraint_topic", Matrix, queue_size=10)

        rospack = rospkg.RosPack()
        package_path = rospack.get_path('action_recognition')
        self.action_no_label_mapping_file = package_path+"/resources/coffee_action_label_file.csv"
        self.build_action_dictionary()

    def build_action_dictionary(self):
        self.action_no_label_mapping = {}

        with open(self.action_no_label_mapping_file) as f_obj:
            reader = csv.reader(f_obj)
            for row in reader:
                self.action_no_label_mapping[row[0]] = row[1]

    def action_recognized_cb(self, action):
        if action.data == -1:
            if self.first_demo:
                self.constraints = self.build_constraint_matrix()
            else:
                self.current_demo_constraints = self.build_constraint_matrix()
                self.update_constraints()
            self.constraint_learned_details()
            self.first_demo = False
            self.action_sequence = []
            self.current_demo_action_map = {}
            self.current_demo_action_count = {}
            self.constraint_publisher.publish(self.constraints.flatten())
        else:
            action_name = str(action.data)
            if self.first_demo:
                if action.data in self.current_demo_action_count:
                    action_count = self.current_demo_action_count[action.data]
                    action_name = '{0}_{1}'.format(action.data, action_count)
                    self.action_map[str(len(self.action_sequence))] = action_name
                    self.current_demo_action_count[action.data] += 1
                else:
                    self.action_map[str(len(self.action_sequence))] = action_name
                    self.current_demo_action_count[action.data] = 1
            else:
                if action.data in self.current_demo_action_count:
                    action_count = self.current_demo_action_count[action.data]
                    action_name = '{0}_{1}'.format(action.data, action_count)
                    self.current_demo_action_map[str(len(self.action_sequence))] = action_name
                    self.current_demo_action_count[action.data] += 1
                else:
                    self.current_demo_action_map[str(len(self.action_sequence))] = action_name
                    self.current_demo_action_count[action.data] = 1
            self.action_sequence.append(action_name)

    def build_constraint_matrix(self):
        constraint_matrix = np.zeros([len(self.action_sequence), len(self.action_sequence)])
        for i in range(len(self.action_sequence)):
            constraint_matrix[i,:i] = 1
        return constraint_matrix

    def update_constraints(self):
        new_action_list = [action for action in self.action_sequence
                           if action not in self.action_map.values()]
        prev_action_count = len(self.action_map.values())
        for action in new_action_list:
            self.action_map[str(prev_action_count)] = action
            prev_action_count += 1

        new_action_count = len(self.action_map.values())
        if new_action_count > 0:
            new_constraint_matrix = np.zeros((new_action_count, new_action_count))
            new_constraint_matrix[0:self.constraints.shape[0], 0:self.constraints.shape[1]] = self.constraints
            self.constraints = new_constraint_matrix

        for i in range(self.current_demo_constraints.shape[0]):
            action1 = self.current_demo_action_map[str(i)]
            for j in range(self.current_demo_constraints.shape[1]):
                action2 = self.current_demo_action_map[str(j)]

                action1_idx = self.get_action_idx(action1)
                action2_idx = self.get_action_idx(action2)
                prev_constraint = self.constraints[action1_idx, action2_idx]
                diff = abs(prev_constraint - self.current_demo_constraints[i,j])
                if diff > 0:
                    if action1 in new_action_list:
                        self.constraints[self.action_map[action1], self.action_map[action2]] = 1
                    else:
                        self.constraints[self.action_map[action1], self.action_map[action2]] = 0

    def get_action_idx(self, action_name):
        for action_idx, action in self.action_map.items():
            if action == action_name:
                return action_idx


    def constraint_learned_details(self):
        print "###############################################"
        print "\n constraints present after from this demo are: \n"
        print "###############################################"
        for row_idx, row in enumerate(self.constraints):
            action_number_2 = self.action_map[str(row_idx)]
            action_name_2 = self.action_no_label_mapping[action_number_2]
            print "***********************************************"
            print "preconditions for {0}:".format(action_name_2)
            for col_idx, col in enumerate(row):
                if col == 0:
                    continue
                else:
                    action_number_1 = self.action_map[str(col_idx)]
                    action_name_1 = self.action_no_label_mapping[action_number_1]
                    print "{0} has to be performed before {1}".format(action_name_1, action_name_2)


if __name__ == '__main__':
    rospy.init_node('action_representation_learning_node')
    action_recognition_topic = rospy.get_param('~action_detected_topic')
    action_representation = ActionRepresentationLearning(action_recognition_topic)
    rospy.spin()
