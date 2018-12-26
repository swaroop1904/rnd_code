#!/usr/bin/env python

import rospy

from std_msgs.msg import Int16
import numpy as np
from action_recognition.msg import Matrix
import csv
import rospkg
from matplotlib.colors import ListedColormap, NoNorm
import matplotlib.pyplot as plt
import networkx as nx


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
        #self.action_no_label_mapping_file = package_path+"/use_case/coffee_use_case_1/coffee_action_label_file.csv"
        #self.action_no_label_mapping_file = package_path+"/use_case/use_case_2/main_files/usecase_2_action_mapping.csv"
        self.action_no_label_mapping_file = package_path+"/use_case/use_case_5/main_files/usecase_5_action_mapping.csv"
        # self.action_no_label_mapping_file = package_path+"/use_case/use_case_4/main_files/usecase_4_action_mapping.csv"
        # self.action_no_label_mapping_file = package_path+"/use_case/use_case_5/main_files/usecase_5_action_mapping.csv"
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
            #self.constraint_learned_details()
            self.draw_constraint_plot()
            constraint_graph = self.create_constraint_graph()
            self.draw_constraint_graph(constraint_graph)
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

        previous_action_list = [action for action in self.action_map.values()
                                if action not in self.action_sequence]

        desired_size = self.current_demo_constraints.shape[0] + len(previous_action_list)
        new_current_constraint_matrix = np.zeros((desired_size, desired_size))

        for i in range(self.current_demo_constraints.shape[0]):
            action1 = self.current_demo_action_map[str(i)]
            for j in range(self.current_demo_constraints.shape[1]):
                action2 = self.current_demo_action_map[str(j)]
                action1_idx = self.get_action_idx(action1)
                action2_idx = self.get_action_idx(action2)
                new_current_constraint_matrix[int(action1_idx)][int(action2_idx)] = self.current_demo_constraints[i][j]
        self.current_demo_constraints = new_current_constraint_matrix

        conflicting_constraints = self.constraints - self.current_demo_constraints
        conflicting_constraints_index = np.where(np.abs(conflicting_constraints) == 1)
        for diff_index in zip(conflicting_constraints_index[0], conflicting_constraints_index[1]):
            action1 = self.action_map[str(diff_index[0])]
            if action1 not in previous_action_list:
                self.constraints[diff_index] = 0
            if action1 in new_action_list:
                self.constraints[diff_index] = 1

        # for i in range(self.current_demo_constraints.shape[0]):
        #     action1 = self.current_demo_action_map[str(i)]
        #     for j in range(self.current_demo_constraints.shape[1]):
        #         action2 = self.current_demo_action_map[str(j)]
        #
        #         action1_idx = self.get_action_idx(action1)
        #         action2_idx = self.get_action_idx(action2)
        #         prev_constraint = self.constraints[action1_idx, action2_idx]
        #         diff = abs(prev_constraint - self.current_demo_constraints[i,j])
        #         if diff > 0:
        #             if action1 in new_action_list:
        #                 self.constraints[self.action_map[action1], self.action_map[action2]] = 1
        #             else:
        #                 self.constraints[self.action_map[action1], self.action_map[action2]] = 0

    def get_action_idx(self, action_name):
        for action_idx, action in self.action_map.items():
            if action == action_name:
                return action_idx


    def constraint_learned_details(self):
        print "###############################################"
        print "\n constraints present after from this demo are: \n"
        print "###############################################"
        for row_idx, row in enumerate(self.constraints):
            action_number_2 = self.action_map[str(row_idx)].split('_')[0]
            action_name_2 = self.action_no_label_mapping[action_number_2]
            print "***********************************************"
            print "Constraints present for {0}:".format(action_name_2)
            for col_idx, col in enumerate(row):
                if col == 0:
                    continue
                else:
                    action_number_1 = self.action_map[str(col_idx)].split('_')[0]
                    action_name_1 = self.action_no_label_mapping[action_number_1]
                    print "{0} has to be performed before {1}".format(action_name_1, action_name_2)


    def draw_constraint_plot(self):
        cmap = ListedColormap(['#00FF8C','#E0E0E0'])

        action_name_list = [self.action_no_label_mapping[self.action_map[str(row_id)].split('_')[0]]
                            for row_id in xrange(len(self.action_map))]

        fig = plt.figure()
        fig.set_figheight(5)
        fig.set_figwidth(12)
        ax = fig.add_subplot(1,1,1)
        ax.set_yticks(xrange(len(self.action_map)))
        ax.set_xticks(xrange(len(self.action_map)))
        ax.set_yticklabels(action_name_list, va='bottom', minor=False)
        ax.set_xticklabels(action_name_list, rotation='vertical', ha='left', minor=False)

        im = ax.pcolor(self.constraints,cmap=cmap,norm=NoNorm())

        plt.title('Constraint plot')
        plt.xlabel('Actions')
        plt.ylabel('Constraint details')
        plt.grid(True)

        cbar = fig.colorbar(im, orientation='vertical')
        cbar.ax.set_yticklabels(['not required', 'required'])
        plt.show()


    def create_constraint_graph(self):
        constraint_graph = nx.DiGraph()
        for i, action_constraints in enumerate(self.constraints):
            action_number_2 = self.action_map[str(i)].split('_')[0]
            action_name_2 = self.action_no_label_mapping[action_number_2]
            for j, constraint in enumerate(action_constraints):
                action_number_1 = self.action_map[str(j)].split('_')[0]
                action_name_1 = self.action_no_label_mapping[action_number_1]
                if constraint > 0:
                    constraint_graph.add_edge(action_name_1, action_name_2)
        return constraint_graph

    def draw_constraint_graph(self, constraint_graph, node_colour='g', edge_colour='b'):
        '''

        Keyword arguments:
        constraint_graph -- a networkx.DiGraph instance
        node_colour -- colour used for the nodes (can be either a single character
                       or a list of colours for each node)
        edge_colour -- colour used for the edges (can be either a single character
                       or a list of colours for each edge)

        '''
        positions = nx.circular_layout(constraint_graph)
        pos_higher = {}
        for k, v in positions.items():
            pos_higher[k] = (v[0], v[1]+0.1)

        nx.draw(constraint_graph, pos=positions,
                node_color=node_colour, edge_color=edge_colour,
                arrowsize=30)
        nx.draw_networkx_labels(constraint_graph, pos_higher, font_size=16, \
                                font_weight='bold')
        plt.show()

if __name__ == '__main__':
    rospy.init_node('action_representation_learning_node')
    action_recognition_topic = rospy.get_param('~action_detected_topic')
    action_representation = ActionRepresentationLearning(action_recognition_topic)
    rospy.spin()
