

class ActionRepresentationValidation(object):
    def __init__(self, action_validation_topic):
        self.action_validation_cb = rospy.Subscriber(action_validation_topic, Int16, self.action_validation_cb)
        self.action_sequence = []
        self.current_constraints = np.zeros([8,8])

    def action_validation_cb(self, action):
        if action.data == -1:
            self.check_constraints()
            self.action_sequence = []
        else:
            for previous_actions in self.action_sequence:
                self.current_constraints[action.data, previous_actions] = 1
            self.action_sequence.append(action.data)

    def check_constraints():
        conflicting_constraints = self.constraints - self.current_constraints
        conflicting_constraints_index = np.where(conflicting_constraints) == 1)
        for diff_index in zip(conflicting_constraints_index[0], conflicting_constraints_index[1]):
            print "{1} has to be performed before {0}".format(diff_index[0], diff_index[1])
