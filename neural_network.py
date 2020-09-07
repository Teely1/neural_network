""" neural network with functions train and query
"""
# pylint: disable=unnecessary-lambda
# pylint: disable=no-member
import numpy as np
# scipy.special for sigmoid function = expit()
import scipy.special

class NeuNet:
    ''' neural network class definition
    '''

    def __init__(self, inputnodes, hiddennotes, outputnodes, learningrate):
        # sets nbr of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennotes
        self.onodes = outputnodes

        # link weight matrics, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.l_rate = learningrate

        # activation func is sigmoid func
        self.activation_func = lambda x: scipy.special.expit(x)


    def train(self, inputs_list, targets_list):
        ''' trains the neural network
        '''
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging form hidden layer
        hidden_outputs = self.activation_func(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_func(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.l_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.l_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))


    def query(self, input_list):
        ''' query NeuNet
        '''
        # convert inputs list to 2d array
        inputs = np.array(input_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_func(hidden_inputs)

        #calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_func(final_inputs)

        return final_outputs
