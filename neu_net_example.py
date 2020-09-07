''' this code is the first attempt of an neural Network, that predicts the mnist dataset
'''
# pylint: disable=too-many-locals
import numpy as np

from neural_network import NeuNet


def main():
    ''' main function
    '''
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    # learning rate is 0.1
    learning_rate = 0.1

    # create new instance of neural network
    neu_net = NeuNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # load the mnist training data CSV file into a list
    training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train neuNet

    # go trough all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        neu_net.train(inputs, targets)


    # loads the mnist test data CSV file into a list
    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # test the neuNet

    # scorecard for how well the network performs, initially empty
    scorecard = []
    # go trough all the records in the test data set
    for record in test_data_list:
        # split the record by ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = neu_net.query(inputs)
        # the index of the highest value corresponds to the label
        label = np.argmax(outputs)
        # append correct or incorrect to list
        if label == correct_label:
            # netw answer matches correct answer, and add 1 to scorecard
            scorecard.append(1)
        else:
            # netw answer doesn't matches correct answer, add 0 to scorecard
            scorecard.append(0)

    # calculate the performance score, the fraction of correct answers
    scorecard_array = np.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)


if __name__ == "__main__":
    main()
