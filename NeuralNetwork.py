import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def threshold(x):
    return (x-0.5)*0.5/abs(x-0.5) + 0.5

'''
Multi layer neural network class from scratch
'''
class NeuralNetwork:

    def __init__(self, nin, nout, nhidden, nlayer_nodes, activation, activation_derivative):
        self.nin = nin
        self.nout = nout
        self.nhidden = nhidden
        self.nlayer_bodes = nlayer_nodes
        self.weightIn = 2 * np.random.random((nin, nlayer_nodes[0] if nhidden > 0 else nout)) - 1
        self.weight = []
        self.bias = []
        self.activation = activation
        self.activation_derivative = activation_derivative
        for i in range(nhidden):
            self.weight.append(2 * np.random.random((nlayer_nodes[i], nlayer_nodes[i+1] if i+1 < nhidden else nout)) - 1)
            self.bias.append(2 * np.random.random((1, nlayer_nodes[i])) - 1)
        self.bias.append(2 * np.random.random((1, nout)) - 1)

    def train(self, input, output, rate, epochs):
        for iteration in range(epochs):
            outputi = []
            for i in range(self.nhidden):
                outputi.append(np.dot(outputi[i-1] if i > 0 else input, self.weight[i-1] if i > 0 else self.weightIn) + self.bias[i])
            outputo = np.dot(self.activation(outputi[self.nhidden - 1]), self.weight[self.nhidden-1]) + self.bias[self.nhidden]

            error = output - self.activation(outputo)

            temp = rate * error * self.activation_derivative(outputo)
            self.weight[self.nhidden-1] += np.dot(outputi[self.nhidden-1].T, temp)
            self.bias[self.nhidden] += np.array([np.sum(temp[:, j]) for j in range(self.nout)])
            for i in range(self.nhidden - 1, 0, -1):
                temp = np.dot(temp, self.weight[i].T) * self.activation_derivative(outputi[i])
                self.weight[i-1] += np.dot(outputi[i-1].T, temp)
                self.bias[i] += np.array([np.sum(temp[:, j]) for j in range(self.nlayer_bodes[i])])
            temp = np.dot(temp, self.weight[0].T) * self.activation_derivative(outputi[0])
            self.weightIn += np.dot(input.T, temp)
            self.bias[0] += np.array([np.sum(temp[:, j]) for j in range(self.nlayer_bodes[0])])
        print("\nOutput:")
        print(threshold(self.activation(outputo)))

    def fit(self, input):
        output = []
        for i in range(self.nhidden):
            output.append(self.activation(np.dot(output[i-1] if i > 0 else input, self.weight[i-1] if i > 0 else self.weightIn) + self.bias[i]))
        outputo = self.activation(np.dot(output[self.nhidden - 1], self.weight[self.nhidden-1]) + self.bias[self.nhidden])
        print("\nOutput:")
        print(threshold(outputo))

    def getweights(self):
        print("\nFirst layer weights:")
        print(self.weightIn)
        print(self.bias[0])
        print("\nSecond layer weights:")
        print(self.weight[0])
        print(self.bias[1])


if __name__ == "__main__":
    n = NeuralNetwork(2, 1, 2, [4, 3], sigmoid, dsigmoid)
    training_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_output = np.array([[0, 1, 1, 0]]).T
    n.train(training_input, training_output, 0.1, 10000)
    n.getweights()
    input = np.array([[1, 0]])
    n.fit(input)
