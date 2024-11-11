import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(output):
    return output * (1 - output)
def relu(x):
    return np.maximum(0, x)
def relu_derivative(output):
    return np.where(output > 0, 1, 0)
class LAYER:
    def __init__(self, inputNum, outputNum, learning_rate):
        self.inputNum = inputNum
        self.outputNum = outputNum
        self.weight = np.random.randn(inputNum + 1, outputNum)
        self.output = None
        self.input = None
        self.isOutputLayer = False
        self.learning_rate = learning_rate

    def forward(self, input):
        self.input = np.insert(input, 0, -1)
        self.output = np.dot(self.input, self.weight)
        self.output = relu(self.output)
        return self.output


    def backward(self, expected):
        if self.isOutputLayer:
            error = expected - self.output
            delta = error * relu_derivative(self.output)
            weight = self.weight[1:]
            weight_update = self.learning_rate * np.outer(self.input, delta)
            self.weight += weight_update
            return np.dot(delta, weight.T)
        else:
            delta = expected * relu_derivative(self.output)
            weight = self.weight[1:]
            weight_update = self.learning_rate * np.outer(self.input, delta)
            self.weight += weight_update
            return np.dot(delta, weight.T)

class MLP:
    def __init__(self, inputNum, outputNum, hiddenNum, learning_rate):
        self.inputNum = inputNum
        self.outputNum = outputNum
        self.hiddenNum = hiddenNum
        self.learning_rate = learning_rate
        self.layers = []
        self.layers.append(LAYER(inputNum, hiddenNum, learning_rate))
        self.layers.append(LAYER(hiddenNum, outputNum, learning_rate))
        self.layers[-1].isOutputLayer = True

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, expected):
        for layer in reversed(self.layers):
            expected = layer.backward(expected)
        return expected