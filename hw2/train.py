from matplotlib.figure import Figure
import numpy as np
import MLP
def train(X, y, lr, epoch):
    inputSize = X.shape[1]
    outputSize = 1
    hiddenSize = inputSize
    mlp = MLP.MLP(inputSize, outputSize, hiddenSize, lr)
    new_y = []
    max_y = max(y)
    min_y = min(y)
    for i in y:
        new_y.append((i-min_y)/(max_y-min_y))
    counter = 0
    while True:
        counter += 1
        print("epoch",counter)
        if counter > epoch:
            break
        for i in range(len(y)):
            x_ = X[i]
            mlp.forward(np.array(x_))
            mlp.backward(np.array([new_y[i]]))
    return mlp