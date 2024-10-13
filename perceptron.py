from matplotlib.figure import Figure
import random
import numpy as np
import MLP
def calAccuracy(x, y, c, mlp):
    correct = 0
    for i in range(len(c)):
        x_ = [-1, x[i], y[i]]
        if mlp.forward(np.array(x_)) >= 0.5 and c[i] == 1:
            correct += 1
        elif mlp.forward(np.array(x_)) < 0.5 and c[i] == 0:
            correct += 1
    return correct/len(c)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))
def perceptron(x, y, c):
    mlp = MLP.MLP(2, 1, 2, 0.1)
    counter = 0
    #normalize c
    new_c = []
    max_c = max(c)
    min_c = min(c)
    for i in c:
        new_c.append((i-min_c)/(max_c-min_c))
    while True:
        counter += 1
        # print(counter)
        if counter > 1000:
            break
        for i in range(len(x)):
            x_ = [x[i], y[i]]
            mlp.forward(np.array(x_))
            mlp.backward(np.array([new_c[i]]))

        if calAccuracy(x, y, c, mlp) == 1:
            break
    return mlp
def create_plot(x, y, c, filename):
    mlp = perceptron(x, y, c)
    acc = calAccuracy(x, y, c, mlp)
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot()
    #find max and min x and y
    max_x = max(x)
    min_x = min(x)
    max_y = max(y)
    min_y = min(y)
    X = np.linspace(min_x, max_x, 100)
    Y = np.linspace(min_y, max_y, 100)
    for i in X:
        for j in Y:
            if mlp.forward([-1, i, j]) >= 0.5:
                ax.plot(i, j, 'bo')
            else:
                ax.plot(i, j, 'ro')
    for i in range(len(c)):
        if c[i] == 1:
            ax.plot(x[i], y[i], 'ro', markeredgecolor='black', markeredgewidth=1.5)
        else:
            ax.plot(x[i], y[i], 'bo', markeredgecolor='black', markeredgewidth=1.5)
    ax.set_title("Filename: " + filename + "\nAccuracy: " + str(acc*100) + "%")
    return fig