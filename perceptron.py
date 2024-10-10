from matplotlib.figure import Figure
import random
import numpy as np
def calAccuracy(x, y, c, w):
    correct = 0
    for i in range(len(c)):
        x_ = [-1, x[i], y[i]]
        if (c[i] == 1 and sum([a*b for a, b in zip(w, x_)]) <= 0.5) or (c[i] == 2 and sum([a*b for a, b in zip(w, x_)]) > 0.5):
            correct += 1
    return correct/len(c)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))
def perceptron(x, y, c):
    lr = 0.1
    w = [random.random() for i in range(3)]
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
        for i in range(len(new_c)):
            x_ = [-1, x[i], y[i]]
            output = sum([a*b for a, b in zip(w, x_)])
            output = sigmoid(output)
            error = new_c[i] - output
            w = [a + lr*error*sigmoid_derivative(output)*b for a, b in zip(w, x_)]

        if calAccuracy(x, y, c, w) == 1:
            break
    return w
def create_plot(x, y, c, filename):
    w = perceptron(x, y, c)
    acc = calAccuracy(x, y, c, w)
    #-w[0] + w[1]*x + w[2]*y = 0
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
            if -w[0] + w[1]*i + w[2]*j >= 0.5:
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