from matplotlib.figure import Figure
import numpy as np
import MLP
from sklearn.model_selection import train_test_split
def calAccuracy(X, c, mlp):
    correct = 0
    for i in range(len(c)):
        x_ = [X[i][0], X[i][1]]
        if mlp.forward(np.array(x_)) >= 0.5 and c[i] == 1:
            correct += 1
        elif mlp.forward(np.array(x_)) < 0.5 and c[i] == 0:
            correct += 1
    return correct/len(c)
def perceptron(X, c, lr, epoch, accuracy):
    mlp = MLP.MLP(2, 1, 2, lr)
    counter = 0
    #normalize c
    new_c = []
    max_c = max(c)
    min_c = min(c)
    for i in c:
        new_c.append((i-min_c)/(max_c-min_c))
    while True:
        counter += 1
        print("epoch",counter)
        if counter > epoch:
            break
        for i in range(len(c)):
            x_ = [X[i][0], X[i][1]]
            mlp.forward(np.array(x_))
            mlp.backward(np.array([new_c[i]]))

        if calAccuracy(X, new_c, mlp) >= accuracy/100:
            break
    return mlp
def create_plot(X, y, lr, epoch, accuracy):
    if(len(X) == 4):
        X_train = X
        y_train = y
        X_test = X
        y_test = y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    new_c = []
    max_c = max(y_train)
    min_c = min(y_train)
    for i in y_train:
        new_c.append((i-min_c)/(max_c-min_c))
    #split into training and testing
    
    mlp = perceptron(X_train, new_c, lr, epoch, accuracy)
    train_acc = calAccuracy(X_train, new_c, mlp)
    weight = mlp.layers[-1].weight
    # print(weight)
    new_c = []
    max_c = max(y_test)
    min_c = min(y_test)
    for i in y_test:
        new_c.append((i-min_c)/(max_c-min_c))
    test_acc = calAccuracy(X_test, new_c, mlp)
    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot()
    #find max and min x and y
    max_x = max(X_test, key=lambda x: x[0])[0]
    min_x = min(X_test, key=lambda x: x[0])[0]
    max_y = max(X_test, key=lambda x: x[1])[1]
    min_y = min(X_test, key=lambda x: x[1])[1]
    Xlim = np.linspace(min_x-((max_x-min_x)/2), max_x+(((max_x-min_x)/2)), 100)
    Ylim = np.linspace(min_y-((max_y-min_y)/2), max_y+((max_y-min_y)/2), 100)
    for i in Xlim:
        for j in Ylim:
            if mlp.forward([i, j]) >= 0.5:
                ax.plot(i, j, 'bo')
            else:
                ax.plot(i, j, 'ro')
    for i in range(len(new_c)):
        if new_c[i] >= 0.5:
            ax.plot(X_test[i][0], X_test[i][1], 'bo', markeredgecolor='black', markeredgewidth=1.5)
        else:
            ax.plot(X_test[i][0], X_test[i][1], 'ro', markeredgecolor='black', markeredgewidth=1.5)
    ax.set_title("Train accuracy: " + f'{train_acc*100:.2f}' + "%" + " Test accuracy: " + f'{test_acc*100:.2f}' + "%" + "\nweight: " + str(weight.T))
    return fig