from matplotlib.figure import Figure
def perceptron(x, y, c):
    lr = 0.1
    w = [0, 0, 1]
    counter = 0
    while True:
        counter += 1
        if counter > 100:
            break
        flag = 0
        for j in range(len(c)):
            x_ = [-1, x[j], y[j]]
            if c[j] == 1 and sum([a*b for a, b in zip(w, x_)]) < 0:
                w = [a + lr*b for a, b in zip(w, x_)]
                flag = 1
            elif c[j] == 2 and sum([a*b for a, b in zip(w, x_)]) >= 0:
                w = [a - lr*b for a, b in zip(w, x_)]
                flag = 1
        if flag == 0:
            break
    return w
def create_plot(x, y, c):
    w = perceptron(x, y, c)
    #-w[0] + w[1]*x + w[2]*y = 0
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot()
    for i in range(len(c)):
        if c[i] == 1:
            ax.plot(x[i], y[i], 'ro', markeredgecolor='black', markeredgewidth=1.5)
        else:
            ax.plot(x[i], y[i], 'bo', markeredgecolor='black', markeredgewidth=1.5)
    if w[2] != 0 and w[1] != 0:
        ax.plot([0,w[0]/w[1]], [w[0]/w[2],0])
    elif w[2] == 0:
        ax.plot([0,0], [0,1])
    else:
        ax.plot([0,1], [0,0])
    return fig