def readFile(file):
    with open('dataset/'+file+'.txt') as f:
        X = []
        y = []
        for line in f:
            l = line.split()
            x = []
            for i in range(len(l)):
                if i < len(l) - 1:
                    x.append(float(l[i]))
                else:
                    y.append(float(l[i]))
            X.append(x)
    return X, y
