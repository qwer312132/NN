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

def readEdge():
    with open('dataset/edge.txt') as f:
        startLine = f.readline().strip()
        startx, starty, startTheta = map(int,startLine.split(','))
        goalLine1 = f.readline().strip()
        goalx1, goaly1 = map(int,goalLine1.split(','))
        goalLine2 = f.readline().strip()
        goalx2, goaly2 = map(int,goalLine2.split(','))
        edgex = []
        edgey = []
        for line in f:
            x, y = map(int,line.split(','))
            edgex.append(x)
            edgey.append(y)
    return startx, starty, startTheta, goalx1, goaly1, goalx2, goaly2, edgex, edgey
