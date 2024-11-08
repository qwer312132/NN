import numpy as np
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
    return edgex, edgey

def distanceToWall(x:float, y:float, angle:float, edge):
    x1, y1, x2, y2 = edge
    if y1 == y2:
        if y > y1 and angle >= 0 and angle <= 180:
            return 1000000
        if y < y1 and angle >= 180 and angle <= 360:
            return 1000000
        orthogonalDis = abs(y - y1)
        tempfrontDistance = orthogonalDis*(1/np.sin(angle))
        intersectionX = x + tempfrontDistance*np.cos(angle)
        if intersectionX >= min(x1, x2) and intersectionX <= max(x1, x2):
            return tempfrontDistance
    elif x1 == x2:
        if x > x1 and angle >= 90 and angle <= 270:
            return 1000000
        if x < x1 and (angle >= 0 and angle <= 90) or (angle >= 270 and angle <= 360):
            return 1000000
        orthogonalDis = abs(x - x1)
        tempfrontDistance = orthogonalDis*(1/np.cos(angle))
        intersectionY = y + tempfrontDistance*np.sin(angle)
        if intersectionY >= min(y1, y2) and intersectionY <= max(y1, y2):
            return tempfrontDistance

def calculateDistance(x:float, y:float, angle:float):
    edgex, edgey = readEdge()
    edge = []
    for i in range(len(edgex)-1):
        edge.append([edgex[i], edgey[i], edgex[i+1], edgey[i+1]])
    edge.append([edgex[-1], edgey[-1], edgex[0], edgey[0]])
    minDistanceFront = 1000000
    minDistanceLeft = 1000000
    minDistanceRight = 1000000
    for line in edge:
        tempDistance = distanceToWall(x, y, angle, line)
        if tempDistance < minDistanceFront:
            minDistanceFront = tempDistance
        tempDistance = distanceToWall(x, y, angle - 45, line)
        if tempDistance < minDistanceRight:
            minDistanceRight = tempDistance
        tempDistance = distanceToWall(x, y, angle + 45, line)
        if tempDistance < minDistanceLeft:
            minDistanceLeft = tempDistance
    return minDistanceFront, minDistanceLeft, minDistanceRight

def run(x:float, y:float, angle:float, mlp):
    minDistanceFront, minDistanceLeft, minDistanceRight = calculateDistance(x, y, angle)
    X = np.array([minDistanceFront, minDistanceLeft, minDistanceRight])
    return mlp.forward(X)