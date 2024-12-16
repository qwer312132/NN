import numpy as np
import read

def distanceToWall(x: float, y: float, angle: float, edge):
    x1, y1, x2, y2 = edge
    angle = angle % (2 * np.pi)   
    if y1 == y2:       
        if (y > y1 and 0 <= angle <= np.pi) or (y < y1 and np.pi <= angle <= 2 * np.pi):
            return 1000000       
        orthogonalDis = abs(y - y1)
        if abs(np.sin(angle)) < 1e-6:  
            return 1000000       
        tempfrontDistance = orthogonalDis / abs(np.sin(angle))
        intersectionX = x + tempfrontDistance * np.cos(angle)      
        if min(x1, x2) <= intersectionX <= max(x1, x2):
            return tempfrontDistance
        return 1000000   
    elif x1 == x2:
        if (x < x1 and np.pi / 2 <= angle <= 3 * np.pi / 2) or (x > x1 and (angle <= np.pi / 2 or angle >= 3 * np.pi / 2)):
            return 1000000        
        orthogonalDis = abs(x - x1)
        if abs(np.cos(angle)) < 1e-6:  
            return 1000000        
        tempfrontDistance = orthogonalDis / abs(np.cos(angle))
        intersectionY = y + tempfrontDistance * np.sin(angle)        
        if min(y1, y2) <= intersectionY <= max(y1, y2):
            return tempfrontDistance
        return 1000000   
    return 1000000


def calculateDistance(x:float, y:float, angle:float):
    startx, starty, startTheta, goalx1, goaly1, goalx2, goaly2, edgex, edgey = read.readEdge()
    edge = []
    for i in range(len(edgex)-1):
        edge.append([edgex[i], edgey[i], edgex[i+1], edgey[i+1]])
    minDistanceFront = 1000000
    minDistanceLeft = 1000000
    minDistanceRight = 1000000
    for line in edge:
        tempDistance = distanceToWall(x, y, angle, line)
        if tempDistance < minDistanceFront:
            minDistanceFront = tempDistance
        tempDistance = distanceToWall(x, y, angle - np.pi/4, line)
        if tempDistance < minDistanceRight:
            minDistanceRight = tempDistance
        tempDistance = distanceToWall(x, y, angle + np.pi/4, line)
        if tempDistance < minDistanceLeft:
            minDistanceLeft = tempDistance
    return minDistanceFront, minDistanceLeft, minDistanceRight

def getAngle(x:float, y:float, angle:float, mlp):
    minDistanceFront, minDistanceLeft, minDistanceRight = calculateDistance(x, y, angle)
    X = np.array([x, y, minDistanceFront, minDistanceRight, minDistanceLeft])
    ret =  mlp.forward(X)[0]
    ret = ret*(40-(-40))+(-40)
    if mlp.inputNum != 5:
        f = open("train4dALL.txt", "a")
        f.write(str("{:.4f}".format(X[2])) + " " + str("{:.4f}".format(X[3]) + " " + str("{:.4f}".format(X[4])) + " " + str("{:.4f}".format(ret)) + "\n"))
        f.close()
    else:
        f = open("train6dALL.txt", "a")
        f.write(str("{:.4f}".format(X[0])) + " " + str("{:.4f}".format(X[1]) + " " + str("{:.4f}".format(X[2])) + " " + str("{:.4f}".format(X[3])) + " " + str("{:.4f}".format(X[4])) + " " + str("{:.4f}".format(ret)) + "\n"))
        f.close()
    return ret

def isStop(x:float, y:float):
    startx, starty, startTheta, goalx1, goaly1, goalx2, goaly2, edgex, edgey = read.readEdge()
    if goalx1 > goalx2:
        goalx1, goalx2 = goalx2, goalx1
    if goaly1 > goaly2:
        goaly1, goaly2 = goaly2, goaly1
    if x >= goalx1 and x <= goalx2 and y >= goaly1 and y <= goaly2:
        return True
    edge = []
    for i in range(len(edgex)-1):
        edge.append([edgex[i], edgey[i], edgex[i+1], edgey[i+1]])

    minDistance:float = 1000000
    for e in edge:     
        minDistance = min(minDistance, distanceToWall(x, y, 0, e))
        minDistance = min(minDistance, distanceToWall(x, y, np.pi/2, e))
        minDistance = min(minDistance, distanceToWall(x, y, np.pi, e))
        minDistance = min(minDistance, distanceToWall(x, y, 3*np.pi/2, e))
    if minDistance < 3:
        return True
    return False

def run(x:float, y:float, angle:float, mlp):

    angle = angle*np.pi/180#degree to radian
    path=[]
    path.append([x, y])
    while True:
        steeringWheelAngle = getAngle(x, y, angle, mlp)
        steeringWheelAngle = steeringWheelAngle*np.pi/180
        x = x + np.cos(angle+steeringWheelAngle) + np.sin(angle)*np.sin(steeringWheelAngle)
        y = y + np.sin(angle+steeringWheelAngle) - np.cos(angle)*np.sin(steeringWheelAngle)
        angle = angle-np.arcsin(2*np.sin(steeringWheelAngle)/6)
        path.append([x, y])
        if(isStop(x, y)):
            break
    return path