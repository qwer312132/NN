import numpy as np
import read

def distanceToWall(x:float, y:float, angle:float, edge):
    x1, y1, x2, y2 = edge
    angle = (angle+2*np.pi)%(2*np.pi)
    # print(x1, y1, x2, y2)
    if y1 == y2:
        if y > y1 and angle >= 0 and angle <= np.pi:
            return 1000000
        if y < y1 and angle >= np.pi and angle <= 2*np.pi:
            return 1000000
        orthogonalDis = abs(y - y1)
        if(np.sin(angle) == 0):
            return 1000000
        tempfrontDistance = abs(orthogonalDis*(1/np.sin(angle)))
        intersectionX = x + tempfrontDistance*np.cos(angle)
        if intersectionX >= min(x1, x2) and intersectionX <= max(x1, x2):
            return tempfrontDistance
        return 1000000
    elif x1 == x2:
        if x > x1 and angle >= np.pi/2 and angle <= 3*np.pi/2:
            return 1000000
        if x < x1 and ((angle >= 0 and angle <= np.pi/2) or (angle >= 3*np.pi/2 and angle <= 2*np.pi)):
            return 1000000
        orthogonalDis = abs(x - x1)
        if(np.cos(angle) == 0):
            return 1000000
        tempfrontDistance = abs(orthogonalDis*(1/np.cos(angle)))
        intersectionY = y + tempfrontDistance*np.sin(angle)
        if intersectionY >= min(y1, y2) and intersectionY <= max(y1, y2):
            return tempfrontDistance
        return 1000000

def calculateDistance(x:float, y:float, angle:float):
    startx, starty, startTheta, goalx1, goaly1, goalx2, goaly2, edgex, edgey = read.readEdge()
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
        tempDistance = distanceToWall(x, y, angle - np.pi/4, line)
        if tempDistance < minDistanceRight:
            minDistanceRight = tempDistance
        tempDistance = distanceToWall(x, y, angle + np.pi/4, line)
        if tempDistance < minDistanceLeft:
            minDistanceLeft = tempDistance
    return minDistanceFront, minDistanceLeft, minDistanceRight

def getAngle(x:float, y:float, angle:float, mlp):
    minDistanceFront, minDistanceLeft, minDistanceRight = calculateDistance(x, y, angle)
    X = np.array([minDistanceFront, minDistanceLeft, minDistanceRight])
    ret =  mlp.forward(X)[0]
    ret = ret*(40-(-40))+(-40)
    return ret

def isStop(x:float, y:float):
    startx, starty, startTheta, goalx1, goaly1, goalx2, goaly2, edgex, edgey = read.readEdge()
    edge = []
    for i in range(len(edgex)-1):
        edge.append([edgex[i], edgey[i], edgex[i+1], edgey[i+1]])
    edge.append([edgex[-1], edgey[-1], edgex[0], edgey[0]])
    minDistance:float = 1000000
    for e in edge:     
        minDistance = min(minDistance, distanceToWall(x, y, 0, e))
        minDistance = min(minDistance, distanceToWall(x, y, 90, e))
        minDistance = min(minDistance, distanceToWall(x, y, 180, e))
        minDistance = min(minDistance, distanceToWall(x, y, 270, e))
    print(minDistance)
    if minDistance <= 3:
        return True
    return False

def run(x:float, y:float, angle:float, mlp):
    angle = (angle+360)%360
    angle = angle*np.pi/180
    path=[]
    path.append([x, y])
    while True:
        steeringWheelAngle = getAngle(x, y, angle, mlp)
        print(steeringWheelAngle)
        phi = angle-1/np.cos(2*np.sin(steeringWheelAngle)/6)
        x = path[-1][0] + np.cos(phi+steeringWheelAngle) + np.sin(phi)*np.sin(steeringWheelAngle)
        y = path[-1][1] + np.sin(phi+steeringWheelAngle) - np.cos(phi)*np.sin(steeringWheelAngle)
        path.append([x, y])
        if(isStop(x, y)):
            break
    return path