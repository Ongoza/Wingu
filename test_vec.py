import numpy as np
import cv2
import math
import random

border_lines = {
    'border1':[[0, 130], [ 416, 130]],
#    'border2':[[100, 0], [ 100, 416]],
    #'border3':[[200, 0], [ 200, 416]]
    }


def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def track_intersection_angle(A,B,id):   
    inter = []
    for key in border_lines:
        C = np.array(border_lines[key][0])
        D = np.array(border_lines[key][1])
        if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):            
            v0 = np.array(B) - np.array(A)
            v1 = np.array(D) - np.array(C)
            angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
            if (angle > 0):
                inter.append(key)
    return inter

def track_intersection_angle3(xy0, xy1, id):    
    frame = np.zeros((416, 416, 3), np.uint8)    
    for key in border_lines:
        s = np.vstack([xy0, xy1, border_lines[key][0], border_lines[key][1]])        # s for stacked
        h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        cv2.line(frame,tuple(border_lines[key][0]),tuple(border_lines[key][1]),(255,255,255),3)
        print("xyz", x, y, z)
        #y1, y2, y3 = pt1[1], pt2[1], pt3[1]
        #slope = (y2 - y1) / (x2 - x1)
        if z != 0:                          # lines are not parallel            
           v0 = np.array(xy1) - np.array(xy0)
           v1 = np.array(border_lines[key][1]) - np.array(border_lines[key][0])
           cv2.line(frame,tuple(xy0),tuple(xy1),(255,0,0),3)            
           cv2.putText(frame, "_"+str(int(x/z))+"_"+str(int(y/z)), (int(x/z), int(y/z)), 0, 1, (0, 255, 0), 1 )
           angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
           if (angle > 0):
               res.append(key)
           else:
               cv2.line(frame,tuple(xy0),tuple(xy1),(0,255,0),3)

    print("draw", id)    
    cv2.imshow("border_"+str(id), frame)
    return res

def get_intersect(a1, a2, b1, b2, id):
    """
    https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])        # s for stacked
    # print('s=',s)
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

def PointsInCircum(r,n=100):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]

if __name__ == "__main__":
    res = []
    #res2 = [[0,0]]
    #print("any", any(res), any(res2))
    frame = np.zeros((416, 416, 3), np.uint8)
    for key in border_lines:
        C = np.array(border_lines[key][0])
        D = np.array(border_lines[key][1])
        cv2.line(frame,tuple(C),tuple(D),(255,255,255),3)
    #print (get_intersect((100, 100), (150, 150), (100, 0), (0, 100), 1))  # parallel  lines
    #print (get_intersect([100, 100], [150, 150], [100, 0], [0, 100], 2))  # parallel  lines
    
    #print (track_intersection_angle(np.array([100, 100]), np.array([240, 10]), 300)) # vertical and horizontal lines
    #print (track_intersection_angle(np.array([150, 150]), np.array([130, 90]), 400))  # another line for fun    
    #print(track_intersection_angle(np.array([100, 100]), np.array([150, 150]), 500))
    n = 100     
    for i in range(n): 
       r =  random.randrange(50, 200)
       A =  np.array([58, 58])
       B =  np.array([int(math.cos(2*math.pi/n*i)*r)+58, int(math.sin(2*math.pi/n*i)*r)+58])
       inter = track_intersection_angle(A, B, i)       
       if inter:
        cv2.line(frame,tuple(A),tuple(B),(0,0,255),1)
       else:
        cv2.line(frame,tuple(A),tuple(B),(255,0,0),1)

       A =  np.array([200, 200])
       B =  np.array([int(math.cos(2*math.pi/n*i)*r)+208, int(math.sin(2*math.pi/n*i)*r)+208])
       inter = track_intersection_angle(A, B, i)
       if inter:
        cv2.line(frame,tuple(A),tuple(B),(0,0,255),1)
       else:
        cv2.line(frame,tuple(A),tuple(B),(0,255,0),1)

    cv2.imshow("border_", frame)    
    key = cv2.waitKey()
    cv2.destroyAllWindows()