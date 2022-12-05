import cv2 as cv
import numpy as np
import device

# cap_index = 0
cap_index = "http://192.168.137.143:4747/video"

cap = cv.VideoCapture(cap_index)
_,frame = cap.read()
rows, cols, _ = frame.shape

x_medium = int(cols/2)
y_medium = int(rows/2)

center_x = int(cols/2)
center_y = int(rows/2)

# Gimbal start position
yaw_position = 180
pitch_position =  45

YAW_LIMIT_ANGLE = 320
PITCH_UPWARD_LIMIT_ANGLE = 0
PITCH_DOWNWARD_LIMIT_ANGLE = 90

tracked_obj_exist = False

FRAME_HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
FRAME_WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

print(FRAME_HEIGHT, FRAME_WIDTH)
while True:
    _, frame = cap.read()
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # hsv_frame = cv.flip(hsv_frame, 1)
    
    
    # BGR Color
    #black color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])

    # low_red = np.array([255,0,0])
    # high_red = np.array([211,63,93])


    
    red_mask = cv.inRange(hsv_frame, low_red, high_red)
    contours , _ = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x:cv.contourArea(x), reverse=True)
    
    if len(contours) == 0:
        x_medium = int(cols/2)
        y_medium =  int(rows/2)
        tracked_obj_exist = False
    
    for cnt in contours:
        (x,y,w,h) = cv.boundingRect(cnt)
        # cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        x_medium = int((x + (x + w))/2)
        y_medium = int((y + (y + h))/2)
        tracked_obj_exist = True
        break
    
    if tracked_obj_exist:
        cv.line(frame, (x_medium, 0), (x_medium, FRAME_HEIGHT), (0,255,0), 2)
        cv.line(frame,(0,y_medium), (FRAME_WIDTH, y_medium), (0,255,0), 2)

    if cap_index == 0:
        frame = cv.flip(frame, 1)
        
    cv.imshow("Frame", frame)
    cv.imshow("Mask", red_mask)
    
    
    key = cv.waitKey(1)
    if key == 27:
        break
    
    
    
    if x_medium < center_x - 50 :
        yaw_position -=  1
        
    elif x_medium > center_x + 50 :
        yaw_position += 1
        
        
        
    # Y-AXIS: y is value is positive in upper quarter and negative in lower quarter
    # so values in upper quarter are always greater than values center_y
    if y_medium < center_y - 50 :
        pitch_position -= 1
        
    elif y_medium > center_y + 50 :
        pitch_position += 1
        

    print(str(pitch_position) + "    ============    " + str(yaw_position)) 
    
    
    
    
    
    
cap.release()
cap.destroyAllWindows()