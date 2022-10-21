from cv2 import cv2
import numpy as np

capture = cv2.VideoCapture('Bouys_1.mp4')

while True:
    isTrue, frame = capture.read()
   
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    BGR = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #for red
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0R    = cv2.inRange(BGR, lower_red, upper_red)
    lower_red = np.array([170,50,50]) 
    upper_red = np.array([180,255,255]) 
    mask1R    = cv2.inRange(BGR,lower_red,upper_red)
    mask_Red  = mask1R+mask0R

    #for Yellow
    lower_yellow = np.array([23,41,133]) 
    upper_yellow = np.array([40,150,255])
    mask0Y       = cv2.inRange(BGR, lower_yellow, upper_yellow)
    lower_yellow = np.array([20, 100, 100]) 
    upper_yellow = np.array([30, 255, 255]) 
    mask1Y       = cv2.inRange(BGR,lower_yellow,upper_yellow)
    mask_Yellow  = mask1Y+mask0Y

    #for green 
    Lower_green  = np.array([37,0,0])
    Upper_green  = np.array([66,255,255])
    mask2G       = cv2.inRange(hsv,Lower_green,Upper_green)
    mask_Green   = mask2G
   
    kernel   = np.ones((13,13),np.uint8)
    opening_Green  = cv2.morphologyEx(mask_Green, cv2.MORPH_OPEN, kernel)
    dilation_Red   = cv2.dilate(mask_Red,kernel,iterations = 1)
    
    ########################################################################################################
    #Green Buoy Contours
    canny_output_Green= cv2.Canny(opening_Green, 50, 200, None, 3) # Find Canny edges
    contours, _ = cv2.findContours(canny_output_Green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Finding Contours
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    #drawing = np.zeros((canny_output_Green.shape[0], canny_output_Green.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if(area>100):
            cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0, 255, 0), 2)
            cv2.putText(frame, 'GREEN BUOY - Safe', (int(boundRect[i][0]), int(boundRect[i][1])), 1, 1, color=(0, 255, 0),thickness=2)
    ########################################################################################################
    ########################################################################################################
    #Red Buoy Contours
    canny_output_Yellow = cv2.Canny(dilation_Red, 50, 200, None, 3) # Find Canny edges
    contours, _ = cv2.findContours(canny_output_Yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Finding Contours
    contours_poly2 = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly2[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly2[i])

    #drawing = np.zeros((canny_output_Yellow.shape[0], canny_output_Yellow.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if(area>100):
            cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), 
            int(boundRect[i][1]+boundRect[i][3])), (0, 0, 255), 2)
            cv2.putText(frame, 'RED BUOY - Dangerous', (int(boundRect[i][0]), int(boundRect[i][1])), 1, 1, 
            color=(0, 0, 255),thickness=2)
    ########################################################################################################
    ########################################################################################################
    #Yellow Buoy Contours
    canny_output_Red = cv2.Canny(mask_Yellow, 50, 200, None, 3) # Find Canny edges
    contours, _ = cv2.findContours(canny_output_Red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Finding Contours
    contours_poly1 = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly1[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly1[i])
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if(area>50):
            cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,255,255), 2)
            cv2.putText(frame, 'YELLOW BUOY - Working', (int(boundRect[i][0]), int(boundRect[i][1])), 1, 1, color=(0,255,255),thickness=2)
    ########################################################################################################

    result_Green=cv2.bitwise_and(frame,frame,mask=mask_Green)
    result_Red=cv2.bitwise_and(frame,frame,mask=mask_Red)
    result_Yellow=cv2.bitwise_and(frame,frame,mask=mask_Yellow)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    #out = cv2.VideoWriter('Buoy1_Enhancement.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    #cv2.waitKey(0)
    if isTrue:    
        cv2.imshow('Video', frame)
        #out.write(frame)
        if cv2.waitKey(2) & 0xFF==ord('d'):
            break            
    else:
        break
#out.release()
capture.release()
cv2.destroyAllWindows()
