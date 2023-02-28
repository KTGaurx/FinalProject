import numpy as np
import time
import cv2


ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}





def aruco_display(corners, ids, rejected, image):
    #global cx0,cy0,cx1,cy1,cx2,cy2,cx3,cy3
    if len(corners) > 0:
        
		
        ids = ids.flatten()
        
		
        for (markerCorner0, markerID0) in zip(corners[0], ids):
            #markerID0 = markerID[0]
            corners0 = markerCorner0.reshape((4, 2))
            (topLeft0, topRight0, bottomRight0, bottomLeft0) = corners0
			
            topRight0 = (int(topRight0[0]), int(topRight0[1]))
            bottomRight0 = (int(bottomRight0[0]), int(bottomRight0[1]))
            bottomLeft0 = (int(bottomLeft0[0]), int(bottomLeft0[1]))
            topLeft0 = (int(topLeft0[0]), int(topLeft0[1]))

            cv2.line(image, topLeft0, topRight0, (0, 255, 0), 2)
            cv2.line(image, topRight0, bottomRight0, (0, 255, 0), 2)
            cv2.line(image, bottomRight0, bottomLeft0, (0, 255, 0), 2)
            cv2.line(image, bottomLeft0, topLeft0, (0, 255, 0), 2)
			
            cX0 = int((topLeft0[0] + bottomRight0[0]) / 2.0)
            cY0 = int((topLeft0[1] + bottomRight0[1]) / 2.0)
            cv2.circle(image, (cX0, cY0), 4, (0, 0, 255), -1)
            
			
            cv2.putText(image, str(markerID0),(topLeft0[0], topLeft0[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID0))
            print(cX0,cY0,ids[0])
            #print(6 in ids)
        #if 6 in ids:
            #print("True")
        #if 10 in ids:
           # print("Trueza")
    if len(corners) > 1:
    
        for (markerCorner1, markerID1) in zip(corners[1], ids):
			
            corners1 = markerCorner1.reshape((4, 2))
            (topLeft1, topRight1, bottomRight1, bottomLeft1) = corners1
			
            topRight1 = (int(topRight1[0]), int(topRight1[1]))
            bottomRight1 = (int(bottomRight1[0]), int(bottomRight1[1]))
            bottomLeft1 = (int(bottomLeft1[0]), int(bottomLeft1[1]))
            topLeft1 = (int(topLeft1[0]), int(topLeft1[1]))

            cv2.line(image, topLeft1, topRight1, (0, 255, 0), 2)
            cv2.line(image, topRight1, bottomRight1, (0, 255, 0), 2)
            cv2.line(image, bottomRight1, bottomLeft1, (0, 255, 0), 2)
            cv2.line(image, bottomLeft1, topLeft1, (0, 255, 0), 2)
			
            cX1 = int((topLeft1[0] + bottomRight1[0]) / 2.0)
            cY1 = int((topLeft1[1] + bottomRight1[1]) / 2.0)
            cv2.circle(image, (cX1, cY1), 4, (0, 0, 255), -1)
            
			
    if len(corners) > 2:
            
        for (markerCorner2, markerID) in zip(corners[2], ids):
			
            corners2 = markerCorner2.reshape((4, 2))
            (topLeft2, topRight2, bottomRight2, bottomLeft2) = corners2
			
            topRight2 = (int(topRight2[0]), int(topRight2[1]))
            bottomRight2 = (int(bottomRight2[0]), int(bottomRight2[1]))
            bottomLeft2 = (int(bottomLeft2[0]), int(bottomLeft2[1]))
            topLeft2 = (int(topLeft2[0]), int(topLeft2[1]))

            cv2.line(image, topLeft2, topRight2, (0, 255, 0), 2)
            cv2.line(image, topRight2, bottomRight2, (0, 255, 0), 2)
            cv2.line(image, bottomRight2, bottomLeft2, (0, 255, 0), 2)
            cv2.line(image, bottomLeft2, topLeft2, (0, 255, 0), 2)
			
            cX2 = int((topLeft2[0] + bottomRight2[0]) / 2.0)
            cY2 = int((topLeft2[1] + bottomRight2[1]) / 2.0)
            cv2.circle(image, (cX2, cY2), 4, (0, 0, 255), -1)
            
    if len(corners) > 3:
        for (markerCorner3, markerID) in zip(corners[3], ids):
			
            corners3 = markerCorner3.reshape((4, 2))
            (topLeft3, topRight3, bottomRight3, bottomLeft3) = corners3
			
            topRight3 = (int(topRight3[0]), int(topRight3[1]))
            bottomRight3 = (int(bottomRight3[0]), int(bottomRight3[1]))
            bottomLeft3 = (int(bottomLeft3[0]), int(bottomLeft3[1]))
            topLeft3 = (int(topLeft3[0]), int(topLeft3[1]))

            cv2.line(image, topLeft3, topRight3, (0, 255, 0), 2)
            cv2.line(image, topRight3, bottomRight3, (0, 255, 0), 2)
            cv2.line(image, bottomRight3, bottomLeft3, (0, 255, 0), 2)
            cv2.line(image, bottomLeft3, topLeft3, (0, 255, 0), 2)
			
            cX3 = int((topLeft3[0] + bottomRight3[0]) / 2.0)
            cY3 = int((topLeft3[1] + bottomRight3[1]) / 2.0)
            cv2.circle(image, (cX3, cY3), 4, (0, 0, 255), -1)
            
            #cv2.line(image, (cX0, cY0), (cX1, cY1), (0, 255, 0), 2)
            #cv2.line(image, (cX1, cY1), (cX2, cY2), (0, 255, 0), 2)
            #cv2.line(image, (cX2, cY2), (cX3, cY3), (0, 255, 0), 2)
            #cv2.line(image, (cX3, cY3), (cX0, cY0), (0, 255, 0), 2)
            list1 = [cX0, cX1, cX2, cX3]
            list2 = [cY0, cY1, cY2, cY3]
            cxmax = max(list1)
            cxmin = min(list1)
            cymax = max(list2)
            cymin = min(list2)
            #cv2.line(image, (cxmax, cymax), (cxmax, cymin), (0, 255, 0), 2)
            cv2.line(image, (cxmax, cymin), (cxmin, cymin), (0, 255, 0), 2)
            cv2.line(image, (cxmin, cymin), (cxmin, cymax), (0, 255, 0), 2)
            cv2.line(image, (cxmin, cymax), (cxmax, cymax), (0, 255, 0), 2)
            
            
    return image




aruco_type = "DICT_5X5_100"

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters_create()


cap = cv2.VideoCapture('D:/Downloads/eraser_type2_aruco_video.mp4')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while cap.isOpened():
    
    ret, img = cap.read()

    h, w, _ = img.shape

    width = 500
    height = int(width*(h/w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
 
    corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    #print(corners[0])
    #print(corners[1])
    

    detected_markers = aruco_display(corners, ids, rejected, img)
    

    cv2.imshow("Image", detected_markers)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
	    break

cv2.destroyAllWindows()
cap.release()