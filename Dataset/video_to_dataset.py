import cv2
import numpy as np
import pandas as pd
r = 0
ob_class = 1
pos = [ob_class]
kernel = np.ones((3, 3), np.uint8)
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('/home/gaurx/Documents/FinalProject/Dataset/data/IMG_0210.MOV')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):

  # Capture frame-by-frame
  ret, frame = cap.read()
  h, w, d = frame.shape
  if ret == True:
 
    # Display the resulting frame
    cv2.imshow('Frame',frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([101,50,38])
    upper_blue = np.array([210,255,255])
    Blue = cv2.inRange(hsv, lower_blue, upper_blue)
    Bluemask = cv2.dilate(Blue, kernel, iterations=1)
    Bluem = cv2.erode(Bluemask, kernel, iterations=1)
    edges = cv2.Canny(Bluem,100,200)
    # Bluem = cv2.erode(Blue, kernel, iterations=3)
    # Bluemask = cv2.dilate(Bluem, kernel, iterations=3)
    cv2.imshow('mask',edges)
    for i in range(h):
      for j in range(w):
        if edges[i,j] == 255 :
          pos.append(float(i/h))
          pos.append(float(j/w))
    npos = np.array([pos])
    cv2.imwrite('/home/gaurx/Documents/FinalProject/Dataset/dataset/image/image_{}.png'.format(r),frame)
    npos.tofile('/home/gaurx/Documents/FinalProject/Dataset/dataset/label/image_{}.txt'.format(r),sep = " ")
    pos = [ob_class]
    r += 1

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()