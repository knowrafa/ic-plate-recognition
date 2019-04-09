import numpy as np
import cv2
import time 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#this is the cascade we just made. Call what you want
watch_cascade = cv2.CascadeClassifier('classifier/cascade.xml')
#watch_cascade = cv2.CascadeClassifier("CASCADE-PLATES-20-2.xml")

#cap = cv2.VideoCapture("carro_andando.mp4")
file = open("plates.txt", "r")
file_names = file.read()
#while 1:
for name in file_names.split("\n"):
    time.sleep(1)
    print(name)
    img = cv2.imread(name, cv2.IMREAD_COLOR)
    
    #ret, img = cap.read()
    #img = cv2.imread("plate0.png", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # add this
    # image, reject levels level weights.
    watches = watch_cascade.detectMultiScale(gray, 1.3, 5)
    
    # add this
    for (x,y,w,h) in watches:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()