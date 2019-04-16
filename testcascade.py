import numpy as np
import cv2
import time 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#this is the cascade we just made. Call what you want
#watch_cascade = cv2.CascadeClassifier('classifier12HORAS-20-STAGES/cascade.xml')
#watch_cascade = cv2.CascadeClassifier('classifier-silver-plates-60x20-11h/cascade.xml')
#watch_cascade = cv2.CascadeClassifier('classifier-red-plates-60x20-11h/cascade.xml')
watch_cascade = cv2.CascadeClassifier('classifier-silver-plates-randomsize-12h/cascade.xml')
#watch_cascade = cv2.CascadeClassifier("classifier/cascade.xml")
#watch_cascade = cv2.CascadeClassifier("CASCADE-PLATES-20-2.xml")
#watch_cascade = cv2.CascadeClassifier('CASCADE-PLATES-20-1.xml')

#watch_cascade = cv2.CascadeClassifier("br.xml")

#cap = cv2.VideoCapture("carro_andando.mp4")
file = open("platesinfo.txt", "r")
file_names = file.read()
#while 1:
cont = 1
for name in file_names.split("\n"):
    
    time.sleep(1/30.0)
    #print(name)
    img = cv2.imread(name, cv2.IMREAD_COLOR)
    
    #ret, img = cap.read()
    #img = cv2.imread("plate0.png", cv2.IMREAD_COLOR)
    
    try:
        img = img
        img = cv2.resize(img, (1280, 720))
    except Exception as e:
        continue
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        continue

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    new_image = img.copy()
    # add this
    # image, reject levels level weights.
    watches = watch_cascade.detectMultiScale(gray, 1.4, 6)
    nx, ny, nw, nh = 0,0,0,0
    # add this
    for (x,y,w,h) in watches:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        nx, ny, nw, nh = x,y,w,h

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    if len(watches)==0:
        continue

        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    try:
        new_image = cv2.resize(new_image[ny:ny+nh,nx:nx+nw], (120, 40))
    except Exception as e:
        new_image = new_image[ny:ny+nh,nx:nx+nw]

    cv2.imwrite("found_by_cascade4/plate-" + str(cont) + ".jpg", img)
    print(name + " " + str(cont))
    cv2.imshow('img',img)
    cont = cont + 1

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()