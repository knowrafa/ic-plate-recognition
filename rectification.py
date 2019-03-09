import cv2
import numpy as np
import time

#image = cv2.imread("plate3.jpg", 1)
#image2 = image.copy()
refPt = []
refPt2 = []
clicks = 0
camera = cv2.VideoCapture(0)
def get_frame():
    if camera.isOpened():
        _, image = camera.read()
        image2 = image.copy()
    else:
        exit()
    return image, image2

def click_and_choose_region(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, clicks
    if clicks < 4:
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.append([x,y])
            refPt2.append((x,y))
            clicks = clicks + 1
    else:
        #refPt = sorted(refPt , key=lambda k: [k[0], k[1]])
        p1, p2, p3, p4 = refPt
        pts = np.array([p1, p2, p3, p4], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(image,[pts],True,(0,255,255))

cv2.namedWindow("Imagem")
cv2.setMouseCallback("Imagem", click_and_choose_region)

while True:
    # Mostra a imagem e aguarda uma tecla ser pressionada

    image, image2 = get_frame()

    if clicks == 4:
        pts = np.array(refPt, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(image,[pts],True,(0,255,255))

    cv2.imshow("Imagem", image)
    key = cv2.waitKey(1) & 0xFF
    if cv2.getWindowProperty("Imagem",cv2.WND_PROP_VISIBLE) < 1:
        exit()
    # Se a tecla ESC for pressionada
    if key == 27:
        cv2.destroyAllWindows()
        exit()
    # Se a tecla enter for pressionada, o corte é feito
    if key == 13 and clicks ==4:
        break
    elif key ==13 and clicks ==4:
        print("Precisamos de 4 pontos para a transformação perspectiva")
    # Se a tecla 'r' é pressionada, os clicks são resetados
    elif key == ord("r"):
        clicks = 0
        refPt = []
        image = image2.copy()
camera.release()

HEIGHT = image.shape[0]
WIDTH = image.shape[1]

p1, p2, p3, p4 = refPt

shape = np.float32([p1, p2, p4, p3])
plot = np.float32([[0,0],[WIDTH,0],[0,HEIGHT],[WIDTH,HEIGHT]])
perspective = cv2.getPerspectiveTransform(shape,plot)

camera.open(0)

while True:
    # Mostra a imagem e aguarda uma tecla ser pressionada
    image, image2 = None, None
    image, image2 = get_frame()

    image2 = cv2.warpPerspective(image2, perspective, (WIDTH,HEIGHT))
    cv2.imshow("Imagem", image2)
    
    key = cv2.waitKey(1)
    
    if key == 27:
        cv2.destroyAllWindows()
        exit()

    if cv2.getWindowProperty("Imagem",cv2.WND_PROP_VISIBLE) < 1:
        exit()

cv2.destroyAllWindows()         
