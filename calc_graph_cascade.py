import numpy as np
import cv2
import time
import math
import sys 
from tabulate import tabulate
import matplotlib.pyplot as plt
import pytesseract

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#CASCATAS FEITAS
#my_cascade = cv2.CascadeClassifier('classifier12HORAS-20-STAGES/cascade.xml')
#my_cascade = cv2.CascadeClassifier('classifier-silver-plates-60x20-11h/cascade.xml')
#my_cascade = cv2.CascadeClassifier('classifier-red-plates-60x20-11h/cascade.xml')
#my_cascade = cv2.CascadeClassifier('classifier-silver-plates-randomsize-12h/cascade.xml')

my_cascade = cv2.CascadeClassifier("classifier/cascade.xml")

#my_cascade = cv2.CascadeClassifier("classifier_120x40/cascade.xml")
#my_cascade = cv2.CascadeClassifier("CASCADE-PLATES-20-2.xml") #Melhor resultado na ALPR

#my_cascade = cv2.CascadeClassifier('CASCADE-PLATES-20-1.xml')

#my_cascade = cv2.CascadeClassifier("br.xml")

file = open("negatives.txt", "r")
#file = open("car_info.txt", "r")
file_names = file.read()
#false_negative = (cont-true_positive) + false_negative
true_positive = 0
false_negative = 0
true_negative = 0
false_positive = 0

for name in file_names.split("\n"):
    
    time.sleep(1/30.0)
    #print(name)
    img = cv2.imread(name, cv2.IMREAD_COLOR)

    #ret, img = cap.read()
    #img = cv2.imread("plate0.png", cv2.IMREAD_COLOR)
    if img is None:
        continue

    currentHeight,currentWidth = img.shape[:2]
    width, height = 640, 480
    try:
        img = img
        #img = cv2.resize(img, (1280,720))
        img = cv2.resize(img, (640, 480))
    except Exception as e:
        false_negative = false_negative + 1
        continue
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        #false_negative = false_negative + 1
        continue

    #gray = cv2.equalizeHist(gray)
    new_image = img.copy()
    # add this
    # image, reject levels level weights.
    plates = my_cascade.detectMultiScale(gray, 1.3, 5)
    nx, ny, nw, nh = 0,0,0,0
    flag = 0
    if len(plates) is 0:
        true_negative = true_negative + 1
    else:
        #Verifica se a região está acima da metade da imagem, se todas estiverem, a placa não pode estar e logo é um verdadeiro negativo
        for (x,y,w,h) in plates:
            #print( str(y + h) + " --- " + str(0.8*height/2))
            if y+h < (height/2):
                continue
            else:
                flag = 1
                break

        if flag == 1:
            false_positive = false_positive + 1
        else:
            true_negative = true_negative + 1

    print(name + " " + str(false_positive+true_negative))
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

#### END OF NEGATIVE CALCULATION ####
false_positive_neg = false_positive
#cap = cv2.VideoCapture("carro_andando.mp4")
file = open(sys.argv[1], "r")
#file = open("car_info.txt", "r")
file_names = file.read()
#while 1:
#cont = 1
t_positive_vec = []
t_positive_vec_accuracy = []
t_positive_vec_recall = []
t_positive_vec_precision = []
t_positive_vec_fscore = []

for dist in range(0,200):
    cont = 1
    true_positive = 0
    false_negative = 0
    false_positive = false_positive_neg
    true_negative = true_negative
    for x in range(1,3):
        file = open(sys.argv[x], "r")
        #file = open("car_info.txt", "r")
        file_names = file.read()

        for name in file_names.split("\n"):
            
            time.sleep(1/30.0)
            #print(name)
            img = cv2.imread(name, cv2.IMREAD_COLOR)

            #ret, img = cap.read()
            #img = cv2.imread("plate0.png", cv2.IMREAD_COLOR)
            if img is None:
                continue

            currentHeight,currentWidth = img.shape[:2]
            width, height = 640, 480
            try:
                img = img
                #img = cv2.resize(img, (1280,720))
                img = cv2.resize(img, (width, height))
            except Exception as e:
                false_negative = false_negative + 1
                continue
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                #false_negative = false_negative + 1
                continue

            #gray = cv2.equalizeHist(gray)
            new_image = img.copy()
            plates = my_cascade.detectMultiScale(gray, 1.3, 5)
            nx, ny, nw, nh = 0,0,0,0
            
            if len(plates)==0:
                false_negative = false_negative + 1
                continue

            new_name = name.split(".jpg")
            #print(new_name)
            info_file = open(new_name[0] + ".txt", 'r')
            plate_position = info_file.read()
            coordinates = []
            if plate_position.find("position_plate:") is not -1:
                positions = plate_position.split(": ")
                #print(positions[1])
                for pos in positions[1].split(" "):
                    coordinates.append(int(pos))

            info_file.close()

            #newX = coordinates[0]
            #newY = coordinates[1]
            #newXf = coordinates[2]
            #newYf = coordinates[3]
            newX = (coordinates[0]/currentWidth)*width + 0.5
            newY = (coordinates[1]/currentHeight)*height + 0.5

            newXf = ((coordinates[0] + coordinates[2])/currentWidth)*width + 0.5
            newYf = ((coordinates[1]+coordinates[3])/currentHeight)*height + 0.5

            #Plate positions é o ground truth da placa
            plate_positions = []
            plate_positions.append(int(newX))
            plate_positions.append(int(newY))
            plate_positions.append(int(newXf))
            plate_positions.append(int(newYf))

            #print(plate_positions)
            # Vetores pro plot
            

            for (x,y,w,h) in plates:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
                nx, ny, nw, nh = x,y,w,h

                g_truth_triangle_center_x = (x+w + x)/2
                g_truth_triangle_center_y = (y+h + y)/2

                false_positive_triangle_center_x = (plate_positions[2] + plate_positions[0])/2
                false_positive_triangle_center_y = (plate_positions[3] + plate_positions[1])/2

                euclidean_dist = math.sqrt(math.pow(false_positive_triangle_center_x-g_truth_triangle_center_x, 2) + math.pow(false_positive_triangle_center_y-g_truth_triangle_center_y, 2))
                #print("Euclidean dist: " + str(euclidean_dist))
                #Tamanho de janela para o ring growing
                tjanela = 20
                
                if y+h < (height/2):
                    continue
                    
                #Verifica se o ground truth está inscrito em uma das regiões de interesse
                #Também verifica se a região de interesse está inscrita no ground truth
                #Verifica se a distância entre os centros dos retângulos é menor que 10
                '''
                if (plate_positions[0] > x and plate_positions[2] < (x + w) and plate_positions[1] > y and plate_positions[3] < (y + h)) or \
                (x > plate_positions[0] and (x + w) < plate_positions[2] and y > plate_positions[1] and (y + h) < plate_positions[3]) or \
                '''
                if euclidean_dist < dist:
                    #print("true_positive: " + str(true_positive))
                    #teste = new_image[ny:ny+nh,nx:nx+nw]
                    #print(pytesseract.image_to_string(testes))
                    true_positive = true_positive + 1
                    #t_positive_vec.append(name, dist, )
                    cv2.rectangle(img,(plate_positions[0],plate_positions[1]),(plate_positions[2],plate_positions[3]),(0,255,0),2)
                    cv2.imwrite("./true_positive_images/plate-" + str(cont) + ".jpg", img)
                    break
                #else:
                #    t_positive_vec.append(name, dist, )
                '''
                elif (plate_positions[0] > (x-tjanela if x-tjanela > 0 else 0) and plate_positions[2] < (x+w+tjanela if x+w+tjanela < width else 0) and plate_positions[1] > (y-tjanela if y-tjanela > 0 else 0)\
                and plate_positions[3] < (y+h+tjanela if y+h+tjanela < height else 0)):
                    true_positive = true_positive + 1
                    cv2.rectangle(img,(plate_positions[0],plate_positions[1]),(plate_positions[2],plate_positions[3]),(0,255,0),2)
                    cv2.imwrite("./true_positive_images/plate-" + str(cont) + ".jpg", img)
                    break
                '''

                #for (x,y,w,h) in faces:
                #    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

                #cv2.rectangle(img,(plate_positions[0],plate_positions[1]),(plate_positions[2],plate_positions[3]),(0,255,0),2)
            
                #t_positive_vec.append(name, dist, )
                
            if len(plates)==0:
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

            #cv2.imwrite("false_positive_by_cascade3/plate-" + str(cont) + ".jpg", img)
            print(name + " " + str(cont) + " - " + str(dist))
            
            #cv2.imshow('img',img)
            
            cont = cont + 1

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
    false_negative = (cont-true_positive) + false_negative
    print("Placas perdidas (False Negative): " + str(false_negative))
    print("Placas com regiões de interesse: " + str(cont))
    print("Imagens em que a placa foi encontrada (True Positive): " + str(true_positive))

    print("Porcentagem de acerto: " + str(int(true_positive*100.0/cont)) + "%")

    #def evaluate_classifier():

    ### ROTINA PARA TESTAR A PERFORMANCE NOS DADOS NEGATIVOS (ONDE NÁ HÁ PLACA) ###
    print("True Positive: " + str(true_positive))
    print("False negative: " + str(false_negative))
    print("Imagens sem placa em que placa não foi identificada (True Negative): " + str(true_negative))
    print("Imagens sem placa em que alguma placa foi identificada (False Positive): " + str(false_positive))

    results = []
    results.append(("[Valor Real] Placas", true_positive, false_negative))
    results.append(("[Valor Real] Não Placas",   false_positive, true_negative))
    print(tabulate(results, headers=[" ", "[Valor Predito] Placas", "[Valor Predito] Não Placas"]))

    accuracy = (true_positive + true_negative)/(true_positive+true_negative+false_negative+false_positive)
    recall = (true_positive/(true_positive+false_negative))
    
    precision= (true_positive/(true_positive+false_positive)) #
    
    if precision == 0 or recall == 0:
        f_score = 0
    else:
        f_score = (2*precision*recall)/(precision+recall)


    metrics = []
    metrics.append(("Accuracy", format(accuracy, '.2f')))
    metrics.append(("Recall", format(recall, '.2f')))
    metrics.append(("Precision", format(precision, '.2f')))
    metrics.append(("f-score", format(f_score, '.2f')))

    t_positive_vec.append(dist)
    t_positive_vec_accuracy.append(accuracy)
    t_positive_vec_precision.append(precision)
    t_positive_vec_fscore.append(f_score)
    t_positive_vec_recall.append(recall)
    print(tabulate(metrics))
    
    print("end of distance " + str(dist))
    plt.xlim(0,200)
    plt.ylim(0,1)
    plt.xlabel('Diferença dos centros (em pixels)')
    plt.ylabel('Accuracy')
    plt.title("Accuracy Graph")
    scat = plt.scatter(t_positive_vec, t_positive_vec_accuracy)
    plt.savefig('accuracy1.pdf')
    #plt.show()

    scat.remove()
    plt.clf()
    plt.xlim(0,200)
    plt.ylim(0,1)
    plt.xlabel('Diferença dos centros (em pixels)')
    plt.ylabel('Recall')
    plt.title("Recall Graph")
    scat = plt.scatter(t_positive_vec, t_positive_vec_recall)
    plt.savefig('recall1.pdf')
    #plt.show()

    scat.remove()
    plt.xlim(0,200)
    plt.ylim(0,1)
    plt.xlabel('Diferença dos centros (em pixels)')
    plt.ylabel('Precision')
    plt.title("Precision Graph")
    scat = plt.scatter(t_positive_vec, t_positive_vec_precision)
    plt.savefig('precision1.pdf')
    #plt.show()

    scat.remove()
    plt.xlim(0,200)
    plt.ylim(0,1)
    plt.xlabel('Diferença dos centros (em pixels)')
    plt.ylabel('f-Score')
    plt.title("f-Score Graph")
    scat = plt.scatter(t_positive_vec, t_positive_vec_accuracy)
    plt.savefig('fscore1.pdf')
    #plt.show()
    scat.remove()
#cap.release()
cv2.destroyAllWindows()
