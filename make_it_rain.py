import urllib.request
import cv2
import numpy as np
import os
import time

def find_car_region():
	file = open("platesinfo.txt", "r")
	file_names = file.read()
	cont = 1
	for name in file_names.split('\n'):
		#print(name)
		try:
			file2 = open(str(name), "r")
			file2_names = file2.read()
			for name2 in file2_names.split('\n'):
				#print(name2.find("position_plate:"))
				if name2.find("position_plate:") is not -1:
					positions = name2.split(": ")
					print(positions[1])
					coordinates = []
					for pos in positions[1].split(" "):
						coordinates.append(pos)
						
					#print(coordinates)
					image_name = name.split(".")
					#image = cv2.imread(str(image_name[0]) + ".png", cv2.IMREAD_COLOR)
					#print(str(image_name[0]) + ".png")
					#cv2.rectangle(image,(int(coordinates[0]),int(coordinates[1])),( int(coordinates[2]), int(coordinates[3])),(0,255,0),2)
					#print(image)
					#print(int(coordinates[1]),int(coordinates[3]), int(coordinates[0]),int(coordinates[2]))
					#roi = image[int(coordinates[1]):int(coordinates[1])+int(coordinates[3]), int(coordinates[0]):int(coordinates[0])+int(coordinates[2])]
					#cv2.imshow("hey", roi)
					#time.sleep(10)
					if cont%30 == 0:
						image = cv2.imread(str(image_name[0]) + ".png", cv2.IMREAD_GRAYSCALE)
						roi = image[int(coordinates[1])-3:int(coordinates[1])-3+int(coordinates[3])+6, int(coordinates[0])-3:int(coordinates[0])-3+int(coordinates[2])+6]
						if roi is not None:
							cv2.imwrite("plates_extracted/"+ str(cont+1) + ".jpg", roi)

					cont = cont + 1
		except Exception as e:
			print(e)
		'''
		img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
		try:
			resized_image = cv2.resize(img, (300, 120))
			cv2.imwrite(name, resized_image)

		except Exception as e:
			print(e)
		'''
		
def find_car_region_and_create_info_file():
	file = open("platesinfo2.txt", "r")
	file_names = file.read()
	cont = 1
	for name in file_names.split('\n'):
		#print(name)
		try:
			file2 = open(str(name), "r")
			file2_names = file2.read()
			coordinates = []
			for name2 in file2_names.split('\n'):
				#print(name2.find("position_plate:"))
				if name2.find("position_vehicle:") is not -1:
					positions = name2.split(": ")
					
					print(positions[1])
					
					for pos in positions[1].split(" "):
						coordinates.append(pos)

					image_name = name.split(".")
					
					image = cv2.imread(str(image_name[0]) + ".png", cv2.IMREAD_GRAYSCALE)
					roi = image[int(coordinates[1]):int(coordinates[1])+int(coordinates[3]), int(coordinates[0]):int(coordinates[0])+int(coordinates[2])]
					if roi is not None:
						cv2.imwrite("car_regions2/"+ str(cont) + ".jpg", roi)
				
				if name2.find ("type: motorcycle") is not -1:
					os.remove("car_regions2/"+ str(cont) + ".jpg")
					break

				if name2.find("position_plate:") is not -1:
					positions = name2.split(": ")
					coordinates2 = []
					print(positions[1])
					
					
					for pos in positions[1].split(" "):
						coordinates2.append(pos)
					
					#Atualizando a nova posição da placa na imagem
					coordinates2[0] = int(coordinates2[0]) - int(coordinates[0])
					coordinates2[1] = int(coordinates2[1]) - int(coordinates[1])
					
					#Abrindo arquivo (e criando) para escrita 
					info_file = open("car_regions2/" + str(cont) + ".txt", 'a')
					info_file.write("position_plate: " + str(coordinates2[0]) + " " + str(coordinates2[1]) + " " + str(coordinates2[2]) + " " + str(coordinates2[3]) + "\n")
					info_file.close()
					
					image_name = name.split(".")
					
					cont = cont + 1
					
		except Exception as e:
			print(e)
		'''
		img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
		try:
			resized_image = cv2.resize(img, (300, 120))
			cv2.imwrite(name, resized_image)

		except Exception as e:
			print(e)
		'''
	
		

#find_car_region()
find_car_region_and_create_info_file()
