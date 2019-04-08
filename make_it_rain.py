import urllib.request
import cv2
import numpy as np
import os
import time
def platesTest():
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
						image = cv2.imread(str(image_name[0]) + ".png", cv2.IMREAD_COLOR)
						roi = image[int(coordinates[1]):int(coordinates[1])+int(coordinates[3]), int(coordinates[0]):int(coordinates[0])+int(coordinates[2])]
						cv2.imwrite("npos/"+ str(cont) + ".jpg", roi)
					
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
platesTest()