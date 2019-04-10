import urllib.request
import cv2
import numpy as np
import os

def store_raw_images():
	neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02123159'
	neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()

	if not os.path.exists('neg'):
		os.makedirs('neg')

	pic_num = 1

	for i in neg_image_urls.split('\n'):
		try:
			print(i)
			urllib.request.urlretrieve(i, "neg/"+str(pic_num) + '.jpg')
			
			img = cv2.imread("neg/"+str(pic_num) + '.jpg', cv2.IMREAD_GRAYSCALE)
			resized_image = cv2.resize(img, (640, 480))
			cv2.imwrite("neg/"+str(pic_num) + '.jpg', resized_image)
			pic_num += 1
		except Exception as e:
			print(str(e))

def find_uglies():
	for file_type in ['neg']:
		for img in os.listdir(file_type):
			for ugly in os.listdir('uglies'):
				try:
					current_image_path = str(file_type) + '/' + str(img)
					ugly = cv2.imread('uglies/' + str(ugly))
					question = cv2.imread(current_image_path)

					if(ugly.shape == question.shape and not (np.bitwise_xor(ugly,question)).any()):
						print("u're ugly!")
						print(current_image_path)
						os.remove(current_image_path)

				except Exception as e:
					print(str(e))
#store_raw_images()

def resize_image(name):
	img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
	resized_image = cv2.resize(img, (400, 320))
	cv2.imwrite(name, resized_image)

#find_uglies()
def create_pos_n_neg():
	for file_type in ['neg']:
		for img in os.listdir(file_type):
			if file_type == 'neg':
				line = file_type + '/' + img + '\n'
				with open('bg.txt', 'a') as f:
					f.write(line)

def resize_all_images():
	file = open("positive10.txt", "r")
	file_names = file.read()

	for name in file_names.split():
		print(name)
		img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
		try:
			resized_image = cv2.resize(img, (60, 20))
			cv2.imwrite(name, resized_image)

		except Exception as e:
			print(e)

#store_raw_images()
#find_uglies()
#resize_image("monitor.jpg")
resize_all_images()
#create_pos_n_neg()