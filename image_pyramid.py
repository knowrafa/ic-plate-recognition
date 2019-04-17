import cv2
import numpy as np

img = cv2.imread("plate_sample.jpg") #ler imagem do disco
img = cv2.resize(img, (640,480)) #redimensionar imagem


# Gaussian Pyramid
layer = img.copy() #copia a camada
gaussian_pyramid = [layer] #guarda a primeira layer
for i in range(6):
    layer = cv2.pyrDown(layer) #diminui a imagem pela metade e guarda
    gaussian_pyramid.append(layer) # 

# Laplacian Pyramid
layer = gaussian_pyramid[5]
laplacian_pyramid = [layer]

for i in range(5, 0, -1):

    size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
    gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
    laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
    laplacian_pyramid.append(laplacian)

reconstructed_image = laplacian_pyramid[0]

for i in range(1, 6):
    size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
    reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=size)
    reconstructed_image = cv2.add(reconstructed_image, laplacian_pyramid[i])
    #cv2.imshow(str(i), reconstructed_image)
    cv2.imshow(str(i), laplacian_pyramid[i])

cv2.imshow("original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()