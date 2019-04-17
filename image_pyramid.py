import cv2
import numpy as np

img = cv2.imread("plate_sample.jpg") #ler imagem do disco
img = cv2.resize(img, (640,480)) #redimensionar imagem


# Gaussian Pyramid
layer = img.copy() #copia a imagem
gaussian_pyramid = [layer] #guarda a a imagem original na lista
for i in range(8):
    layer = cv2.pyrDown(layer) #diminui a imagem pela metade
    gaussian_pyramid.append(layer) # guarda na lista de imagens

# Laplacian Pyramid
layer = gaussian_pyramid[7] #pega o topo da pirâmide
laplacian_pyramid = [layer] #guarda o topo na lista da pirâmida laplaciana

for i in range(7, 0, -1):

    size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
    gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size) #aumenta a dimensão da imagem
    laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded) #obtêm a imagem residual
    laplacian_pyramid.append(laplacian) #guarda na lista d eimagens

reconstructed_image = laplacian_pyramid[0] #recebe o topo da pirâmide laplaciana

for i in range(1, 8):
    size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
    reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=size)
    reconstructed_image = cv2.add(reconstructed_image, laplacian_pyramid[i])
    #cv2.imshow(str(i), reconstructed_image)
    cv2.imshow(str(i), laplacian_pyramid[i])

cv2.imshow("original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()