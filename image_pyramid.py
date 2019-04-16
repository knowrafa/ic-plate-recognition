import cv2

def get_laplacian_pyramid_layer(img, n):
'''Returns the n-th layer of the laplacian pyramid'''  
    currImg, i = img, 0
    while i < n: # and currImg.size > max_level (83)
        down, up = new_empty_img(img.shape), new_empty_img(img.shape)
        down = cv2.pyrDown(img)
        up = cv2.pyrUp(down, dstsize=currImg.shape)
        lap = currImg - up
        currImg = down
        i += 1
    return lap

