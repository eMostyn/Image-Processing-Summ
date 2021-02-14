import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('face1.jpg',0)


rows, cols = img.shape
crow,ccol = int(rows/2) , int(cols/2)


# create a mask, centered square has values 1,
# the rest of the mask all zeros
halfEdge = 10
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-halfEdge:crow+halfEdge, ccol-halfEdge:ccol+halfEdge] = 1


# apply DFT 
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)


# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])


# plotting
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Low frequncies'), plt.xticks([]), plt.yticks([])
plt.show() 
