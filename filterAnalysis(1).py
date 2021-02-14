import cv2
import numpy as np
from matplotlib import pyplot as plt

# create the 5x5 mean filter 
meanFilter = np.ones((5,5))/25

# create a 5x5 guassian filter
x = cv2.getGaussianKernel(5,1)
gaussian = x*x.T

# create a Laplacian filter 
laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])

filters = [meanFilter, gaussian, laplacian]



# zero pad the filter (requirement of the convolution theorem)
# compute the Fourier transforms of the filters and plot the magnitude spectrum 
padSize = 100
filters = [np.pad(m, ((padSize, ), (padSize, )), 'constant') for m in filters] 
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]


# plotting 
filter_name = ['Mean', 'Gaussian','Laplacian']
for i in range(3):
    plt.subplot(1,3,i+1),plt.imshow(mag_spectrum[i],cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])

plt.show()
