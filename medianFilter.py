#####################################################################

import numpy as np
import cv2

#####################################################################

# Define the size of the filter 
n = 3 

# Load the image
img = cv2.imread('face1.jpg')

# Read the dimensions of the image
rows,cols,channels = img.shape

# Initialise the new image (where the filtered values will be stored) 
imgNew = np.zeros((rows, cols, channels))

#####################################################################

# MEDIAN FILTER
for x in range(n, rows-n):          # loop through rows (ignore boundary)
    for y in range(n, cols-n):      # loop through cols (ignore boundary)
        for z in range(channels):   # loop through BGR colour channels 
            neighbourhood = img[x-n:x+n, y-n:y+n, z]
            # define filter neighbourhood 
            imgNew[x,y,z] = np.median(neighbourhood)    # compute median value    

#####################################################################

# Save the result 
cv2.imwrite("medianFiltered.png", imgNew);

##################################################################### 
