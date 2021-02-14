#####################################################################

import numpy as np
import cv2

#####################################################################

# Load the images
img = cv2.imread('peppers.png')
logo = cv2.imread('duLogo.png')
mask = cv2.imread('logoMask.png')

# Read the dimensions of the logo
rows,cols,channels = logo.shape

# Extract the region of interest (ROI) of the original image
# This is where the logo will be placed  
roi = img[0:rows, 0:cols]

# Save the ROI as an intermediate result
#(if you want to see the progress of the algorithm) 
 
cv2.imwrite("step1.png", roi);


#####################################################################

# LOGICAL OPERATIONS

# Make black the pixels in the ROI where the logo will be placed 
roi = cv2.bitwise_and(roi,mask) 
# Save an intermediate result
cv2.imwrite("step2.png", roi);

# Make black the background pixels of the logo 
logo = cv2.bitwise_and(logo, cv2.bitwise_not(mask))
# Save an intermediate result
cv2.imwrite("step3.png", logo);

# OR the ROI and the logo (as have been processed by the mask) 
roi = cv2.bitwise_or(roi,logo)
# Save an intermediate result
cv2.imwrite("step4.png", roi);

#####################################################################

# Copy the ROI which now has the logo on it on to the original image 
img[0:rows, 0:cols] = roi

# Save the result 
cv2.imwrite("imLogo.png", img);

#####################################################################


