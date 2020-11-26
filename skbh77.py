import numpy as np
import cv2
import math

img = cv2.imread('face1.jpg',-1)
def problem1(image,dCo,bCo,mode):
    llFilter1 = np.zeros([400,400,1],dtype=np.uint8)
    llFilter1.fill(255)
    llFilter2 = np.zeros([400,400,1],dtype=np.uint8)
    llFilter2.fill(255)
    llFilter3 = np.zeros([400,400,1],dtype=np.uint8)
    llFilter3.fill(255)
    llFilter = cv2.merge((llFilter1,llFilter2,llFilter3))
    #llFilter = cv2.imread('simpleMaskTest.png')
    mask = cv2.imread('filterMask2.png')
    height,width,chans = image.shape
    roi = image[0:height,0:width]
    for i in range(0,height):
        for j in range(0,width):
            for c in range(0,chans):
                if(np.any(mask[i, j] == 0)):
                    llFilter[i,j,c] = np.clip(1.5*image[i,j,c]+50, 0, 255)
                    

    image = image/255
    image = (image**dCo)
    image = np.floor(255*image)
    cv2.imwrite("step1.png", roi);
   
    roi = cv2.bitwise_and(roi,mask)
    cv2.imwrite("step2.png", roi);

    # Make black the background pixels of the logo 
    llFilter = cv2.bitwise_and(llFilter, cv2.bitwise_not(mask))
    # Save an intermediate result
    cv2.imwrite("step3.png", llFilter);

    # OR the ROI and the logo (as have been processed by the mask) 
    roi = cv2.bitwise_or(roi,llFilter)
    # Save an intermediate result
    cv2.imwrite("step4.png", roi);


    image = (1-bCo)*image + bCo*llFilter
    cv2.imwrite("imLogo.png", image);
    return image


img = problem1(img,4,1/2,0)


# Window name in which image should be displayed 
window_name = 'image'

# Conversion so that imshow works as expected - rectifies conversion to float above
img = img.astype(np.uint8)
cv2.imshow(window_name, img);

#waits for user to press any key  
cv2.waitKey(0)  
  
#closing all open windows  
cv2.destroyAllWindows()  
