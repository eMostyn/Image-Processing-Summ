import numpy as np
import cv2
import math
import random
from matplotlib import pyplot as plt
import argparse
def problem1(image,dCo,bCo,mode):
    #Get the dimensions of the shape
    height,width,chans = image.shape
    roi = image[0:height,0:width]
    #If doing the light leak
    if mode == 0:
        #Create matrices of all white 
        llFilter1 = np.zeros([height,width,1],dtype=np.uint8)
        llFilter1.fill(255)
        llFilter2 = np.zeros([height,width,1],dtype=np.uint8)
        llFilter2.fill(255)
        llFilter3 = np.zeros([height,width,1],dtype=np.uint8)
        llFilter3.fill(255)
        llFilter = cv2.merge((llFilter1,llFilter2,llFilter3))
        #Read in the black mask
        mask = cv2.imread('mask.jpg')
        dim = (height,width)
        #Resize this mask to the same size as the image
        mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
        #For each value
        for i in range(0,height):
            for j in range(0,width):
                for c in range(0,chans):
                    #If the mask is black at that location
                    if(np.any(mask[i, j] == 0)):
                        #The filter is a brightened version of the original image at that location, clipped to max of 255                   
                        llFilter[i,j,c] = np.clip(1.5*image[i,j,c]+50, 0, 255)
                        
        #Normalise the image
        image = image/255
        #Decrease the brightness by the specified amount
        image = (image**dCo)        
        image = np.floor(255*image)


        #And the mask and the roi, making the pixels in the mask region black on the roi image
        roi = cv2.bitwise_and(roi,mask)
        #Black entire filter except for mask region
        theFilter = cv2.bitwise_and(llFilter, cv2.bitwise_not(mask))

    #Rainbow mode
    else:
        #Read in the rainbow mask
        rfilter = cv2.imread('rMask.jpg')
        #Get the dimensions of the image and resize the mask if necessary 
        dim = (height,width)
        rfilter = cv2.resize(rfilter, dim, interpolation = cv2.INTER_AREA)
        #Read in the black mask
        mask = cv2.imread('mask.jpg')
        #Resize mask if necessary
        mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)

        #Darken the image
        image = image/255
        image = (image**dCo)
        image = np.floor(255*image)

        #And the mask and the roi, making the pixels in the mask region black on the roi image
        roi = cv2.bitwise_and(roi,mask)

        #Black entire filter except for mask region
        theFilter = cv2.bitwise_and(rfilter, cv2.bitwise_not(mask))



    #xScalar will increase/decrease as we go across to alter opacity
    xScalar = 0.0

    #This region represents the top/bottom of the mask where it will be more transparent
    heightStart = height * 21/100 #84
    heightEnd = int(height * 0.695) #278
    heightRegion1End = int(height*0.26) # 104
    heightRegion2Start = int(height *0.645) # 258

    #This region represents the left/right of the mask where it will be more transparent
    widthStart = int(width *0.4025) #161
    widthEnd = int(width*0.5) # 200
    widthRegion1End = int(width*0.4275) #171
    widthRegion2Start = int(width*0.475) #190
    #For each height
    for i in range(0,height):
        #If the i is within range
        if i>=heightStart and i <=heightRegion1End:
            xScalar += 1/20
        if i>=heightRegion2Start and i <=heightEnd:
            xScalar -= 1/20
        yScalar = 0.0
        #Across the width
        for j in range(0,width):
            #Alter the y scalar
            if(i>=heightStart and i<=heightEnd) and (j>=widthStart and j<=widthRegion1End):
                yScalar += 1/10
            if(i>=heightStart and i<=heightEnd) and (j>=widthRegion2Start and j<=widthEnd):
                yScalar -= 1/10
            #If in the region we use both the y and x scalar
            if ((i>=heightStart and i <heightRegion1End) or (i>heightRegion2Start and i <=heightEnd)) and (j>=widthStart and j<=widthEnd):
                image[i,j] = np.clip((1-bCo)*image[i,j] + bCo*yScalar*xScalar*theFilter[i,j], 0, 255)
            else:
                #Not in region so only need the y scalar
                image[i,j] = np.clip((1-bCo)*image[i,j] + bCo*yScalar*theFilter[i,j], 0, 255)

    cv2.imwrite("problem1Mode:"+str(mode)+".jpg", image);
    return image
        



def problem2(img,bCo,mode):
    #Convert the colour to greyscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Create a noise array of that size
    noise = np.zeros(img.shape,np.uint8)
    #For each pixel 
    for i in range(0,img.shape[0]):
            for j in range(0,img.shape[1]):
                #Randomly generate noise between 0 and 1
                noise[i][j] = 255*random.uniform(0,1)
    #Save the noise
    cv2.imwrite("noise1.jpg", noise);
    #Size of the motion blue
    size = 8
    motion_blur = np.zeros((size, size))
    #Fill it diagonally 
    np.fill_diagonal(motion_blur, 1)
    motion_blur = motion_blur / size
    #Apply the convolution
    noise = cv2.filter2D(noise, -1, motion_blur)
    #Save the noise
    cv2.imwrite("motionBlur1.jpg", noise);
    #If its just the greyscale pencil
    if mode == 0:
        img= (1-bCo)*img + bCo*noise
    else:
        #Generate another noise
        noise2 = np.zeros(img.shape,np.uint8)
        for i in range(0,img.shape[0]):
                for j in range(0,img.shape[1]):
                    noise2[i][j] = 255*random.uniform(0.5,1)
        #Save the noise
        cv2.imwrite("noise2.jpg", noise);
        #Motion blur horizontally
        size = 8
        motion_blur = np.zeros((size, size))
        motion_blur[int((size-1)/2), :] = np.ones(size)
        motion_blur = motion_blur / size
        noise2 = cv2.filter2D(noise, -1, motion_blur)
        #Save the noise
        cv2.imwrite("motionBlur2.jpg", noise);
        #Convert the colour back to bgr
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        for i in range(0,img.shape[0]):
            for j in range(0,img.shape[1]):
                #Apply the different noises with motion blur to 2 of the three rgb channels
                img[i,j,0] = (1-bCo)*img[i,j,0] + bCo*noise[i,j]
                img[i,j,1] = (1-bCo)*img[i,j,1] + bCo*noise2[i,j]
        
    #Print 
    fileName = "problem2_Mode_"+str(mode)+".jpg"
    cv2.imwrite(fileName, img);
    return img


def problem3(img,size, bCo,brCo,dCo,sigma1):
    #Get dimensions of image
    height, width, chans = img.shape
    #Create a copy image
    smoothedImage =  img.copy()
    #Middle of the size
    mid = size//2
    #Generate the gaussian distance matrix
    distMatrix = generateDistGaussian(size,sigma1)
    #For each pixel
    for y in range(mid,height-mid):
        for x in range(mid,width-mid):
            for c in range(chans):
                #Get the neighbourhood
                neighbourhood = img[x-mid:x+mid, y-mid:y+mid, c]
                weight = 0
                top = 0
                #For each pixel in the neighbourhood
                for i in range(neighbourhood.shape[0]):
                    for j in range(neighbourhood.shape[1]):
                        #Add value to the summations 
                        neighbour = neighbourhood[i,j]
                        value = distMatrix[i,j]
                        top += value * neighbour
                        weight += value
                #Using formula in lecture slides
                smoothedImage[x,y,c] = (top) / weight
    #Save the image
    cv2.imwrite("smoothedImage.jpg", smoothedImage);
    #Convert the image to hsv in order to alter the saturation channel
    hsvImage = cv2.cvtColor(smoothedImage, cv2.COLOR_BGR2HSV)
    #Create 2 LUTs, one for brightening, 1 for darkening 
    lut1,lut2 = LUT(brCo,dCo)
    #For each pixel
    for a in range(height):
        for b in range(width):
            #Lookup the value in the lut, brightening the saturation and darkening the value channels
            hsvImage[a,b,1] = lut1[hsvImage[a,b,1]]
            hsvImage[a,b,2] = lut2[hsvImage[a,b,2]]
    #Convert back to BGR to get the output image
    outputImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)
    #Print out the image 
    cv2.imwrite("problem3.jpg",outputImage)
    return outputImage



#Used to create a matrix which is the gaussian value of the distance             
def generateDistGaussian(size,sigma):
    mid = (size//2)
    matrix = np.zeros((size, size))
    #For each item 
    for i in range(size):
        for j in range(size):
            #The value is the gaussian value at the distance
            matrix[i][j] = gaussian(math.sqrt((i-mid)**2 + (j-mid)**2),sigma)
    return matrix

#Return gaussian value of x
def gaussian(x, sigma):
    value = (1/(sigma*math.sqrt(2*math.pi))*math.exp(-x**2/(2*sigma**2)))
    return value

#Generate the LUTs for brightening and darkening
def LUT(brCo,dCo):
    #For every possible value 0->255
    lut1 = [i for i in range(256)]
    lut2 = [i for i in range(256)]
    #For every i, increase the value by the specified amount, limiting it to the maximum/minimum possible values
    #Then save it in the corresponding LUT
    for i in range(256):
        value1 =  brCo*i
        if(value1>255):
            value1 = 255
        lut1[i] = value1
        value2 =  dCo*i
        if(value2<0):
            value2 = 0
        lut2[i] = value2
    return lut1,lut2
                    

def problem4(img,theta,maxr):
    height,width,chans = img.shape


    #Split the image into bgr channels
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    #Convert each channel into fourier space
    b = np.fft.fft2(b)
    g = np.fft.fft2(g)
    r = np.fft.fft2(r)

    #Shift each of the channels so that 0 is in the middle
    b = np.fft.fftshift(b)
    g = np.fft.fftshift(g)
    r = np.fft.fftshift(r)

    #Create a a mask of 0s
    mask = np.zeros((height,width),np.uint8)
    size = 100
    midH = height//2
    midW = width//2
    mask[midH-size:midH+size, midW-size:midW+size] = 1

    #Multiply each channel by the mask           
    b= b*mask
    #Shift it back
    b = np.fft.ifftshift(b)
    #Convert back 
    b = np.fft.ifft2(b)
    #Get absolute values
    b = np.abs(b)

    #Repeat for green
    g= g*mask
    g = np.fft.ifftshift(g)
    g = np.fft.ifft2(g)
    g = np.abs(g)


    #Repeat for red
    r= r*mask
    r = np.fft.ifftshift(r)
    r = np.fft.ifft2(r)
    r = np.abs(r)

    #New image
    blurred = img.copy()
    #Set the channels of the image to bgr
    blurred[:,:,0] = b
    blurred[:,:,1] = g
    blurred[:,:,2] = r


    #newImage = np.concatenate((img,cp),axis=1)
    #Print out the image 
    cv2.imwrite("blurred.jpg",outputImage)
    swirledImage = transform(blurred,theta,maxr)
    #Print out the image 
    cv2.imwrite("swirled.jpg",outputImage)
    unSwirled = transform(swirledImage,-theta,maxr)
    #Print out the image 
    cv2.imwrite("unswirled.jpg",outputImage)
    return swirledImage

#Function to perform the transform
def transform(img,theta,maxr):
    #Get the dimensions of the image
    height,width,chans = img.shape
    #Middle point of the image
    centre_point = (height//2,width//2)
    outputImage = img.copy()
    #For each value
    for h in range(height):
        for w in range(width):
            for c in range(chans):
                #Get the r value (polar coords)
                r = math.sqrt((h-centre_point[0])**2+(w-centre_point[0])**2)
                #Get the point 
                point = (h-centre_point[0], w-centre_point[1])
                #Calculate the angle (theta in polar coords)
                pAngle = math.atan2(point[0],point[1])
                #Sets swirl amount, so only transforms in circular range 
                swirlAmount = 1-(r/maxr)
                #If within the swirl range
                if swirlAmount>0:
                    #Alter the angle of the swirl using the parameter and swirl amount so its not constant
                    pAngle += swirlAmount * theta
                    #Get the new point by converting the polar back to cartesian
                    point = (math.sin(pAngle) *r, math.cos(pAngle)*r)
                    #Top left , bottom left, top right, bottom right
                    neighbours = [[math.floor(point[0]),math.floor(point[1])],[math.ceil(point[0]),math.floor(point[1])],[math.floor(point[0]),math.ceil(point[1])],[math.ceil(point[0]),math.ceil(point[1])]]
            #Types of interpolation
                #Nearest Neighbour
                    #Current minimum dist will be the first neighbour 
                    minDist = math.sqrt((point[0]-neighbours[0][0])**2 + (point[1]-neighbours[0][1])**2)
                    #Holds index of nearest neighbour 
                    nN = 0
                    #For every other neighbour
                    for i in range(1,len(neighbours)):
                        #Calculate the distance
                        dist = math.sqrt((point[0]-neighbours[i][0])**2 + (point[1]-neighbours[i][1])**2)
                        #If distance is less we've found a nearer neighbour
                        if dist<minDist:
                            minDist = dist
                            nN = i
                    #Set the point to the nearest neighbour
                    point = neighbours[nN]
##                #Bilinear
##                    #Coordinates for easier use in formula
##                    n10,n20 = neighbours[0]
##                    n11,n21 = neighbours[1]
##                    n12,n22 = neighbours[2]
##                    n13,n23 = neighbours[3]
##                    #Matrix to be inverted - using formula
##                    matrix = np.matrix([[1,n10,n20,n10,n20],[1,n11,n21,n11,n21],[1,n12,n22,n12,n22],[1,n13,n23,n13,n23]])
##                    inv =np.linalg.pinv(matrix)
##                    #Multiply the 2 matrices to get the 
##                    coords = np.matrix([[n10,n20],[n11,n21],[n12,n22],[n13,n23]])
##                    point = (
##                    As = np.matmul(inv,coords)
                outputImage[h,w,c] = img[point[0]+centre_point[0],point[1]+centre_point[1],c]
    return outputImage







img = cv2.imread('face1.jpg',-1)
img = problem1(img,2,0.3,1)
#img = problem2(img,0.4,1)
##img = problem3(img,5,0.4,1.5,0.9)
#img = problem4(img,math.pi/2,150)

# Window name in which image should be displayed 
window_name = 'image'

# Conversion so that imshow works as expected - rectifies conversion to float above
img = img.astype(np.uint8)
cv2.imshow(window_name, img);

#waits for user to press any key  
cv2.waitKey(0)  
  
#closing all open windows  
cv2.destroyAllWindows()  
