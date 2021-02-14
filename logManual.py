import numpy as np
import cv2
import math

# Naive and slow implementation of LoG just for learning purposes

range_inc = lambda start, end: range(start, end+1) 

def l_o_g(x, y, sigma): # LoG reaaranged for readability, from https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    nom = ( (y**2)+(x**2)-2*(sigma**2) )
    denom = ( (2*math.pi*(sigma**6) ))
    expo = math.exp( -((x**2)+(y**2))/(2*(sigma**2)) )
    return nom*expo/denom

def create_log(sigma, size = 6):
    w = math.ceil(float(size)*float(sigma))
    # If dimension is even, make it odd
    if(w%2 == 0):
        w = w + 1
    l_o_g_mask = []
    w_range = int(math.floor(w/2))
    for i in range_inc(-w_range, w_range):
        for j in range_inc(-w_range, w_range):
            l_o_g_mask.append(l_o_g(i,j,sigma))
    l_o_g_mask = np.array(l_o_g_mask)
    l_o_g_mask = l_o_g_mask.reshape(w,w)
    return l_o_g_mask

if __name__ == "__main__":
    img = create_log(1.0, size = 100)**(1/10) # Raise to power just to aid visualisation
    normalizedImg = np.zeros((100, 100))
    normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    imC = cv2.applyColorMap(normalizedImg.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imshow("kernel",imC)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

