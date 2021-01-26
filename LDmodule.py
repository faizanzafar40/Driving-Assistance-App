import numpy as np
import cv2
from matplotlib import pyplot as plt

ANCHOR_THRESH = 14
SCAN_INTERVAL = 5
HORIZONTAL = 0
VERTICAL = 90
EDGE = 255

def getSobelX(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    return sobelx

def getSobelY(image):
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return sobely

def getGradientMagnitudeFromSobel(image):
    sobelx = getSobelX(image)
    sobely = getSobelY(image)
    dxabs = cv2.convertScaleAbs(sobelx)
    dyabs = cv2.convertScaleAbs(sobely)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag


def getGradientMagnitudeFromScharr(image):
    scharrx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    scharry = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sxabs = cv2.convertScaleAbs(scharrx)
    syabs = cv2.convertScaleAbs(scharry)
    magS = cv2.addWeighted(sxabs, 0.5, syabs, 0.5, 0)
    return magS

def isAnchor(x,y,G,D,ANCHOR_THRESH):
    if(D[x,y] == 0):

        if ((G[x, y] - G[x, y-1] >= 8) and (G[x, y] - G[x, y+1] >= 8)):
            return 1
    else:
        if ((G[x, y] - G[x-1, y] >= ANCHOR_THRESH)and (G[x, y] - G[x+1, y] >= ANCHOR_THRESH)):
            return 1
    return 0

def goLeft(x,y,G,D,E,counter):

    while(counter == 1):
        E[x, y] = EDGE;        #Mark this pixel as an edgel
        # Look at 3 neighbors to the left & pick the one with the max. gradient value
        if(G[x-1, y-1] > G[x-1, y] and G[x-1, y-1] > G[x-1, y+1]):


            x=x-1
            y = y-1   # Up-Left
        elif (G[x-1, y+1] > G[x-1, y] and G[x-1, y+1] > G[x-1, y-1]):
            x = x-1
            y = y+1  # Down-Left
        else:
            x = x-1

        if(E[x,y] != EDGE):
            counter=0


    while (G[x, y] > 0 and D[x, y] == HORIZONTAL and E[x, y] != EDGE):

        E[x, y] = EDGE;        #Mark this pixel as an edgel
        # Look at 3 neighbors to the left & pick the one with the max. gradient value
        if(G[x-1, y-1] > G[x-1, y] and G[x-1, y-1] > G[x-1, y+1]):


            x=x-1
            y = y-1   # Up-Left
        elif (G[x-1, y+1] > G[x-1, y] and G[x-1, y+1] > G[x-1, y-1]):
            x = x-1
            y = y+1  # Down-Left
        else:

            x = x-1                 # Straight-Left
        if(x>190 or y >256):
            break


def goRight(x,y,G,D,E,counter):
    while(counter == 1):
        E[x, y] = EDGE;        #Mark this pixel as an edgel
        # Look at 3 neighbors to the left & pick the one with the max. gradient value
        if(G[x-1, y-1] > G[x-1, y] and G[x-1, y-1] > G[x-1, y+1]):


            x=x-1
            y = y-1   # Up-Left
        elif (G[x-1, y+1] > G[x-1, y] and G[x-1, y+1] > G[x-1, y-1]):
            x = x-1
            y = y+1  # Down-Left
        else:
            x = x-1

        if(E[x,y] != EDGE):
            counter=0

    while (G[x,y] > 0 and D[x, y] == HORIZONTAL and E[x, y] != EDGE):
        E[x, y] = EDGE;        #Mark this pixel as an edgel
        # Look at 3 neighbors to the right & pick the one with the max. gradient value
        if(G[x+1,y-1]> G[x+1,y] and G[x+1,y-1] > G[x+1,y+1]):
            x = x+1
            y =y-1
        elif (G[x+1, y] > G[x+1, y-1] and G[x+1, y] > G[x+1, y+1]):
            x = x+1
            y=y
        else:
            x = x+1
            y = y+1
        if(x>190 or y >256):
             break

def goUp(x,y,G,D,E,counter):
    while(counter == 1):
        E[x, y] = EDGE;        #Mark this pixel as an edgel
        # Look at 3 neighbors to the left & pick the one with the max. gradient value
        if(G[x-1, y-1] > G[x-1, y] and G[x-1, y-1] > G[x-1, y+1]):


            x=x-1
            y = y-1   # Up-Left
        elif (G[x-1, y+1] > G[x-1, y] and G[x-1, y+1] > G[x-1, y-1]):
            x = x-1
            y = y+1  # Down-Left
        else:
            x = x-1

        if(E[x,y] != EDGE):
            counter=0

    while (G[x, y] > 0 and D[x, y] == VERTICAL and E[x, y] != EDGE):
        E[x, y] = EDGE;        #Mark this pixel as an edgel
        # Look at 3 neighbors to the left & pick the one with the max. gradient value
        if(G[x-1, y-1] > G[x, y-1] and G[x-1, y-1] > G[x+1, y-1]):
            x = x-1
            y = y-1   # Up-Left
        elif (G[x, y-1] > G[x-1, y-1] and G[x, y-1] > G[x+1, y-1]):
            x = x
            y = y-1  # Down-Left
        else:
          x = x+1
          y = y-1# Straight-Left
          if(x>190 or y >256):
             break

def goDown(x,y,G,D,E,counter):
    while(counter == 1):
        E[x, y] = EDGE;        #Mark this pixel as an edgel
        # Look at 3 neighbors to the left & pick the one with the max. gradient value
        if(G[x-1, y-1] > G[x-1, y] and G[x-1, y-1] > G[x-1, y+1]):


            x=x-1
            y = y-1   # Up-Left
        elif (G[x-1, y+1] > G[x-1, y] and G[x-1, y+1] > G[x-1, y-1]):
            x = x-1
            y = y+1  # Down-Left
        else:
            x = x-1

        if(E[x,y] != EDGE):
            counter=0

    while (G[x, y] > 0 and D[x, y] == VERTICAL and E[x, y] != EDGE):

        E[x, y] = EDGE;        #Mark this pixel as an edgel
        # Look at 3 neighbors to the left & pick the one with the max. gradient value
        if(G[x-1, y+1] > G[x, y+1] and G[x-1, y+1] > G[x+1, y+1]):
            x = x-1
            y = y+1   # Up-Left
        elif (G[x, y+1] > G[x-1, y+1] and G[x, y+1] > G[x+1, y+1]):
            x = x
            y = y+1  # Down-Left
        else:
          x = x+1
          y = y+1# Straight-Left
        if(x>190 or y >256):
             break


cap = cv2.imread('test.jpg', 0)


     # Step 1


# Apply gaussian blur

blur = cv2.GaussianBlur(cap,(5,5),0)

    # Step 2


#get the gradient magintude image and apply thresholding : gradient map

gradient_map = getGradientMagnitudeFromSobel(blur)
cv2.imshow('gradient-map', gradient_map)
kernel = np.ones((3, 3), np.uint8)
gradient_map = cv2.erode(gradient_map, kernel, iterations=1)


# get direction map

sobelx = getSobelX(blur)
sobely = getSobelY(blur)

rows, cols = sobelx.shape
direction_map = np.zeros((rows, cols, 1), np.uint8)

for i in xrange(rows):
    for j in xrange(cols):
        k = sobelx.item(i, j)
        l = sobely.item(i, j)
        if(k>l):
            direction_map.itemset((i, j, 0), 0)
        else:
            direction_map.itemset((i, j, 0), 90)

# get edge-area-image

rows_g, cols_g = gradient_map.shape

edge_area_image = np.zeros((rows_g, cols_g, 1), np.uint8)

for i in xrange(rows_g-1):
    for j in xrange(cols_g-1):
        k = gradient_map.item(i, j)
        if(k<36):
            edge_area_image.itemset((i, j, 0), 0)
        else:
            edge_area_image.itemset((i, j, 0), 255)

cv2.imshow('thresholded-gradient map : Edge Area Image', edge_area_image)


# getting the direction map from the x and y derivative

rows1, cols1 = gradient_map.shape


anchor_image = np.zeros((rows1, cols1, 1), np.uint8)

    # Step 3 : Find Anchors

for i in xrange(rows-1):
    for j in xrange(cols-1):
        if(i%SCAN_INTERVAL==0):
            if(j%SCAN_INTERVAL==0):
                if(isAnchor(i, j, gradient_map, direction_map, ANCHOR_THRESH)==1):
                    anchor_image.itemset((i, j, 0), 255)
                else:
                    anchor_image.itemset((i, j, 0), 0)
            else:
                anchor_image.itemset((i, j, 0), 0)
cv2.imshow('After step 3', anchor_image)

anchor_image = anchor_image.squeeze()

    # Step 4

rowsa = rows-3
colsa = cols-3
print rowsa, colsa, rowsa
for i in xrange((rowsa)):
    for j in xrange(colsa):

        if(i==0 or j==0):
            continue
        k = anchor_image.item(i, j)
        if(k==255):

            if(direction_map[i,j]==0):

                goLeft(i, j, gradient_map, direction_map, anchor_image, 1)
                goRight(i, j, gradient_map, direction_map, anchor_image, 1)
            else:
                goUp(i, j, gradient_map, direction_map, anchor_image, 1)
                goDown(i, j, gradient_map, direction_map, anchor_image, 1)

# print anchor_image

cv2.imshow('After step 4', anchor_image)

# When everything done, release the capture
k = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
