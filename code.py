
import cv2
from tkinter import *
import numpy as np
import random 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


# construct the argument parser and parse the arguments
####

# path
path = 'example2.jpeg'
# font
font = cv2.FONT_HERSHEY_COMPLEX

old_boundaries = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]

boundaries = [
	([20,20,20], [100,100,180])
]

blue = np.uint8([[[0, 0, 255]]])
hsvBlue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
print(hsvBlue)
# 'blue': [[128, 255, 255], [90, 50, 70]],

#lowerLimit = hsvBlue[0][0][0] - 10, 100, 100
#upperLimit = hsvBlue[0][0][0] + 10, 255, 255

lowerValues = np.array([90, 50, 70])
upperValues = np.array([128, 255, 255])

# Reading an image and saving it
image = cv2.imread(path)
original_image = image.copy()


grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

bluepenMask = cv2.inRange(hsvImage, lowerValues, upperValues)

kernelSize = 3
# Set morph operation iterations:
opIterations = 1
# Get the structuring element:
morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
# Perform closing:
bluepenMask = cv2.morphologyEx(bluepenMask, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)

cv2.imwrite('bluepenMask.jpg', bluepenMask)

colorMask = cv2.add(grayscaleImage, bluepenMask)
cv2.imwrite('colorMask.jpg', colorMask)

_, binaryImage = cv2.threshold(colorMask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite('bwimage.jpg',binaryImage)

#thresh, im_bw = cv2.threshold(binaryImage, 210, 230, cv2.THRESH_BINARY)
thresh, im_bw = cv2.threshold(binaryImage, 0, 200, cv2.THRESH_BINARY)
kernel = np.ones((1, 1), np.uint8)
imgfinal = cv2.dilate(im_bw, kernel=kernel, iterations=1)

cv2.imwrite('final.jpg', imgfinal)

coord = cv2.findNonZero(bluepenMask)

_x = []
_y = []
coords_fit = []


for item in coord:
    _x.append(item[0][0])
    _y.append(item[0][1])
    coords_fit.append([item[0][0], item[0][1]])
  
# Create feature and target arrays

print("Input centers")
n_centers = int(input())
centers = []

image_h = image.shape[0]
image_w = image.shape[1]

"""
if (n_centers == 1):
    centers.append([image_h / 2, image_w / 2])
else:
    for i in range(n_centers):
        t_x = i * image_h / (n_centers -1 )
        t_y = i * image_w / (n_centers -1 )
        centers.append([t_x,t_y])
"""

for i in range(n_centers):
    t_x = random.randint(0, image_h)
    t_y = random.randint(0, image_w)
    centers.append([t_x,t_y])



print(centers)
print(image_h, image_w)

import time
from sklearn.cluster import KMeans

#k_means = KMeans(init="k-means++", n_clusters = n_centers, n_init=10)
k_means = KMeans(init="random", n_clusters = n_centers, n_init=10)
t0 = time.time()
k_means.fit(coords_fit)
t_batch = time.time() - t0

print(k_means.cluster_centers_)

for center in k_means.cluster_centers_:

    cv2.circle(original_image, (int(center[0]),int(center[1])), radius = 0, color = (0,255,0), thickness = 5)

cv2.imshow("", original_image)
cv2.waitKey(0)