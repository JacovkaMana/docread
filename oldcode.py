import imghdr
from itertools import count
import cv2
import math
from tkinter import *
import numpy as np
from keras.models import load_model
import argparse
import cv2
import easyocr
import pytesseract

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split




# path
path = '1-29.png'
# font
# Reading an image and saving it
image = cv2.imread(path)
original_image = image.copy()


print("finished cropping")




# Setting the reader 
reader = easyocr.Reader(['ru'])

# Reading an image

print("/////////////////////////////////////////////////")
#print("Reader")
#result = reader.readtext(image)
#for item in result:
    #print(item[1])


pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'
#text = pytesseract.image_to_string(image)
textrus = pytesseract.image_to_string(image, lang='rus')
#print('Pytesseract any')
#print(text)
print('Pytesseract rus')
print(textrus)



