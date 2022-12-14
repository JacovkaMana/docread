
import cv2

from tkinter import *
import numpy as np
#from keras.models import load_model

import cv2
#import easyocr
import pytesseract

from alive_progress import alive_bar

from dataclasses import dataclass

import parse


def resToString(s):
    output = ''
    if (s == None):
        return ""
    elif (type(s) == str):
        return s

    for n in s:
        output += str(n) + "."
    output = output[:-1]
    return output


def read_img(path):
    image = cv2.imread(path)
    original_image = image.copy()


    lowerValues = np.array([0, 0, 0])
    upperValues = np.array([188, 255, 30])

    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    bluepenMask = cv2.inRange(hsvImage, lowerValues, upperValues)

    kernelSize = 3
    # Set morph operation iterations:
    opIterations = 1
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform closing:
    bluepenMask = cv2.morphologyEx(bluepenMask, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)
    coord = cv2.findNonZero(bluepenMask)
    print("Finished mask")
    cv2.imwrite('newmask.jpg', bluepenMask)

    #######################   Mask final
    #######################   Start grouping

    image_h = image.shape[0]
    image_w = image.shape[1]
    d_h = image_h * 4 / 100
    d_w = image_w * 10 / 100
    #print(d_h, d_w)

    rects = []

    x1 = coord[0][0][0]
    x2 = coord[0][0][0]
    y1 = coord[0][0][1]
    y2 = coord[0][0][1]

    _prev_x = coord[0][0][0]
    _prev_y = coord[0][0][1]

    new = False

    print("Finding groups in mask")
    with alive_bar(len(coord)) as bar:
        for item in coord:
            _x = (item[0][0])
            _y = (item[0][1])

            dif_x = abs(_x - _prev_x )
            dif_y = abs(_y - _prev_y )
            #print(dif_x, dif_y)

            if ((dif_y > d_h)):
                new = True
                rects.append([x1, x2, y1, y2])
                x1 = _x
                x2 = _x
                y1 = _y
                y2 = _y



            if(not new):
                if (_x < x1):
                    x1 = _x
                elif (_x > x2):
                    x2 = _x

                if (_y < y1):
                    y1 = _y
                elif (_y > y2):
                    y2 = _y

            _prev_x = _x
            _prev_y = _y
            new = False
            bar()

    if(not new):
        rects.append([x1, x2, y1, y2])


    #print(rects)


    crops = [] 
    print("Cropping")
    with alive_bar(len(rects)) as bar:
        for item in rects:
            cv2.rectangle(original_image, [item[0], item[2]], [item[1], item[3]], color = (0,0,255), thickness=2)
            t_image = image [item[2] - 5: item[3] + 5, item[0] - 5: item[1] + 5]
            crops.append(t_image)
            bar()


    cv2.imwrite("rected_image.png",original_image)



    # Setting the reader 
    #reader = easyocr.Reader(['ru'])

    # Reading an image

    #print("Reader")
    #result = reader.readtext(image)
    #for item in result:
        #print(item[1])


    pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'
    #text = pytesseract.image_to_string(image)
    _i = 0
    textrus = []
    print("Decyphiring text")
    with alive_bar(len(crops)) as bar:
        for crop in crops:
            s = pytesseract.image_to_string(crop, lang='rus')
            s = s.replace('\n', ' ')
            textrus.append(s)
            bar()
        


    @dataclass
    class PText:
        name: str = "unassigned"

        has_full_date: bool = False
        has_short_date: bool = False
        has_inn: bool = False
        has_town: bool = False
        has_number: bool = False
        has_fio: bool = False
        has_name: bool = False

        text: str = ""
        contents: str = ''

    @dataclass
    class PDoc:

        short_date: str = ""
        full_date: str = ""
        number: str = ""
        town: str = ""
        inn: str = ""
        name: str = ""
        fulltext: str = ""


    parsed = []
    doc = PDoc()

    print("Parsing text")
    with alive_bar(len(textrus)) as bar:
        for txt in textrus:
            doc.fulltext += txt
            curr = PText()
            curr.text = txt
            if ("ИНН" in txt):
                curr.has_inn = True
                _p = parse.search('ИНН {} ', curr.text)
                print("inn " + str(_p))
                _r = resToString(_p)
                doc.inn = _r

            if ("№" in txt):
                curr.has_number = True
                _p = parse.search('№ {} ', curr.text)
                print("number " + str(_p))
                _r = resToString(_p)
                doc.number = _r
            if ("г." in txt):
                curr.has_town = True
                _p = parse.search('г.{},', curr.text)
                print("town "  + str(_p))
                _r = resToString(_p)
                doc.town = _r
            if ("от" in txt):
                curr.has_short_date = True
                _p = parse.search('от {:d}.{:d}.{:d}', curr.text)
                print("sdate " + str(_p))
                _r = resToString(_p)
                doc.short_date = _r
            if ("года" in txt): 
                curr.has_full_date = True
                _p = parse.search('{:d} {} {} года', curr.text)
                print("fdate " + str(_p))
                _r = resToString(_p)
                doc.full_date = _r
            if ("«" in txt): 
                curr.name = True
                _p = parse.search('«{}»', curr.text)
                print("name " + str(_p))
                _r = resToString(_p)
                doc.name = _r
            parsed.append(curr)
            bar()

    for p in parsed:
        print("//////////")
        print(p.text)
        print("INN " + str(p.has_inn))
        print("Number" + str(p.has_number))
        print("Town" + str(p.has_town))
        print("Date" + str(p.has_short_date))
        print("FDate" + str(p.has_full_date))
        #_p = parse.search('от {:d}.{:d}.{:d}', p.text)
        #print(_p)

    print(doc)
    return doc  


import pandas as pd

def save_to(doc):
    df = pd.DataFrame(columns=["Номер документа", "Название","Полная дата", "Короткая дата", "Город", "ИНН", "Текст"])
    df.loc[len(df.index)] = [doc.number, doc.name, doc.full_date, doc.short_date, doc.town, doc.inn, doc.fulltext]
    df.to_csv('result.csv', index=False)
    df.to_csv('result.xls', index=False)
    df.to_csv('result.xlsx', index=False)



import sys
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication)
from PyQt5.QtGui import QFont

class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        QToolTip.setFont(QFont('SansSerif', 10))

        self.setToolTip('This is a <b>QWidget</b> widget')

        btn = QPushButton('Button', self)
        btn.setToolTip('This is a <b>QPushButton</b> widget')
        btn.resize(btn.sizeHint())
        btn.move(50, 50)

        self.setGeometry(300, 300, 640, 360)
        self.setWindowTitle('Tooltips')
        self.show()

if __name__ == '__main__':


    # path = '1-29.png'
    # doc = read_img(path)
    # save_to(doc)

    app = QApplication(sys.argv)

    ex = Example()
    sys.exit(app.exec_())
