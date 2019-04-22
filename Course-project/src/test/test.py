import numpy as np
import shutil
import cv2
import os
import sys

libdir = os.path.dirname('./src/')
sys.path.append(os.path.split(libdir)[0])
from src import lib_preprocessing as module
from src import canny

IMAGE = './data/green/IMG_3105.JPG'
TYPE = 'green'
path = './'

def read_image(filename):
    return cv2.imread(filename)

def preprocessing(image, type=None, save=False):
    img = image.copy()
    height, width, depth = img.shape
    if ((height > 900) | (width > 900)):
        img = module.resizeImage(img, width=900, height=900, save=True)

    if ((type == 'Green') | (type == 'green')):
        # filtering
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 2)
        imgbit = cv2.bitwise_not(blur_gray)
        if save == True:
            module.saveImage(path + 'preprocessing.jpg', imgbit)

        return imgbit
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if save == True:
            module.saveImage(path + 'preprocessing.jpg', gray)

        return gray


if __name__ == "__main__":
    image = read_image(IMAGE)

    print(image.shape)

    image = preprocessing(image, type=TYPE)

    print(image.shape)

    canny = canny.coordCanny(image)

    print(image)
    print(canny)

    height, width = image.shape
    img = np.zeros((height, width, 3), np.uint8)

    for i in canny:
        cv2.circle(img, (i[1], i[0]), 1, (0, 0, 255), -1)

    module.saveImage(path + 'BLACK.jpg', img)