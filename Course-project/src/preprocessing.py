# USAGE
# python src/preprocessing.py -i ./data/green/IMG_3095.JPG -t green
# python src/preprocessing.py -i ./data/white/6.JPG -t white

import argparse
import shutil
import cv2
import os
import sys

libdir = os.path.dirname('./src/')
sys.path.append(os.path.split(libdir)[0])
from src.libs import lib_preprocessing as module


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
    # create dir
    path = './result/preprocessing/'
    module.createFolder(path)
    #
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-t", "--type", required=True, help="green/white")
    args = vars(ap.parse_args())
    # load the image
    image_path = str(args["image"])
    type_img = str(args["type"])
    # open image
    image = cv2.imread(image_path)
    # preprocessing
    preprocessing(image, type_img, True)