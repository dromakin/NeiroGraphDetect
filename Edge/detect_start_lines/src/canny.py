import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
import hashlib
import shutil
import math
import cv2
import sys
import cv2
import os


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


# save image
def save_image(image, filename):
    cv2.imwrite(filename, image)


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def coordCanny(image, path='./', save_middle_result=False, save_result=False, savetxt=False):
    img = image.copy()
    gray = img
    if len(img.shape) == 3:
        # filters
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gaus = cv2.GaussianBlur(gray, (3, 3), 0)
    # Canny
    edges = auto_canny(gray)
    # edges = cv2.Canny(gray, 100, 255)
    kernel = np.ones((5, 5), np.uint8)
    edges_k = cv2.dilate(edges, kernel, iterations=3)
    # Find coordinates of edges
    indices = np.where(edges != [0])

    if savetxt == True:
        datcoord = path + 'CannyOutput_y_x' + '.txt'
        dat = open(datcoord, 'w')
        for i in range(0, len(indices[0])):
            dat.write(str(indices[0][i]) + ' ' + str(indices[1][i]) + '\n')
        dat.close()

    if save_middle_result == True:
        save_image(path + 'Canny_orig.jpg', edges)

    save_image(path + 'Canny_dilation.jpg', edges_k)

    img_k = cv2.imread(path + 'Canny_dilation.jpg')
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.erode(img_k, kernel, iterations=3)
    # kernel = np.ones((3, 3), np.uint8)
    # edges = cv2.erode(edges, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=2)

    if save_result == True:
        save_image(path + 'Canny_result.jpg', edges)
    else:
        os.remove(path + 'Canny_dilation.jpg')

    indices = np.where(edges != [0])
    # make coordinates in simple form
    detcoord = []
    for i in range(0, len(indices[0])):
        detcoord.append([indices[0][i], indices[1][i]])

    return (detcoord)


if __name__ == '__main__':
    # create folder
    folder = './' + 'CannyOutput' + '/'
    createFolder(folder)

    # image
    image = sys.argv[1]

    coordCanny(image)

    sys.exit(0)