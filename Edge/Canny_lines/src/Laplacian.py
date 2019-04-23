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


def coordCanny(image):
    img = cv2.imread(image)
    # filters
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gaus = cv2.GaussianBlur(gray, (3, 3), 0)
    # Canny
    edges = cv2.Laplacian(img,cv2.CV_64F)
    # Find coordinates of edges
    indices = np.where(edges != [0])
    print(indices)
    #     print('y: ', indices[0], '\nx: ', indices[1])
    # coordinates = zip(indices[1], indices[0])

    # make coordinates in simple form
    detcoord = []

    datcoord = folder + 'LaplacianOutput_y_x' + '.txt'
    dat = open(datcoord, 'a')

    # dat.write('y x\n')

    for i in range(0, len(indices[0])):
        dat.write(str(indices[0][i]) + ' ' + str(indices[1][i]) + '\n')
        detcoord.append([indices[0][i], indices[1][i]])

    dat.close()

    cv2.imwrite(folder + 'Laplacian.jpg', edges)

    return (detcoord)



if __name__ == '__main__':
    # create folder
    folder = './' + 'Laplacian' + '/'
    createFolder(folder)

    # image
    image = sys.argv[1]

    coordCanny(image)

    sys.exit(0)