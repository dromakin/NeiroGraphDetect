#!/usr/bin/env python3

import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
import hashlib
import shutil
import cv2
import sys
import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def getndarray(img):
    newarr = img.tolist()
    # print(newarr)
    for i in newarr:
        # print(type(i))
        for j in i:
            # print(j)
            j.remove(0)
            j.remove(0)
            # print(j)

    return np.asarray(newarr)

def convolve2d(image, kernel):
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))
    # convolution output
    output = np.zeros_like(image)
    # Add zero padding to the input image
    image_pad = np.zeros(
        (image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] - 1))
    sh1 = int((kernel.shape[0] - 1) / 2)
    sh2 = int((kernel.shape[1] - 1) / 2)
    image_pad[sh1:-sh1, sh2:-sh2] = image

    # Loop over every pixel of the image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_pad[y:y + kernel.shape[0], x:x + kernel.shape[1]]).sum()
    return output

def savematrix(img, file):
    # height, width, channels
    # print(img.shape)
    # print(img)

    text_file = open('./' + str(sys.argv[3]) + '/' + file, 'w')
    # width
    for i in range(img.shape[0]):
        for j in img[i,:]:
            txt = str(j) + ' '
            text_file.write(txt)
        text_file.write('\n\n')
    text_file.close()

def getresult(r, g, b, height, width):
    r = r.tolist()
    g = g.tolist()
    b = b.tolist()

    result = list()

    for i in range(height):
        res = list()
        for j in range(width):
            res.append([r[i][j], g[i][j], b[i][j]])
        result.append(res)

    return np.asarray(result)

# read arg
def readarg(image, kern, var):
    if var == 'kern':
        # Load the image
        img = cv2.imread(image)
        # Convert the image to grayscale (1 channel)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # convert original image to gray
        plt.imshow(img, cmap=plt.cm.gray)
        plt.axis('off')
        savematrix(img, 'grayorig.txt')
        plt.savefig('./' + str(sys.argv[3]) + '/grayorig.png')
        plt.show()

        # KERNEL
        # Convolve the sharpen kernel and the image
        kernel = kern

        # call to action
        imageconv = convolve2d(img, kernel)

        # Plot the convolve image
        plt.imshow(imageconv, cmap=plt.cm.gray)
        plt.axis('off')
        savematrix(imageconv, 'result.txt')
        plt.savefig('./' + str(sys.argv[3]) + '/result.png')
        plt.show()

        # Test
        print('(grayorig.png == result.png): ', (img == imageconv).all())
        print('Hash: ',hashlib.md5(img).hexdigest() == hashlib.md5(imageconv).hexdigest())

    elif var == 'kern3':
        # load the image
        img = misc.imread(image, mode='RGB')
        r = img.copy()
        g = img.copy()
        b = img.copy()

        # RGB
        # -----------------------
        # RED
        r[:, :, 1] = 0
        r[:, :, 2] = 0

        kernel_r = kern[0]

        print('RED:')

        img = getndarray(r)
        img.shape = (img.shape[0], img.shape[1])
        savematrix(img, 'r.txt')

        # call to action
        result_r = convolve2d(img, kernel_r)

        # Plot the convolve channel of the image
        plt.imshow(result_r)
        plt.axis('off')
        savematrix(result_r, 'result_r.txt')
        plt.savefig('./' + str(sys.argv[3]) + '/result_r.png')
        plt.show()

        #-------------------------
        # GREEN
        g[:, :, 0] = 0
        g[:, :, 2] = 0

        kernel_g = kern[1]

        print('GREEN:')

        img = getndarray(g)
        img.shape = (img.shape[0], img.shape[1])
        savematrix(img, 'g.txt')

        # call to action
        result_g = convolve2d(img, kernel_g)

        # Plot the convolve channel of the image
        plt.imshow(result_g)
        plt.axis('off')
        savematrix(result_g, 'result_g.txt')
        plt.savefig('./' + str(sys.argv[3]) + '/result_g.png')
        plt.show()

        #--------------------------
        # BLUE
        b[:, :, 0] = 0
        b[:, :, 1] = 0

        kernel_b = kern[2]

        print('BLUE:')

        img = getndarray(b)
        img.shape = (img.shape[0], img.shape[1])
        savematrix(img, 'b.txt')

        # call to action
        result_b = convolve2d(img, kernel_b)

        # Plot the convolve channel of the image
        plt.imshow(result_b)
        plt.axis('off')
        savematrix(result_b, 'result_b.txt')
        plt.savefig('./' + str(sys.argv[3]) + '/result_b.png')
        plt.show()

        #----------------------------
        # RGB convolve2d result
        result = getresult(result_r, result_g, result_b, img.shape[0], img.shape[1])

        print('GREEN:')

        # plot 3 channels
        plt.imshow(result)
        plt.axis('off')
        plt.savefig('./' + str(sys.argv[3]) + '/result.png')
        plt.show()

# converter to numpy array
def readkern(file, var):
    i = 0
    index = list()
    narray = list()
    for line in file:
        if line == '\n':
            index.append(i)
        text = line.replace('\n', '').split(' ')
        while '' in text:
            text.remove('')
        # print(text)
        # list to create numpy array
        narr = list(
            map(
                lambda x: float(x),
                text
            )
        )
        # index
        i += 1
        narray.append(narr)
    if var == 'kern':
        # print(narray)
        return np.asarray(narray, dtype=float)
    elif var == 'kern3':
        kernel1 = np.asarray(narray[:index[0]], dtype=float)
        kernel2 = np.asarray(narray[index[0] + 1:index[1]], dtype=float)
        kernel3 = np.asarray(narray[index[1] + 1:], dtype=float)
        return [kernel1, kernel2, kernel3]

if __name__ == '__main__':
    # create folder
    folder = './' + str(sys.argv[3]) + '/'
    createFolder(folder)

    # image
    image = sys.argv[1]

    # file
    if sys.argv[2].split('.')[1] == 'kern':

        filename = sys.argv[2]
        file = open(filename, 'r')

        # get kernel from file
        kernel = readkern(file, 'kern')
        print('KERNEL: ', kernel)

        readarg(image, kernel, 'kern')

    elif sys.argv[2].split('.')[1] == 'kern3':

        filename = sys.argv[2]
        file = open(filename, 'r')

        # get kernel from file
        kernel = readkern(file, 'kern3')
        # print('KERNEL: ', kernel)

        readarg(image, kernel, 'kern3')



    sys.exit(0)
