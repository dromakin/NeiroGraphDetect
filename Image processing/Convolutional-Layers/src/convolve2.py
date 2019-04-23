#!/usr/bin/env python3

import matplotlib.pyplot as plt
from skimage import exposure
from skimage import io, color
import numpy as np
import sys

def convolve2d(image, kernel):
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))
    # convolution output
    output = np.zeros_like(image)
    # Add zero padding to the input image
    image_pad = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_pad[1:-1, 1:-1] = image

    # Loop over every pixel of the image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_pad[y:y + 3, x:x + 3]).sum()
    return output

# read arg
def readarg(image, kern):
    # Load the image
    img = io.imread(image)
    # img = image
    # Convert the image to grayscale (1 channel)
    img = color.rgb2gray(img)

    # Adjust the contrast of the image by applying Histogram Equalization
    # image_equalized = exposure.equalize_adapthist(img / np.max(np.abs(img)), clip_limit=0.03)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('orig.png')
    plt.show()

    # KERNEL
    # Convolve the sharpen kernel and the image
    kernel = kern

    # call to action
    image_sharpen = convolve2d(img, kernel)

    # Plot the filtered image
    plt.imshow(image_sharpen, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('result.png')
    plt.show()

# converter to numpy array
def readkern(file):
    narray = list()
    for line in file:
        text = line.replace('\n', '').split(' ')
        while '' in text:
            text.remove('')
        print(text)
        # list to create numpy array
        narr = list(
            map(
                lambda x: float(x),
                text
            )
        )
        narray.append(narr)

    print(narray)
    return np.asarray(narray, dtype=float)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('To few arguments!')

    # image
    image = sys.argv[1]

    # file
    filename = sys.argv[2]
    file = open(filename, 'r')
    # get kernel from file
    kernel = readkern(file)
    print('KERNEL: ', kernel)

    readarg(image, kernel)

    sys.exit(0)