#!/usr/bin/env python3

from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import sys
import argparse


# This function which takes an image and a kernel
# and returns the convolution of them
# Args:
#   image: a numpy array of size [image_height, image_width].
#   kernel: a numpy array of size [kernel_height, kernel_width].
# Returns:
#   a numpy array of size [image_height, image_width] (convolution output).
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

def readarg(image, kern):
    # Load the image
    img = io.imread(image)
    # img = image
    # Convert the image to grayscale (1 channel)
    img = color.rgb2gray(img)

    # Adjust the contrast of the image by applying Histogram Equalization
    image_equalized = exposure.equalize_adapthist(img / np.max(np.abs(img)), clip_limit=0.03)
    plt.imshow(image_equalized, cmap=plt.cm.gray)
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
    plt.savefig('gray.png')
    plt.show()

    # Adjust the contrast of the filtered image by applying Histogram Equalization
    image_sharpen_equalized = exposure.equalize_adapthist(
        image_sharpen / np.max(np.abs(image_sharpen)),
        clip_limit=0.03)
    plt.imshow(image_sharpen_equalized, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('black.png')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True, help="Path to the image")
        ap.add_argument('-f', '--file', type=argparse.FileType('r'), help="Open specified file")

        # Text
        argum = ap.parse_args()
        kernel = np.loadtxt(argum.file, dtype=int)
        print('KERNEL: ', kernel)

        # Image
        args = vars(ap.parse_args())
        img = args["image"]

        # Call to action
        readarg(img, kernel)

    else:
        print("Error.")
        sys.exit(1)