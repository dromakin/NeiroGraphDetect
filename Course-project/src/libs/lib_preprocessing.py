from PIL import Image
import numpy as np
import shutil
import cv2
import os


# create folder by path
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
        return True
    except OSError:
        print('DEBUG:\nError: Creating directory. ' + directory + '\n')


# save image
def saveImage(path_with_name, img):
    cv2.imwrite(path_with_name, img)
    return True


# resize image with params (msg == anable messages)
def resizeImage(input_image, width=None, height=None, save=False, msg = False):
    # copy -> save -> load
    original_image = input_image.copy()
    saveImage('./original_image.jpg', original_image)
    original_image = Image.open('./original_image.jpg')
    w, h = original_image.size

    if msg == True:
        print('The original image size is {wide} wide x {height} high'.format(wide=w, height=h))

    if width and height:
        max_size = (width, height)
    elif width:
        max_size = (width, h)
    elif height:
        max_size = (w, height)
    else:
        # No width or height specified
        raise RuntimeError('Width or height required!')

    # save and load
    original_image.thumbnail(max_size, Image.ANTIALIAS)
    original_image.save('./resize_image.jpg')
    resize_img = cv2.imread('./resize_image.jpg')
    resize_origimg = resize_img.copy()

    # delete
    os.remove('./original_image.jpg')
    os.remove('./resize_image.jpg')

    if save == True:
        saveImage('./result/preprocessing/resize_origimg.jpg', resize_origimg)

    if msg == True:
        height, width, depth = resize_origimg.shape
        print('The scaled image size is {wide} wide x {height} high'.format(wide=width, height=height))

    return resize_origimg