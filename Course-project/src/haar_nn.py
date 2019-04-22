# USAGE
# python src/haar_nn.py --image ./result/preprocessing/preprocessing.jpg --modelh ./models/haar/haar_2020_2/cascade.xml --modelnn ./models/neural_networks/vertex_sort.model

import argparse
import cv2
import sys
import os

libdir = os.path.dirname('./src/')
sys.path.append(os.path.split(libdir)[0])
from src.libs import lib_haar_nn as module
from src.libs import canny

def haar_NN(image, modelh, modelnn):
    try:
        # copy images
        img = image.copy()
        img_c = img.copy()

        '''Module: Haar -> NN Classifer -> Filtering'''
        module.haarDetect(img, img_c, modelh, modelnn)

        canny.coordCanny(img, path, True, True, True)   # Canny
    except:
        print("DEBUG_ERROR: Unexpected error" + sys.exc_info()[0])


if __name__ == "__main__":
    # create VS dir
    pathvs = './result/vs/'
    module.createFolder(pathvs)
    path = './result/vs/debug_Haar_NN/'
    # create dir
    module.createFolder(path)

    # errors and debug logs
    sys.stdout = open(path + 'output.log', 'w')

    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")

    ap.add_argument("-mh", "--modelhaar", required=True,
                    help="path to trained model of haar")

    ap.add_argument("-mnn", "--modelnn", required=True,
                    help="path to trained model of neural network")

    args = vars(ap.parse_args())

    modelh = str(args["modelhaar"])
    modelnn = str(args["modelnn"])
    # load the image
    image_path = str(args["image"])
    # open image
    image = cv2.imread(image_path)

    # init global variables
    module.path = path
    module.pathvs = pathvs
    module.image_path = image_path

    haar_NN(image, modelh, modelnn)
