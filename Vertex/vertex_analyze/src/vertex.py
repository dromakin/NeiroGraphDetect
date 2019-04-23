# USAGE
# without features
# python src/vertex.py --image ./data/exe.jpg --model ./models/haar/haar3030_realise/cascade.xml --Filters 0 --Params 0

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import shutil
import math
import cv2
import sys
import os

def neural_network_2828(image):
    # debug log
    logs = path + 'debug.log'
    log = open(logs, 'a')

    orig = image
    log.write("OUTPUT:\norig = image\n\n")

    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    log.write("OUTPUT:\ncv2.resize(image, (28, 28))\n\n")

    image = image.astype("float") / 255.0
    log.write("OUTPUT:\nimage.astype(float) / 255.0\n\n")

    image = img_to_array(image)
    log.write("OUTPUT:\nimg_to_array(image)\n\n")

    image = np.expand_dims(image, axis=0)
    log.write("OUTPUT:\nnp.expand_dims(image, axis=0)\n\n")

    # load the trained convolutional neural network
    log.write("OUTPUT:\n[INFO] loading network...\n\n")
    #
    # NOT_VERTEX
    # model = load_model('./models/neural_networks/not_vertex_sort.model')
    #
    # # classify the input image
    # (vertex, notVertex) = model.predict(image)[0]

    # # VERTEX
    model = load_model('./models/neural_networks/vertex_sort.model')
    log.write("OUTPUT:\nload_model('./models/neural_networks/vertex_sort.model')\n\n")

    # classify the input image
    (notVertex, vertex) = model.predict(image)[0]
    log.write("OUTPUT:\nmodel.predict(image)[0]\n\n")

    # build the label
    label = "Vertex" if vertex > notVertex else "Not Vertex"
    proba = vertex if vertex > notVertex else notVertex
    labelprob = "{}: {:.2f}%".format(label, proba * 100)
    log.write("OUTPUT:\nformat(label, proba * 100)\n\n")

    log.write("OUTPUT:\nreturn\n\n")

    return (label, labelprob, str(float("{0:.2f}".format(proba * 100))))


def drawrectangle(img, vertex, gray=False, gaus=False): #, gray_gaus=False):
    # debug log
    logs = path + 'debug.log'
    log = open(logs, 'a')
    #
    try:
        # vertex30
        it = 0
        # get rectangle
        log.write("OUTPUT:\nGet rectangle(-s):\n\n")
        #
        for (x, y, w, h) in vertex:
            crop_img = img[y:y + h, x:x + w]
            label = neural_network_2828(crop_img)
            log.write("OUTPUT:\nGet label:\n" + str(label) + "\n\n")

            # put text
            log.write("OUTPUT:\nPut text:\n\n")
            #
            if (label[0] == "Vertex"):
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'V' + label[2], (x - 2, y - 2), font, 0.5, (255, 0, 255), 1,
                            cv2.LINE_AA)
                #
                log.write("OUTPUT:\nVERTEX:     cv2.putText(img, 'V'+label[2], (x - 2, y - 2), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)\n\n")

            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'NV' + label[2], (x - 2, y - 2), font, 0.5, (0, 255, 0), 1,
                            cv2.LINE_AA)
                #
                log.write(
                    "OUTPUT:\ncv2.putText(img, 'NV'+label[2], (x - 2, y - 2), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)\n\n")

            log.write("OUTPUT:\nSTAGE: " + str(it) + '\n')
            it += 1

        # save image
        x = image.split("/")
        log.write("OUTPUT:\nimage.split()\n\n")
        # write results
        # path = './results/result_Haar_NN/'
        cv2.imwrite(path + x[-1], img)
        log.write("OUTPUT:\nimwrite(path + x[-1], img)\n\n")

        if gray == True:
            cv2.imwrite(path + 'gray_' + x[-1], img)
            log.write("OUTPUT:\nimwrite(path + x[-1], img)\n\n")
        elif gaus == True:
            cv2.imwrite(path + 'gaus_' + x[-1], img)
            log.write("OUTPUT:\nimwrite(path + x[-1], img)\n\n")
        # elif gray_gaus == True:
        #     cv2.imwrite(path + 'gray_gaus_' + x[-1], img)
        #     log.write("OUTPUT:\nimwrite(path + x[-1], img)\n\n")

    except:
        log.write("DEBUG: Unexpected error:" + sys.exc_info())

    log.close()


def detection(img, img_c, model, Filters=False, Params=False):
    # debug log
    logs = path + 'debug.log'
    log = open(logs, 'a')
    #
    try:
        # This is the cascade we just made. Call what you want
        cascade30 = cv2.CascadeClassifier(model)
        log.write("OUTPUT:\nRead model\n\n")

        if Params == False:
            # without any params
            vertex30 = cascade30.detectMultiScale(img_c)
            log.write("OUTPUT:\ncv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)\n\n")

            # draw rectangles
            drawrectangle(img, vertex30)

            # filters
            if Filters == True:
                # GRAY
                gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
                img_gray = np.zeros_like(img_c)
                img_gray[:, :, 0] = gray
                img_gray[:, :, 1] = gray
                img_gray[:, :, 2] = gray
                log.write("OUTPUT:\ncascade30.detectMultiScale(img_c)\n\n")

                vertex30_gray = cascade30.detectMultiScale(img_gray)
                log.write("OUTPUT:\ncascade30.detectMultiScale(gray)\n\n")

                drawrectangle(img_gray, vertex30_gray, gray=True)

                # GAUS
                gaus = cv2.GaussianBlur(img_c, (5, 5), 2)
                log.write("OUTPUT:\ncv2.GaussianBlur(img_c, (5, 5), 2)\n\n")

                vertex30_gaus = cascade30.detectMultiScale(gaus)
                log.write("OUTPUT:\ncascade30.detectMultiScale(gaus)\n\n")

                drawrectangle(gaus, vertex30_gaus, gaus=True)

                # GRAY_GAUS
                # gray_gaus = cv2.GaussianBlur(img_gray, (5, 5), 2)
                # log.write("OUTPUT:\ncv2.GaussianBlur(gray, (5, 5), 2)\n\n")
                #
                # vertex30_gray_gaus = cascade30.detectMultiScale(gray_gaus)
                # log.write("OUTPUT:\ncascade30.detectMultiScale(gray_gaus)\n\n")
                #
                # drawrectangle(gray_gaus, vertex30_gray_gaus, gray_gaus=True)

        else:
            # without any params
            h, w, ch = img_c.shape
            vertex30 = cascade30.detectMultiScale(img_c, scaleFactor=1.02, minSize=(10,10),
                                                  maxSize=(math.ceil(h/3), math.ceil(w/3)))
            log.write("OUTPUT:\ncv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)\n\n")

            # draw rectangles
            drawrectangle(img, vertex30)

            # filters
            if Filters == True:
                # GRAY
                gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
                img_gray = np.zeros_like(img_c)
                img_gray[:, :, 0] = gray
                img_gray[:, :, 1] = gray
                img_gray[:, :, 2] = gray
                log.write("OUTPUT:\ncascade30.detectMultiScale(img_c)\n\n")

                vertex30_gray = cascade30.detectMultiScale(img_gray, scaleFactor=1.02, minSize=(10,10),
                                                           maxSize=(math.ceil(h/3), math.ceil(w/3)))
                log.write("OUTPUT:\ncascade30.detectMultiScale(gray)\n\n")

                drawrectangle(img_gray, vertex30_gray, gray=True)

                # GAUS
                gaus = cv2.GaussianBlur(img_c, (5, 5), 2)
                log.write("OUTPUT:\ncv2.GaussianBlur(img_c, (5, 5), 2)\n\n")

                vertex30_gaus = cascade30.detectMultiScale(gaus, scaleFactor=1.02, minSize=(10,10),
                                                           maxSize=(math.ceil(h/3), math.ceil(w/3)))
                log.write("OUTPUT:\ncascade30.detectMultiScale(gaus)\n\n")

                drawrectangle(gaus, vertex30_gaus, gaus=True)

                # GRAY_GAUS
                # gray_gaus = cv2.GaussianBlur(img_gray, (5, 5), 2)
                # log.write("OUTPUT:\ncv2.GaussianBlur(gray, (5, 5), 2)\n\n")
                #
                # vertex30_gray_gaus = cascade30.detectMultiScale(gray_gaus, scaleFactor=1.02,
                #                                                 minSize=(10,10),
                #                                                 maxSize=(math.ceil(h/3),
                #                                                          math.ceil(w/3)))
                # log.write("OUTPUT:\ncascade30.detectMultiScale(gray_gaus)\n\n")
                #
                # drawrectangle(gray_gaus, vertex30_gray_gaus, gray_gaus=True)


    except:
        log.write("DEBUG: Unexpected error:" + sys.exc_info()[0])

    log.close()


def haarvertex(image, model, filters, params):
    # debug log
    logs = path + 'debug.log'
    log = open(logs, 'w')

    try:
        # read image
        img = cv2.imread(image)
        log.write("OUTPUT:\nRead image: " + str(image) + '\n\n')

        # get exception if None
        if img is None:
            log.write("DEBUG: Error cv2.imread(image): " + str(image) + '\n\n')

        # copy image
        img_c = img.copy()
        log.write("OUTPUT:\nCopy original image: " + str(img[-1]) + '\n\n')

        detection(img, img_c, model, filters, params)

    except:
        log.write("DEBUG: Unexpected error" + sys.exc_info()[0])

    log.close()


def createFolder(directory):
    # debug log
    logs = './output.log'
    log = open(logs, 'w')
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
    except OSError:
        log.write('DEBUG:\nError: Creating directory. ' + directory + '\n')

    log.close()


if __name__ == "__main__":
    # create dir
    path = './results/result_Haar_NN/'
    # os.mkdir(path)
    createFolder(path)

    # errors
    sys.stdout = open(path + 'output.log', 'w')

    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")

    ap.add_argument("-m", "--model", required=True,
                    help="path to trained model model")

    ap.add_argument("-f", "--Filters", required=True,
                    help="True/False")

    ap.add_argument("-p", "--Params", required=True,
                    help="True/False")

    args = vars(ap.parse_args())

    # load the image
    image = str(args["image"])
    model = str(args["model"])
    filters = bool(int(args["Filters"]))
    params = bool(int(args["Params"]))
    haarvertex(image, model, filters, params)
