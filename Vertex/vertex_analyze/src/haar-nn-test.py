# USAGE
# python haar-nn-test.py --image ./test/1.JPG
# python src/haar-nn-test.py --dir ./data/tests/test/

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import shutil
import cv2
import os

def neural_network_2828(image):
    orig = image
    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    #
    # NOT_VERTEX
    # model = load_model('./models/neural_networks/not_vertex_sort.model')
    #
    # # classify the input image
    # (vertex, notVertex) = model.predict(image)[0]

    # # VERTEX
    model = load_model('./models/neural_networks/vertex_sort.model')

    # classify the input image
    (notVertex, vertex) = model.predict(image)[0]

    # build the label
    label = "Vertex" if vertex > notVertex else "Not Vertex"
    proba = vertex if vertex > notVertex else notVertex
    labelprob = "{}: {:.2f}%".format(label, proba * 100)

    # draw the label on the image
    # output = imutils.resize(orig, width=400)
    # cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return (label, labelprob, str(float("{0:.2f}".format(proba * 100))))


def haartest(image):
    # This is the cascade we just made. Call what you want
    cascade30 = cv2.CascadeClassifier('./models/haar/haar4040/cascade.xml')

    img = cv2.imread(image)
    if img is None:
        print('Error!!!!')
    img_c = img.copy()

    # filters
    gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    # gaus = cv2.GaussianBlur(gray, (5, 5), 2)

    vertex30 = cascade30.detectMultiScale(gray)

    it = 0
    # get rectangle
    for (x, y, w, h) in vertex30:
        crop_img = img[y:y + h, x:x + w]
        label = neural_network_2828(crop_img)

        # put text
        if (label[0] == "Vertex"):
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'V'+label[2], (x - 2, y - 2), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'NV'+label[2], (x - 2, y - 2), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        it += 1


    # save image
    x = image.split("/")
    print(x[-1])
    path = './results/result_Haar_NN_test/'
    # os.mkdir(path)

    # cv2.imshow('img',img)
    cv2.imwrite(path + x[-1], img)
    # exit(0)
    # cv2.waitKey()

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


if __name__ == "__main__":
    # create dir
    path = './results/result_Haar_NN_test/'
    # os.mkdir(path)
    createFolder(path)

    ap = argparse.ArgumentParser()
    # ap.add_argument("-m", "--model", required=True,
    #                 help="path to trained model model")
    # ap.add_argument("-i", "--image", required=True,
    #                 help="path to input image")
    ap.add_argument("-d", "--dir", required=True,
                    help="path with example images")
    args = vars(ap.parse_args())
    # load the image
    # name = str(args["image"])
    # print(name)
    # haartest(name)

    dir = str(args["dir"])
    # print(dir)
    listOfexamples = os.listdir(dir)
    wastefiles = ['.DS_Store', '.ipynb_checkpoints']
    for i in listOfexamples:
        if i not in wastefiles:
            haartest(dir + i)



