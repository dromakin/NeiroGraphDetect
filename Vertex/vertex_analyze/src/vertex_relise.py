# USAGE
# python src/vertex_relise.py --image ./data/exe6.jpg --modelh ./models/haar/haar3030_realise/cascade.xml --modelnn ./models/neural_networks/vertex_sort.model
# python src/vertex_relise.py --image ./data/exe9.jpg --modelh ./models/haar/haar3030_realise/cascade.xml --modelnn ./models/neural_networks/vertex_sort.model

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


def getRmean(vertex_sort, r_ignore):
    sum = 0
    length = 0

    for j in range(0, len(vertex_sort)):
        x = vertex_sort[j][0]
        y = vertex_sort[j][1]
        w = vertex_sort[j][2]
        h = vertex_sort[j][3]
        r = math.ceil((math.ceil(w / 2) + math.ceil(h / 2)) / 2)

        if r != r_ignore:
            sum += r
            length += 1

    return math.ceil(sum / length)


def inPolygon(xd, yd, x, y, w, h):
    if (((xd >= x) & (xd <= (x + w))) & ((yd >= y) & (yd <= (y + h)))):
        return True
    else:
        return False


def filtering_vertex(orig, vertex_sort):
    # draw actions rectangle(-s)
    img_act = orig.copy()

    # filtering list
    filterVS = []

    # get Actions
    actions = []
    for i in range(0, len(vertex_sort)):
        xd = vertex_sort[i][0]
        yd = vertex_sort[i][1]
        wd = vertex_sort[i][2]
        hd = vertex_sort[i][3]

        for j in range(i + 1, len(vertex_sort)):
            x = vertex_sort[j][0]
            y = vertex_sort[j][1]
            w = vertex_sort[j][2]
            h = vertex_sort[j][3]
            if ((inPolygon(xd,yd,x,y,w,h) == True) | (inPolygon(xd+wd,yd+hd,x,y,w,h) == True) |
                    (inPolygon(xd,yd+hd,x,y,w,h) == True) | (inPolygon(xd+wd,yd,x,y,w,h) == True)):
                actions.append((i,j))
                cv2.rectangle(img_act, (xd, yd), (xd + wd, yd + hd), (255, 0, 255), 2)
                cv2.rectangle(img_act, (x, y), (x + w, y + h), (255, 0, 255), 2)

    print('actions: ', actions)

    # save action image
    x = image.split("/")
    cv2.imwrite(path + 'actions_' + x[-1], img_act)

    # filtering
    if len(actions) == 1:
        print("len == 1")
        # 1 filter
        r_mean = getRmean(vertex_sort, 0)
        print("r_mean: ", r_mean)

        i = actions[0][0]

        xi = vertex_sort[i][0]
        yi = vertex_sort[i][1]
        wi = vertex_sort[i][2]
        hi = vertex_sort[i][3]
        # x_c -> 4 y_c -> 5
        pi = vertex_sort[i][6]
        ri = math.ceil((math.ceil(wi / 2) + math.ceil(hi / 2)) / 2)
        ri_mean = getRmean(vertex_sort, ri)
        #
        print('ri_mean', ri_mean)
        razi = (r_mean - ri_mean) > 0.1 * r_mean

        j = actions[0][1]

        xj = vertex_sort[j][0]
        yj = vertex_sort[j][1]
        wj = vertex_sort[j][2]
        hj = vertex_sort[j][3]
        # x_c -> 4 y_c -> 5
        pj = vertex_sort[j][6]
        rj = math.ceil((math.ceil(wj / 2) + math.ceil(hj / 2)) / 2)
        rj_mean = getRmean(vertex_sort, rj)
        #
        print('rj_mean', rj_mean)
        razj = (r_mean - rj_mean) > 0.1 * r_mean

        if (abs(ri - rj) <= 0.1*max(ri, rj)):
            if ((razi == True) & (razj == False)):
                print("i")
                filterVS = vertex_sort
                del filterVS[i]

                print('filterVS: ', filterVS)
            elif ((razi == False) & (razj == True)):
                print("j")
                filterVS = vertex_sort
                del filterVS[j]

                print('filterVS: ', filterVS)
            elif ((razi == False) & (razj == False)):
                print("Variant rectangles:")

                # 1
                if ((inPolygon(xi, yi, xj, yj, wj, hj) == True)):
                    filterVS = vertex_sort

                    filterVS[i][0] = xj
                    filterVS[i][1] = yj
                    filterVS[i][2] = wj + (xi + wi - xj - wj)
                    filterVS[i][3] = hj + (yi + hi - yj - hj)
                    filterVS[i][4] = xj + math.ceil(filterVS[i][2] / 2)
                    filterVS[i][5] = yj + math.ceil(filterVS[i][3] / 2)

                    del filterVS[j]
                # 1.2
                elif ((inPolygon(xj, yj, xi, yi, wi, hi) == True)):
                    filterVS = vertex_sort

                    filterVS[j][0] = xi
                    filterVS[j][1] = yi
                    filterVS[j][2] = wi + (xj + wj - xi - wi)
                    filterVS[j][3] = hi + (yj + hj - yi - hi)
                    filterVS[j][4] = xi + math.ceil(filterVS[j][2] / 2)
                    filterVS[j][5] = yi + math.ceil(filterVS[j][3] / 2)

                    del filterVS[i]
                # 2
                elif ((inPolygon(xi + wi, yi, xj, yj, wj, hj) == True)):
                    filterVS = vertex_sort

                    filterVS[i][0] = xi
                    filterVS[i][1] = yj
                    filterVS[i][2] = wi + (xi - xj)
                    filterVS[i][3] = hj + (yi - yj)
                    filterVS[i][4] = xi + math.ceil(filterVS[i][2] / 2)
                    filterVS[i][5] = yj + math.ceil(filterVS[i][3] / 2)

                    del filterVS[j]
                # 2.1
                elif ((inPolygon(xj + wj, yj, xi, yi, wi, hi) == True)):
                    filterVS = vertex_sort

                    filterVS[j][0] = xj
                    filterVS[j][1] = yi
                    filterVS[j][2] = wj + (xj - xi)
                    filterVS[j][3] = hi + (yj - yi)
                    filterVS[j][4] = xj + math.ceil(filterVS[j][2] / 2)
                    filterVS[j][5] = yi + math.ceil(filterVS[j][3] / 2)

                    del filterVS[i]

                print('filterVS: ', filterVS)
            elif ((razi == True) & (razj == True)):
                filterVS = vertex_sort

                del filterVS[i]
                del filterVS[j]

                print('filterVS: ', filterVS)
        else:
            if pi > pj:
                filterVS = vertex_sort

                del filterVS[j]
                print('filterVS: ', filterVS)
            elif pi <= pj:
                filterVS = vertex_sort

                del filterVS[i]
                print('filterVS: ', filterVS)
    elif len(actions) > 1:

        act = actions[0]
        act_1 = act[0]
        act_2 = act[1]
        #act_f = []
        act_f = -1
        act_ff = []
        act_ff.append((act[0], act[1]))

        for i in range(1, len(actions)):
            acti = actions[i]
            if act_1 == acti[0]:
                act_f = acti[0]
            elif act_2 == acti[0]:
                act_f = acti[0]
            elif act_1 == acti[1]:
                act_f = acti[1]
            elif act_2 == acti[1]:
                act_f = acti[1]
            else:
                act_ff.append((acti[0], acti[1]))

        if act_f != -1:
            filterVS = vertex_sort
            print('act_filtering: ', act_f)
            del(filterVS[act_f])
        else:
            for i in act_ff:
                per0 = float(vertex_sort[i[0]][6])
                per1 = float(vertex_sort[i[1]][6])

                if per0 > per1:
                    filterVS = vertex_sort
                    del(filterVS[i[1]])

                elif per0 < per1:
                    filterVS = vertex_sort
                    del(filterVS[i[0]])



        print('filterVS: ', filterVS)

    img = orig.copy()

    for i in range(0, len(vertex_sort)):
        xd = vertex_sort[i][0]
        yd = vertex_sort[i][1]
        wd = vertex_sort[i][2]
        hd = vertex_sort[i][3]
        cv2.rectangle(img, (xd, yd), (xd + wd, yd + hd), (255, 0, 255), 2)

    # save rect image
    x = image.split("/")
    cv2.imwrite(path + 'filtering_' + x[-1], img)

    return filterVS


def neural_network_2828(image, modelnn):
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
    model = load_model(modelnn)
    log.write("OUTPUT:\nload_model(modelnn)\n\n")

    # classify the input image
    (notVertex, vertex) = model.predict(image)[0]
    log.write("OUTPUT:\nmodel.predict(image)[0]\n\n")

    # build the label
    label = "Vertex" if vertex > notVertex else "Not Vertex"
    proba = vertex if vertex > notVertex else notVertex
    labelprob = "{}: {:.2f}%".format(label, proba * 100)
    log.write("OUTPUT:\nformat(label, proba * 100)\n\n")

    log.write("OUTPUT:\nreturn\n\n")

    return (label, round(proba * 100, 2), str(float("{0:.2f}".format(proba * 100))))


def drawrectangle(orig, img, vertex, modelnn, gray=False):
    # debug log
    logs = path + 'debug.log'
    log = open(logs, 'a')
    #
    try:
        # center of rectangles
        graphvs = open(path + 'vertex_search.graphvs', 'w')

        it = 0

        # vertex search
        vertex_sort = []

        # get rectangle
        log.write("OUTPUT:\nGet rectangle(-s):\n\n")
        #
        for (x, y, w, h) in vertex:
            crop_img = img[y:y + h, x:x + w]
            label = neural_network_2828(crop_img, modelnn)
            log.write("OUTPUT:\nGet label:\n" + str(label) + "\n\n")

            # put text
            log.write("OUTPUT:\nPut text:\n\n")
            #
            if ((label[0] == "Vertex") & (label[1] > 54)):
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'V' + label[2], (x - 2, y - 2), font, 0.5, (255, 0, 255), 1,
                            cv2.LINE_AA)
                # write center
                x_c = x + math.ceil(w / 2)
                y_c = y + math.ceil(h / 2)
                vertex_sort.append([x, y, w, h, x_c, y_c, label[1]])
                #
                if gray == True:
                    graphvs.write(str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + ' ' + str(x_c) + ' ' + str(y_c) + '\n')
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

        graphvs.close()


        # VS
        # save original image
        cv2.imwrite(pathvs + x[-1], orig)
        gr = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(pathvs + 'gray_' + x[-1], gr)
        # VS Filter
        vertexFS = filtering_vertex(orig, vertex_sort)
        # center of rectangles
        vertexfs = open(pathvs + 'vertexFS.graphvs', 'w')

        for (x, y, w, h, x_c, y_c, label) in vertexFS:
            if gray == True:
                vertexfs.write(str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + ' ' + str(x_c) + ' ' + str(y_c) + '\n')

        vertexfs.close()

    except:
        log.write("DEBUG: Unexpected error:" + sys.exc_info())

    log.close()


def detection(img, img_c, modelh, modelnn):
    orig_img = img.copy()
    # debug log
    logs = path + 'debug.log'
    log = open(logs, 'a')
    #
    try:
        # This is the cascade we just made. Call what you want
        cascade30 = cv2.CascadeClassifier(modelh)
        log.write("OUTPUT:\nRead model\n\n")

        # without any filtering
        vertex30 = cascade30.detectMultiScale(img_c)
        log.write("OUTPUT:\ncv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)\n\n")

        drawrectangle(orig_img, img, vertex30, modelnn)

        # GRAY
        gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
        img_gray = np.zeros_like(img_c)
        img_gray[:, :, 0] = gray
        img_gray[:, :, 1] = gray
        img_gray[:, :, 2] = gray
        log.write("OUTPUT:\ncascade30.detectMultiScale(img_c)\n\n")

        vertex30_gray = cascade30.detectMultiScale(img_gray)
        log.write("OUTPUT:\ncascade30.detectMultiScale(gray)\n\n")

        drawrectangle(orig_img, img_gray, vertex30_gray, modelnn, gray=True)

    except:
        log.write("DEBUG: Unexpected error:" + sys.exc_info()[0])

    log.close()


def haarvertex(image, modelh, modelnn):
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

        detection(img, img_c, modelh, modelnn)

    except:
        log.write("DEBUG: Unexpected error" + sys.exc_info()[0])

    log.close()


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
    except OSError:
        print('DEBUG:\nError: Creating directory. ' + directory + '\n')


if __name__ == "__main__":
    # create VS dir
    pathvs = './vs/'
    createFolder(pathvs)

    path = './vs/debug_Haar_NN/'

    # create dir
    createFolder(path)

    # errors
    sys.stdout = open(path + 'output.log', 'w')

    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")

    ap.add_argument("-mh", "--modelhaar", required=True,
                    help="path to trained model of haar")

    ap.add_argument("-mnn", "--modelnn", required=True,
                    help="path to trained model of neural network")

    args = vars(ap.parse_args())

    # load the image
    image = str(args["image"])
    modelh = str(args["modelhaar"])
    modelnn = str(args["modelnn"])
    haarvertex(image, modelh, modelnn)
