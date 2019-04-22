from keras.preprocessing.image import img_to_array
from keras.models import load_model
import math
import sys
import os

libdir = os.path.dirname('./src/')
sys.path.append(os.path.split(libdir)[0])
from src.libs.lib_preprocessing import *

# global
path = str()
pathvs = str()
image_path = str()


'''1 STEP: Haar detection'''
def haarDetect(img, img_c, modelh, modelnn):
    '''
    Detect circles with Haar function

    :param img: original image
    :param img_c: copy image
    :param modelh: model Haar
    :param modelnn: model neural network
    :return:
    '''
    try:
        orig_img = img.copy()
        cascade30 = cv2.CascadeClassifier(modelh)  # Haar Cascade

        '''
        Code for testing haar detection to original image (without filtering)
        # without any filtering
        # vertex30 = cascade30.detectMultiScale(img_c)
        # nnFilering_DrawRect(orig_img, img, vertex30, modelnn)
        '''

        # GRAY
        gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
        img_gray = np.zeros_like(img_c)
        img_gray[:, :, 0] = gray
        img_gray[:, :, 1] = gray
        img_gray[:, :, 2] = gray

        vertex30_gray = cascade30.detectMultiScale(img_gray)  # detect our vertex

        ''' NN Classifer'''
        nnFilering_DrawRect(orig_img, img_gray, vertex30_gray, modelnn, gray=True)

    except:
        print("DEBUG_ERROR: Unexpected error:" + sys.exc_info()[0])


'''Load Neural Network'''
def neural_network_2828(image, modelnn):
    '''
    Neural network

    :param image: image
    :param modelnn: model of NN
    :return: label with percents of classification truth
    '''
    orig = image.copy()
    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # load the trained convolutional neural network
    print("[INFO] loading network...")

    '''2 NN for Classifer Testing *.model and not_*.model
    *.model     - train by positive samples
    not_*.model - train by neg samples
    # NOT_VERTEX
    # model = load_model('./models/neural_networks/not_vertex_sort.model')
    #
    # # classify the input image
    # (vertex, notVertex) = model.predict(image)[0]
    '''

    # load NN
    model = load_model(modelnn)     # modelnn - path to your model
    # classify the input image
    (notVertex, vertex) = model.predict(image)[0]
    # build the label
    label = "Vertex" if vertex > notVertex else "Not Vertex"
    proba = vertex if vertex > notVertex else notVertex
    # labelprob = "{}: {:.2f}%".format(label, proba * 100)
    return (label, round(proba * 100, 2), str(float("{0:.2f}".format(proba * 100))))


'''2 STEP: NN Classifer'''
def nnFilering_DrawRect(orig, img, vertex, modelnn, gray=True):
    '''

    :param orig: original image
    :param img: image
    :param vertex: list of detect circles
    :param modelnn: path to model
    :param gray: Convert to gray scale? True or False
    :return: none
    '''
    img_c = orig.copy()
    try:
        # vertex search
        vertex_sort = []

        for (x, y, w, h) in vertex:
            # crop
            crop_img = img[y:y + h, x:x + w]
            # NN
            label = neural_network_2828(crop_img, modelnn)
            # Classificaton -> Filtering
            if ((label[0] == "Vertex") & (label[1] > 54)):      # > 54% Vertex
                cv2.rectangle(img_c, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_c, 'V' + label[2], (x - 2, y - 2), font, 0.5, (255, 0, 255), 1,
                            cv2.LINE_AA)
                cv2.putText(img, 'V' + label[2], (x - 2, y - 2), font, 0.5, (255, 0, 255), 1,
                            cv2.LINE_AA)
                # write center
                x_c = x + math.ceil(w / 2)
                y_c = y + math.ceil(h / 2)
                vertex_sort.append([x, y, w, h, x_c, y_c, label[1]])
            else:
                cv2.rectangle(img_c, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_c, 'NV' + label[2], (x - 2, y - 2), font, 0.5, (0, 255, 0), 1,
                            cv2.LINE_AA)
                cv2.putText(img, 'NV' + label[2], (x - 2, y - 2), font, 0.5, (0, 255, 0), 1,
                            cv2.LINE_AA)

        x = image_path.split("/")
        if gray == True:
            saveImage(path + 'gray_' + x[-1], img)

        # VS
        # save original image
        saveImage(path + x[-1], orig)

        '''Vertex Filter after NN Classifer'''
        vertex_lines = filtering_vertex(orig, vertex_sort)

        # center of rectangles
        vertexfs = open(pathvs + 'vertex_lines.graphvs', 'w')
        vertex_predict = open(pathvs + 'vertex_predict.graphvs', 'w')

        if len(vertex_lines) == 0:
            vertex_lines = vertex_sort

        for (x, y, w, h, x_c, y_c, label) in vertex_lines:
            if gray == True:
                # for lines
                vertexfs.write(str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + ' '
                               + str(x_c) + ' ' + str(y_c) + '\n')
                r = math.ceil((math.ceil(w / 2) + math.ceil(h / 2)) / 2)    # get radius for write circle
                # result
                vertex_predict.write(str(x_c) + ' ' + str(y_c) + ' ' + str(r) + '\n')

        vertexfs.close()
        vertex_predict.close()

    except:
        print("DEBUG_ERROR: Unexpected error:" + sys.exc_info())

'''VF: return mean radius of circle vertex'''
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

'''VF: Check is (xd, yd) in rectangle'''
def inPolygon(xd, yd, x, y, w, h):
    if (((xd >= x) & (xd <= (x + w))) & ((yd >= y) & (yd <= (y + h)))):
        return True
    else:
        return False

'''3 STEP: Vertex Filter'''
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

    print('DEBUG_INFO: actions: ', actions)

    # save action image
    x = image_path.split("/")
    saveImage(path + 'actions_' + x[-1], img_act)

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
    vertex_predict = orig.copy()

    for i in range(0, len(vertex_sort)):
        xd = vertex_sort[i][0]
        yd = vertex_sort[i][1]
        wd = vertex_sort[i][2]
        hd = vertex_sort[i][3]
        x_d = vertex_sort[i][4]
        y_d = vertex_sort[i][5]
        rd = math.ceil((math.ceil(wd / 2) + math.ceil(hd / 2)) / 2)
        # rect
        cv2.rectangle(img, (xd, yd), (xd + wd, yd + hd), (255, 0, 255), 2)
        # circles
        cv2.circle(vertex_predict, (x_d, y_d), rd, (255, 0, 255), 2)

    # save rect image
    x = image_path.split("/")
    saveImage(path + 'filtering_' + x[-1], img)
    saveImage(pathvs + 'vertex_predict_' + x[-1], vertex_predict)

    return filterVS