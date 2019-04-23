import matplotlib.pyplot as plt
from PIL import Image, ImageTk
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


def loaddat(data_file):
    vertex_search = []
    vs = open(data_file, 'r')

    for line in vs:
        cleanedLine = line.strip()
        x = cleanedLine.split(" ")
        vertex_search.append(x)

    return vertex_search


def inPolygon(xd, yd, x, y, w, h, r):
    if (((xd >= (x - r)) & (xd <= (x + w + r))) & ((yd >= (y - r)) & (yd <= (y + h + r)))):
        return True
    else:
        return False


def inPolygonContur(xd, yd, x, y, w, h, r):
    pix = 0
    if ( ( (abs(xd - x + r) <= pix) & (y - r - pix <= yd <= y + h + r + pix) ) | ( (abs(xd - x - w - r) <= pix) & (y - r - pix <= yd <= y + h + r + pix) ) |
           ( (abs(yd - y + r) <= pix) & (x - r - pix <= xd <= x + w + r + pix) ) | ( (abs(yd - y - h - r) <= pix) & (x - r - pix <= xd <= x + w + r + pix) ) ):
        return True
    else:
        return False


def inboxvertex(xd, yd, vs):
    for vertex in vs:
        x = int(vertex[0])
        y = int(vertex[1])
        w = int(vertex[2])
        h = int(vertex[3])
        x_c = int(vertex[4])
        y_c = int(vertex[5])
        r = int(vertex[6])
        d = int(vertex[7])

        d = math.ceil(d // 2 - 0.2 * (d // 4))

        if ((inPolygon(xd, yd, x, y, w, h, r // 4) == False) &
                (inPolygonContur(xd, yd, x, y, w, h, d) == True)):
            return True
    return False


def countUnfilteringDots(ds, i):
    filtx = [i]
    filty = [i]

    for j in ds:
        # print(j) #, i[1], j[0], i[0])
        if (j != i):
            if ((j[1] == i[1]) & (abs(j[0] - i[0]) < 9)): #  & (abs(j[0] - i[0]) < 9)
                filtx.append(j)
            elif ((j[0] == i[0]) & (abs(j[1] - i[1]) < 9)): #  & (abs(j[1] - i[1]) < 9)
                filty.append(j)

    if ((len(filtx) > 1)):
        # print(filtx)
        return filtx
    elif (len(filty) > 1):
        # print(filty)
        return filty

    return []


def filterDots_1(detcoord):
    xs = [detcoord[0][1]]
    ys = [detcoord[0][0]]
    ds = []

    for i in range(1, len(detcoord)):
        if ((detcoord[i][1] in xs) & (len(xs) == 1) & (
                (max(ys + [detcoord[i][0]]) - min(ys + [detcoord[i][0]])) <= 9)):
            # print("max(ys) - min(ys)", max(ys) - min(ys), sep=" ")
            ys.append(detcoord[i][0])
            # print("add ys ", ys)

        elif ((detcoord[i][0] in ys) & (len(ys) == 1) & (
                (max(xs + [detcoord[i][1]]) - min(xs + [detcoord[i][1]])) <= 9)):
            # print("max(xs) - min(xs)", max(xs) - min(xs), sep=" ")
            xs.append(detcoord[i][1])
            # print("add xs ", xs)

        elif ((detcoord[i][1] not in ys) | (detcoord[i][0] not in xs)):
            # print("Result: ys ", ys)
            # print("Result: xs ", xs)
            # print("Result: ", (math.ceil(sum(xs) / len(xs)), math.ceil(sum(ys) / len(ys))))

            ds.append((math.ceil(sum(ys) / len(ys)), math.ceil(sum(xs) / len(xs))))

            xs.clear()
            ys.clear()
            xs.append(detcoord[i][1])
            ys.append(detcoord[i][0])

    # print("Result: ys ", ys)
    # print("Result: xs ", xs)
    # print("Result: ", (math.ceil(sum(xs) / len(xs)), math.ceil(sum(ys) / len(ys))))
    ds.append((math.ceil(sum(ys) / len(ys)), math.ceil(sum(xs) / len(xs))))

    if len(ds) == 1:
        return ds[0]

    return ds


def filterDots_2(ds):
    for j in ds:
        filt = countUnfilteringDots(ds, j)
        if (len(filt) > 1):
            dis = filterDots_1(filt)
            for i in filt:
                if i in ds:
                    ds.remove(i)
            if dis not in ds:
                ds.append(dis)

    # print(ds, len(ds))

    for j in ds:
        filt = countUnfilteringDots(ds, j)
        if len(filt) != 0:
            print("Not filtering:", filt)

    return ds


def find_start(img, vs):
    # filters
    # gaus = cv2.GaussianBlur(img, (3, 3), 0)
    # Canny
    edges = cv2.Canny(img, 100, 255)

    kernel = np.ones((5, 5), np.uint8)
    edges_k = cv2.dilate(edges, kernel, iterations=3)

    cv2.imwrite(folder + 'Canny_k.jpg', edges_k)
    img_k = cv2.imread(folder + 'Canny_k.jpg')

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.erode(img_k, kernel, iterations=3)
    # kernel = np.ones((3, 3), np.uint8)
    # edges = cv2.erode(edges, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=2)

    # Find coordinates of edges
    indices = np.where(edges != [0])
    # print(indices)

    for vertex in vs:
        x = int(vertex[0])
        y = int(vertex[1])
        w = int(vertex[2])
        h = int(vertex[3])
        x_c = int(vertex[4])
        y_c = int(vertex[5])
        r = int(vertex[6])
        d = int(vertex[7])

        cv2.rectangle(edges, (x - r // 4, y - r // 4), (x + w + r // 4, y + h + r // 4),
                      (255, 255, 255), 1)
        k = math.ceil(0.1 * (r // 4))
        cv2.rectangle(edges, (x - k - d // 4, y - k - d // 4),
                      (x + w + k + d // 4, y + h + k + d // 4), (255, 255, 255), 1)

        d = math.ceil(d // 2 - 0.2 * (d // 4))
        cv2.rectangle(edges, (x - d, y - d), (x + w + d, y + h + d), (255, 255, 255), 1)

    cv2.imwrite(folder + 'canny.jpg', edges)

    # make coordinates in simple form
    detcoord = []
    img_c = img.copy()

    for i in range(0, len(indices[0])):
        if (inboxvertex(indices[1][i], indices[0][i], vs) == True):
            if ([indices[0][i], indices[1][i]] not in detcoord):
                detcoord.append([indices[0][i], indices[1][i]])
                cv2.circle(img_c, (indices[1][i], indices[0][i]), 3, (255, 255, 255), 1)

    cv2.imwrite(folder + 'line_start_before_filtering.jpg', img_c)
    # print(detcoord)

    # Filtering
    preds = filterDots_1(detcoord)
    ds = filterDots_2(preds)

    datcoord = folder + 'Startlines_y_x' + '.txt'
    dat = open(datcoord, 'w')

    # dat.write('y x\n')
    for i in ds:
        dat.write(str(i[0]) + ' ' + str(i[1]) + '\n')
        cv2.circle(img, (i[1], i[0]), 10, (255, 255, 255), 1)

    dat.close()

    detcoord = ds

    cv2.imwrite(folder + 'line_start.jpg', img)

    return detcoord


def dis(x1, x2, y1, y2, r1, r2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2) - r1 - r2


def d(x1, x2, y1, y2, r1, r2):
    return max(abs(x2 - x1), abs(y2 - y1)) - r1 - r2


def getDistanceVertex(vs):
    newvs = []

    for i in range(0, len(vs)):
        distance = []

        xi = int(vs[i][0])
        yi = int(vs[i][1])
        wi = int(vs[i][2])
        hi = int(vs[i][3])
        x1 = int(vs[i][4])
        y1 = int(vs[i][5])
        r1 = math.ceil((math.ceil(wi / 2) + math.ceil(hi / 2)) / 2)

        for j in range(0, len(vs)):
            if j != i:
                xj = int(vs[j][0])
                yj = int(vs[j][1])
                wj = int(vs[j][2])
                hj = int(vs[j][3])
                x2 = int(vs[j][4])
                y2 = int(vs[j][5])
                r2 = math.ceil((math.ceil(wj / 2) + math.ceil(hj / 2)) / 2)

                dist = d(x1, x2, y1, y2, r1, r2)
                distance.append(dist)

        # print(distance)
        d1 = min(distance)
        # print(d1)

        newvs.append([xi, yi, wi, hi, x1, y1, r1, d1])

    return newvs


def line_start(image):
    img = cv2.imread(image)

    img_c = img.copy()

    gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 2)

    edges = cv2.Canny(gray, 100, 255)

    vs = loaddat(path)

    newvs = getDistanceVertex(vs)

    # img_rect = img.copy()
    cropgray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    # crop_gray = cv2.GaussianBlur(cropgray, (kernel_size, kernel_size), -1)
    cropedges = find_start(cropgray, newvs)

    # print('\n')
    # print(cropedges)

    iter = 0
    for vertex in newvs:
        x = int(vertex[0])
        y = int(vertex[1])
        w = int(vertex[2])
        h = int(vertex[3])
        x_c = int(vertex[4])
        y_c = int(vertex[5])
        r = int(vertex[6])
        d = int(vertex[7])

        cv2.rectangle(img_c, (x - r // 4, y - r // 4), (x + w + r // 4, y + h + r // 4),
                      (255, 0, 255), 2)
        k = math.ceil(0.1*(r // 4))
        cv2.rectangle(img_c, (x - k - d // 4, y - k - d // 4),
                      (x + w + k + d // 4, y + h + k + d // 4), (0, 215, 255), 2)

        d = math.ceil(d // 2 - 0.2*(d // 4))
        cv2.rectangle(img_c, (x - d, y - d), (x + w + d, y + h + d), (0, 165, 255), 2)

        iter += 1

    for i in cropedges:
        cv2.circle(img_c, (i[1], i[0]), 10, (255, 0, 255), 3)

    cv2.imwrite(folder + "result.jpg", img_c)


if __name__ == '__main__':
    # create folder
    folder = './result/start_lines/'
    createFolder(folder)

    # image
    image = sys.argv[1]

    # path to folder
    path = sys.argv[2] #"./vs2/vertex_search.graphvs"

    line_start(image)

    sys.exit(0)
