import numpy as np
import shutil
import math
import sys
import cv2
import os


# Common functions
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


def updateDvertex(d):
    return math.ceil(d // 2 - 0.4 * (d // 4))


'''Start line functions'''
# Load information and refactoring list with coordinates
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

# filter Dots
def inPolygon(xd, yd, x, y, w, h, r):
    if (((xd >= (x - r)) & (xd <= (x + w + r))) & ((yd >= (y - r)) & (yd <= (y + h + r)))):
        return True
    else:
        return False


def inPolygonContur(xd, yd, x, y, w, h, r, pix = 0):
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

        d = updateDvertex(d)

        if ((inPolygon(xd, yd, x, y, w, h, r // 4) == False) &
                (inPolygonContur(xd, yd, x, y, w, h, d, 0) == True)):
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


def filterDots_1(dotcoord):
    xs = [dotcoord[0][1]]
    ys = [dotcoord[0][0]]
    ds = []

    for i in range(1, len(dotcoord)):
        if ((dotcoord[i][1] in xs) & (len(xs) == 1) & (
                (max(ys + [dotcoord[i][0]]) - min(ys + [dotcoord[i][0]])) <= 9)):
            # print("max(ys) - min(ys)", max(ys) - min(ys), sep=" ")
            ys.append(dotcoord[i][0])
            # print("add ys ", ys)

        elif ((dotcoord[i][0] in ys) & (len(ys) == 1) & (
                (max(xs + [dotcoord[i][1]]) - min(xs + [dotcoord[i][1]])) <= 9)):
            # print("max(xs) - min(xs)", max(xs) - min(xs), sep=" ")
            xs.append(dotcoord[i][1])
            # print("add xs ", xs)

        elif ((dotcoord[i][1] not in ys) | (dotcoord[i][0] not in xs)):
            # print("Result: ys ", ys)
            # print("Result: xs ", xs)
            # print("Result: ", (math.ceil(sum(xs) / len(xs)), math.ceil(sum(ys) / len(ys))))

            ds.append((math.ceil(sum(ys) / len(ys)), math.ceil(sum(xs) / len(xs))))

            xs.clear()
            ys.clear()
            xs.append(dotcoord[i][1])
            ys.append(dotcoord[i][0])

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