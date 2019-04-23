# python ./src/tracker_reliase.py ./data/vs1/exe1.jpg ./data/vs1/vertexFS.graphvs
# python ./src/tracker_reliase.py ./data/vs/exe6.jpg ./data/vs/vertexFS.graphvs

# https://github.com/XingangPan/SCNN
# https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132

import numpy as np
import shutil
import math
import sys
import cv2
import os

libdir = os.path.dirname('./src/')
sys.path.append(os.path.split(libdir)[0])
from src import lib_lineStart_Tracker as module
# from src import lib_connect_lines_hough as hough
from src import canny

# global
iterator = 0    # for images

# exit recursivery
recur = []
recur_iters = 0

# dis
dist = {}   # for write coordinates of start in file
n = 0   # count of opening file (n must be = 2)


def find_start(img, vs):
    # filters
    # gaus = cv2.GaussianBlur(img, (3, 3), 0)
    # Canny
    edges = cv2.Canny(img, 100, 255)    # canny.auto_canny(img)

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
    # print('y: ', indices[0], '\nx: ', indices[1])
    # print(indices)

    # make coordinates in simple form
    # for tracker
    trackcoord = []
    #
    detcoord = []
    img_c = img.copy()
    imgcc = img.copy()
    height, width = imgcc.shape
    img_hough = np.zeros((height, width, 1), np.uint8)

    for i in range(0, len(indices[0])):
        if (module.inboxvertex(indices[1][i], indices[0][i], vs) == True):
            if ([indices[0][i], indices[1][i]] not in detcoord):
                detcoord.append([indices[0][i], indices[1][i]]) # y x
                # draw circles
                cv2.circle(img_c, (indices[1][i], indices[0][i]), 3, (255, 255, 255), 1)
        elif (trackerInBox(indices[1][i], indices[0][i], vs, []) == False):
            if ([indices[0][i], indices[1][i]] not in trackcoord):
                trackcoord.append([indices[0][i], indices[1][i]]) # y x
                cv2.circle(imgcc, (indices[1][i], indices[0][i]), 1, (255, 0, 255), 1)
                cv2.circle(img_hough, (indices[1][i], indices[0][i]), 1, (255, 255, 255), -1)

    cv2.imwrite(folder + 'tracker_road.jpg', imgcc)
    cv2.imwrite(folder + 'tracker_road_black.jpg', img_hough)
    cv2.imwrite(folder + 'line_start_before_filtering.jpg', img_c)

    # Filtering
    preds = module.filterDots_1(detcoord)
    ds = module.filterDots_2(preds)

    # Mapping vertex and starts of lines Vertex: [(), (), ... ]
    vertex_tracker = {}

    # Canny save
    # it = 0
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

        d = module.updateDvertex(d)
        cv2.rectangle(edges, (x - d, y - d), (x + w + d, y + h + d), (255, 255, 255), 1)

        # Tracker
        img_steps = img.copy()
        start_lines = []
        for stl in ds:
            xd = stl[1]
            yd = stl[0]
            if (module.inPolygonContur(xd, yd, x, y, w, h, d, 5) == True):
                start_lines.append(tuple(stl))

        # cv2.rectangle(img_steps, (x - d, y - d), (x + w + d, y + h + d), (0, 165, 255), 2)
        #
        # for i in start_lines:
        #     cv2.circle(img_steps, (i[1], i[0]), 10, (255, 0, 255), 3)
        #
        # cv2.imwrite(folder + str(it) + 'img_steps.jpg', img_steps)
        #
        # it += 1

        vertex_tracker.update({tuple(vertex): tuple(start_lines)})


    cv2.imwrite(folder + 'canny.jpg', edges)

    datcoord = folder + 'Startlines_y_x' + '.txt'
    dat = open(datcoord, 'w')

    # dat.write('y x\n')
    for i in ds:
        dat.write(str(i[0]) + ' ' + str(i[1]) + '\n')
        cv2.circle(img, (i[1], i[0]), 10, (255, 255, 255), 1)

    dat.close()

    detcoord = ds

    cv2.imwrite(folder + 'line_start.jpg', img)

    return [trackcoord, vertex_tracker, detcoord]


def getVertexIgnore(centre, newvs, vertex_tracker):
    for vertex in newvs:
        start = vertex_tracker.get(tuple(vertex))
        print(start)
        if centre in start:
            print(centre)
            return vertex


def refactorTrackcoord(img, vs, vertex_ignore):
    # Canny
    edges = cv2.Canny(img, 100, 255)  # canny.auto_canny(img)

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
    # make coordinates in simple form
    # for tracker
    trackcoord = []
    #
    detcoord = []
    img_c = img.copy()
    imgcc = img.copy()
    height, width = imgcc.shape
    img_hough = np.zeros((height, width, 1), np.uint8)

    for i in range(0, len(indices[0])):
        if (trackerInBox(indices[1][i], indices[0][i], vs, vertex_ignore) == False):
            if ([indices[0][i], indices[1][i]] not in trackcoord):
                trackcoord.append([indices[0][i], indices[1][i]])  # y x
                cv2.circle(imgcc, (indices[1][i], indices[0][i]), 1, (255, 0, 255), 1)
                cv2.circle(img_hough, (indices[1][i], indices[0][i]), 1, (255, 255, 255), -1)

    cv2.imwrite(folder + 'tracker_road.jpg', imgcc)
    cv2.imwrite(folder + 'tracker_road_black.jpg', img_hough)

    return trackcoord


def trackerInBox(xd, yd, vs, vertex_ignore):
    answ = []
    for vertex in vs:
        x = int(vertex[0])
        y = int(vertex[1])
        w = int(vertex[2])
        h = int(vertex[3])
        x_c = int(vertex[4])
        y_c = int(vertex[5])
        r = int(vertex[6])
        d = int(vertex[7])

        d = module.updateDvertex(d)

        if vertex_ignore == vertex:
            # (x - r // 4, y - r // 4), (x + w + r // 4, y + h + r // 4)
            if (((x - r // 4) <= xd <= (x + w + r // 4)) & ((y - r // 4) <= yd <= (y + h + r // 4))):
                answ.append(True)
            else:
                answ.append(False)
        else:
            # cv2.rectangle(edges, (x - d, y - d), (x + w + d, y + h + d), (255, 255, 255), 1)
            if ((x - d <= xd <= x + w + d) & (y - d <= yd <= y + h + d)):
                answ.append(True)
            else:
                answ.append(False)

    # print(answ)
    # print((answ.count(True), answ.count(False)))
    if answ.count(False) == len(answ):
        return False
    else:
        return True


def inCircle(xd, yd, x_c, y_c, r):
    d = module.dis(xd, x_c, yd, y_c, 0, 0) - r
    if ((d < 0)): # & (inCircleCont(xd, x_c, yd, y_c, r, 0) == True)):
        return True
    return False


def inCircleCont(xd, yd, x_c, y_c, r, pix = 0):
    if (math.ceil(abs(module.dis(xd, x_c, yd, y_c, 0, 0) - r)) <= pix):
        return True
    return False


def trackerFilterDots_2(trackcoord_prefiltering):
    if len(trackcoord_prefiltering) == 1:
        return trackcoord_prefiltering

    dotcoord = trackcoord_prefiltering
    if len(dotcoord) == 1:
        return dotcoord

    filt_dotcoord = []

    for i in range(0, len(dotcoord)):
        xs = int(dotcoord[i][1])
        ys = int(dotcoord[i][0])

        for j in range(i + 1, len(dotcoord)):
            if ((abs(xs - dotcoord[i][1]) < 9) & (abs(ys - dotcoord[i][0]) < 9)):
                if (((dotcoord[j][0], dotcoord[j][1]) not in filt_dotcoord)):
                    filt_dotcoord.append((dotcoord[j][0], dotcoord[j][1]))
                    if ((ys, xs) not in filt_dotcoord):
                        filt_dotcoord.append((ys, xs))

    # print(filt_dotcoord)

    x = []
    y = []
    for i in filt_dotcoord:
        x.append(i[1])
        y.append(i[0])

    check_ = (math.ceil(sum(y) / len(y)), math.ceil(sum(x) / len(x)))

    filt_dotcoord.clear()
    filt_dotcoord.append(check_)
    return filt_dotcoord


def intersectionCircles(x_c1, y_c1, r1, x_c2, y_c2, r2):
    if (math.ceil(module.dis(x_c1,x_c2,y_c1,y_c2,0,0)) - (r1 + r2) <= 1):
        di = math.ceil(module.dis(x_c1,x_c2,y_c1,y_c2,0,0)) - (r1 + r2)
        print("DIS: ", di)
        dist[di] = [x_c1, y_c1, x_c2, y_c2]
        return True
    return False


def trackerStepALL(img_c, step, trackcoord, R):
    steps = []
    while len(trackcoord) > 0:
        for centre in step:
            cv2.circle(img_c, (centre[1], centre[0]), R, (0, 0, 255), 1)

            trackcoord_prefiltering = []

            print(len(trackcoord))

            for coord in trackcoord:
                if inCircleCont(coord[1], coord[0], centre[1], centre[0], R, 1) == True:
                    if (coord not in trackcoord_prefiltering):
                        print(len(trackcoord))
                        trackcoord_prefiltering.append([coord[1], coord[0]])

            trackcoord_filtering = []
            delete = []
            #
            if len(trackcoord_prefiltering) > 0:
                # trackcoord_filt = trackerFilterDots_1(trackcoord_prefiltering, trackcoord)
                trackcoord_filtering = trackerFilterDots_2(trackcoord_prefiltering)
                # trackcoord_filtering = trackcoord_prefiltering

                steps.append(tuple((trackcoord_filtering[0][1], trackcoord_filtering[0][0])))

                print("INFO: step loading...")

                # for dl in range

                delete = [trackcoord_filtering[0][1], trackcoord_filtering[0][0]]

            for coord in trackcoord:
                if ((inCircle(coord[1], coord[0], centre[1], centre[0], R) == True) &
                        (inCircleCont(coord[1], coord[0], centre[1], centre[0], R, 1) == False)):
                    if delete != coord:
                        trackcoord.remove(coord)

            for coord in trackcoord_filtering:
                cv2.circle(img_c, (coord[0], coord[1]), R, (0, 0, 255), 1)

        print("INFO: next step!")
        print("INFO: len(steps): ", len(steps))

        global iterator
        cv2.imwrite(folder + str(iterator) + '_tracker.jpg', img_c)
        iterator += 1

        recur.append(len(trackcoord))

        # exit
        if ((len(steps) != 0) & (recur.count(recur[-1]) == 1)):
            trackerStepALL(img_c, steps, trackcoord, R)
        elif ((len(steps) == 0) | (recur.count(recur[-1]) > 1)):
            trackcoord.clear()

    print("INFO: END of steps")
    cv2.imwrite(folder + 'tracker.jpg', img_c)


def intersectionCirclesST(centre, starts_tracker, R):
    for cen in starts_tracker:
        if intersectionCircles(centre[0], centre[1], R, cen[0], cen[1], R) == True:
            d = min(dist.keys())
            res_d = dist[d]
            global n
            n += 1
            print(centre)
            if n <= 2:
                print("ACTION!")
                action = open('./../result/action.txt', 'a')
                action.write(str(res_d[0]) + ' ' + str(res_d[1]) + ' ' + str(res_d[2]) + ' ' + str(res_d[3]) + '\n')
                action.close()
            return True
    return False


def loaddatTracker(data_file):
    actions = []
    vs = open(data_file, 'r')

    for line in vs:
        cleanedLine = line.strip()
        x = cleanedLine.split(" ")
        actions.append(x)

    return actions


def confirmVertex(img_c, starts_tracker_auth, trackcoord, R):
    actions = loaddatTracker('./../result/action.txt')
    action = actions[0]
    print(action)
    centre = (int(action[-2]), int(action[-1]))
    print(centre, type(centre))

    if centre in starts_tracker_auth:
        print("Delete")
        starts_tracker_auth.remove(centre)

    global n
    n = 1
    global dist
    dist = {}
    global recur
    recur = []

    action = open('./../result/action.txt', 'a')
    action.write(str(centre[0]) + ' ' + str(centre[1]) + ' ')
    action.close()

    print(centre in starts_tracker_auth)
    print(len(trackcoord))

    trackerStepSingle(img_c, centre, starts_tracker_auth, trackcoord, R)


def trackerStepSingle(img_c, onetracker, starts_tracker, trackcoord, R):
    steps = []
    centre = onetracker
    print("CENTER: ", centre)
    if ((intersectionCirclesST(centre, starts_tracker, R) == False) | (len(trackcoord) > 0)):
        cv2.circle(img_c, (centre[1], centre[0]), R, (0, 0, 255), 1)

        trackcoord_prefiltering = []

        print(len(trackcoord))

        for coord in trackcoord:
            if inCircleCont(coord[1], coord[0], centre[1], centre[0], R, 1) == True:
                if (coord not in trackcoord_prefiltering):
                    print(len(trackcoord))
                    trackcoord_prefiltering.append([coord[1], coord[0]])

        trackcoord_filtering = []
        delete = []
        #

        if len(trackcoord_prefiltering) > 0:
            trackcoord_filtering = trackerFilterDots_2(trackcoord_prefiltering)
            print("len(trackcoord_filtering) ", len(trackcoord_filtering), trackcoord_filtering)
            steps = (trackcoord_filtering[0][1], trackcoord_filtering[0][0])

            delete = [trackcoord_filtering[0][1], trackcoord_filtering[0][0]]

        print("INFO: step loading...")

        for coord in trackcoord:
            if ((inCircle(coord[1], coord[0], centre[1], centre[0], R) == True) &
                    (inCircleCont(coord[1], coord[0], centre[1], centre[0], R, 1) == False)):
                if delete != coord:
                    trackcoord.remove(coord)

        for coord in trackcoord_filtering:
            cv2.circle(img_c, (coord[0], coord[1]), R, (0, 0, 255), 1)

        print("INFO: next step!")
        print("INFO: len(steps): ", len(steps), type(steps), steps)

        global iterator
        cv2.imwrite(folder + str(iterator) + '_tracker.jpg', img_c)
        iterator += 1

        recur.append(len(trackcoord))

        # exit
        if ((len(steps) != 0) & (recur.count(recur[-1]) == 1)):
            trackerStepSingle(img_c, steps, starts_tracker, trackcoord, R)
        elif ((len(steps) == 0) | (recur.count(recur[-1]) > 1)):
            trackcoord.clear()

    print("INFO: END of steps")
    cv2.imwrite(folder + str(iterator) + '_END_tracker.jpg', img_c)
    return


def getRTrack(newvs):
    rad = []
    for vertex in newvs:
        d = int(vertex[7]) // 4
        rad.append(module.updateDvertex(d))

    return math.ceil(sum(rad) / len(rad))


def getStartTracker(vertex_tracker, newvs):
    start_tracker = []
    for vertex in newvs:
        start = vertex_tracker.get(tuple(vertex))

        for i in start:
            start_tracker.append(i)


    return start_tracker


def findIterTrackVertex(newvs, vertex_tracker, step):
    actions = loaddatTracker('./../result/action.txt')
    action = actions[step - 1]
    centre = (int(action[-2]), int(action[-1]))

    i = 0
    for vertex in newvs:
        start = vertex_tracker.get(tuple(vertex))
        if (centre in start):
            return i
        i += 1
    return -1


def getCoefFunct(centre1, centre2):
    # print(centre1, centre2)
    # print(centre1[0] - centre2[0], centre1[1] - centre2[1])
    #
    # k_find = 0
    # if ((centre1[0] > centre2[0]) & (centre1[1] > centre2[1])):  # 1
    #     k_find = abs(round(((centre1[0] - centre2[0]) / (centre1[1] - centre2[1])), 2))
    # elif ((centre1[0] > centre2[0]) & (centre1[1] < centre2[1])):  # 2
    #     k_find = abs(round(((centre1[0] - centre2[0]) / (centre1[1] - centre2[1])), 2))
    # elif ((centre1[0] < centre2[0]) & (centre1[1] > centre2[1])):  # 3
    #     k_find = -1*abs(round(((centre1[0] - centre2[0]) / (centre1[1] - centre2[1])), 2))
    # elif ((centre1[0] < centre2[0]) & (centre1[1] < centre2[1])):  # 4
    #     k_find = -1*abs(round(((centre1[0] - centre2[0]) / (centre1[1] - centre2[1])), 2))
    #
    # return k_find
    if (centre1[1] - centre2[1] != 0):
        return (round(((centre1[0] - centre2[0]) / (centre1[1] - centre2[1])), 2))


def getKRect(img_c, centre, centreRect, vertex):
    x = int(vertex[0])
    y = int(vertex[1])
    w = int(vertex[2])
    h = int(vertex[3])
    x_c = int(vertex[4])
    y_c = int(vertex[5])
    r = int(vertex[6])
    d = int(vertex[7])

    if (((centre[1] < centreRect[0]) & (centre[0] < centreRect[1]) &
         (centre[1] < (x - r // 4)) & (centre[0] < (y - r // 4))) |
        ((centre[1] > centreRect[0]) & (centre[0] > centreRect[1]) &
         (centre[1] > (x + w + r // 4)) & (centre[0] > (y + h + r // 4)))):
        cv2.line(img_c, (centre[1], centre[0]), (x - r // 4, y + h + r // 4), (0, 255, 0), 2)
        cv2.line(img_c, (centre[1], centre[0]), (x + w + r // 4, y - r // 4), (0, 255, 0), 2)
    elif (((centre[1] > centreRect[0]) & (centre[0] < centreRect[1]) &
           (centre[1] > (x + w + r // 4)) & (centre[0] < (y - r // 4))) |
          ((centre[1] < centreRect[0]) & (centre[0] > centreRect[1]) &
           (centre[1] < (x - r // 4)) & (centre[0] > (y + h + r // 4)))):
        cv2.line(img_c, (centre[1], centre[0]), (x - r // 4, y - r // 4), (0, 255, 0), 2)
        cv2.line(img_c, (centre[1], centre[0]), (x + w + r // 4, y + h + r // 4), (0, 255, 0), 2)
    elif (((x - r // 4) < centre[1] < (x + w + r // 4)) & (centre[0] < centreRect[1])):
        cv2.line(img_c, (centre[1], centre[0]), (x - r // 4, y - r // 4), (0, 255, 0), 2)
        cv2.line(img_c, (centre[1], centre[0]), (x + w + r // 4, y - r // 4), (0, 255, 0), 2)
    elif (((x - r // 4) <= centre[1] <= (x + w + r // 4)) & (centre[0] > centreRect[1])):
        cv2.line(img_c, (centre[1], centre[0]), (x - r // 4, y + h + r // 4), (0, 255, 0), 2)
        cv2.line(img_c, (centre[1], centre[0]), (x + w + r // 4, y + h + r // 4), (0, 255, 0), 2)
    elif ((centre[1] < centreRect[0]) & ((y - r // 4) <= centre[0] <= (y + h + r // 4))):
        cv2.line(img_c, (centre[1], centre[0]), (x - r // 4, y - r // 4), (0, 255, 0), 2)
        cv2.line(img_c, (centre[1], centre[0]), (x - r // 4, y + h + r // 4), (0, 255, 0), 2)
    elif ((centre[1] > centreRect[0]) & ((y - r // 4) <= centre[0] <= (y + h + r // 4))):
        cv2.line(img_c, (centre[1], centre[0]), (x + w + r // 4, y - r // 4), (0, 255, 0), 2)
        cv2.line(img_c, (centre[1], centre[0]), (x + w + r // 4, y + h + r // 4), (0, 255, 0), 2)


def trackerShoot(img_c, newvs, vertex_tracker, R, step):
    numV = findIterTrackVertex(newvs, vertex_tracker, step)
    if numV != -1:
        vertex = newvs[numV]
        x = int(vertex[0])
        y = int(vertex[1])
        w = int(vertex[2])
        h = int(vertex[3])
        x_c = int(vertex[4])
        y_c = int(vertex[5])
        r = int(vertex[6])
        d = int(vertex[7])

        cv2.rectangle(img_c, (x - r // 4, y - r // 4), (x + w + r // 4, y + h + r // 4),
                      (0, 0, 255), 1)
        k = math.ceil(0.1 * (r // 4))
        cv2.rectangle(img_c, (x - k - d // 4, y - k - d // 4),
                      (x + w + k + d // 4, y + h + k + d // 4), (0, 215, 255), 2)
        d = module.updateDvertex(d)
        cv2.rectangle(img_c, (x - d, y - d), (x + w + d, y + h + d), (0, 165, 255), 2)

        actions = loaddatTracker('./../result/action.txt')

        action = actions[step - 1]

        # centreRect
        # cv2.circle(img_c, (x_c, y_c), 10, (255, 255, 255), 1)

        centre = (int(action[0]), int(action[1]))
        cv2.circle(img_c, (centre[1], centre[0]), R, (255, 255, 255), 1)

        centre1 = (int(action[-2]), int(action[-1]))
        cv2.circle(img_c, (centre1[1], centre1[0]), R, (255, 255, 255), 1)

        centre2 = (int(action[2]), int(action[3]))
        cv2.circle(img_c, (centre2[1], centre2[0]), R, (0, 0, 255), 1)

        cv2.line(img_c, (centre1[1], centre1[0]), (centre2[1], centre2[0]), (0, 255, 0), 2)

        k_find = getCoefFunct(centre1, centre2)
        b_find = centre1[0] - k_find * centre1[1]
        cv2.line(img_c, (centre1[1], centre1[0]), (x_c, int(k_find * x_c + b_find)), (0, 255, 0), 2)

        getKRect(img_c, centre1, (x_c, y_c), vertex)

        global iterator
        iterator += 1

        cv2.imwrite(folder + str(iterator) + '_SHOOT.jpg', img_c)

        if (module.inPolygon(x_c, int(k_find * x_c + b_find), x, y, w, h, r)) == True:
            return (True, centre1)
        else:
            return (False, centre1)
    else:
        return (False, tuple())


def getTrackcoord(trackcoord_):
    trackcoord__ = []
    for tr_ in trackcoord_:
        trackcoord__.append(tr_)

    return trackcoord__


def tracker(cropgray, img, trackcoord, vertex_tracker, newvs):
    trackcoord_ = []
    for tr in trackcoord:
        trackcoord_.append(tr)

    img_c = img.copy()
    # for i in trackcoord_:
    #     cv2.circle(img_c, (i[1], i[0]), 1, (255, 255, 255), 1)

    R = getRTrack(newvs)

    for vertex in newvs:
        x = int(vertex[0])
        y = int(vertex[1])
        w = int(vertex[2])
        h = int(vertex[3])
        x_c = int(vertex[4])
        y_c = int(vertex[5])
        r = int(vertex[6])
        d = int(vertex[7])

        d = module.updateDvertex(d)

        starts_tracker = getStartTracker(vertex_tracker, newvs)

        start = vertex_tracker.get(tuple(vertex))

        starts_tracker_auth = []
        for i in starts_tracker:
            starts_tracker_auth.append(i)

        for i in start:
            starts_tracker.remove(i)

        # start = list(start)

        for onetracker in start:
            img_c1 = img.copy()
            img_c2 = img.copy()
            img_s1 = img.copy()
            img_s2 = img.copy()

            trackcoord_1 = getTrackcoord(trackcoord_)
            trackcoord_2 = getTrackcoord(trackcoord_)

            # action
            global recur_iters
            global n
            print('n: ', n)
            n += 1
            action = open('./../result/action.txt', 'w')
            action.write(str(onetracker[0]) + ' ' + str(onetracker[1]) + ' ')
            action.close()

            trackerStepSingle(img_c1, onetracker, starts_tracker, trackcoord_1, R)
            (answ1, centre1) = trackerShoot(img_s1, newvs, vertex_tracker, R, 1)
            print('(answ1, centre1): ', (answ1, centre1))

            while (answ1 == False):
                # remove centre
                if (centre1 in starts_tracker):
                    starts_tracker.remove(centre1)
                if (centre1 in starts_tracker_auth):
                    starts_tracker_auth.remove(centre1)
                # refactoring road for tracker
                cropgray_c = cropgray.copy()
                cropgray_cc = cropgray.copy()
                # n
                n = 1
                vertex_ignore = getVertexIgnore(centre1, newvs, vertex_tracker)
                trackcoord_1 = refactorTrackcoord(cropgray_c, newvs, vertex_ignore)
                trackcoord_2 = refactorTrackcoord(cropgray_cc, newvs, vertex_ignore)
                #
                img_cc = img.copy()
                img_ss = img.copy()
                #
                # trackcoord_1_ = getTrackcoord(trackcoord_)
                #
                trackerStepSingle(img_cc, onetracker, starts_tracker, trackcoord_1, R)
                (answ1, centre1) = trackerShoot(img_ss, newvs, vertex_tracker, R, 1)
                print('(answ1, centre1): ', (answ1, centre1))
                #
                recur_iters += 1
                if recur_iters > 3:
                    break
                print(recur_iters)

            confirmVertex(img_c2, starts_tracker_auth, trackcoord_2, R)
            (answ2, centre2) = trackerShoot(img_s2, newvs, vertex_tracker, R, 2)
            print('(answ2, centre2): ', (answ2, centre2))

            recur_iters = 0
            print(recur_iters)
            # print(starts_tracker)
            # print(starts_tracker_auth)
            # exit()

            while (answ2 == False):
                # remove centre
                if (centre2 in starts_tracker):
                    starts_tracker.remove(centre2)
                if (centre2 in starts_tracker_auth):
                    starts_tracker_auth.remove(centre2)
                # refactoring road for tracker
                cropgray_c = cropgray.copy()
                cropgray_cc = cropgray.copy()
                vertex_ignore = getVertexIgnore(centre2, newvs, vertex_tracker)
                trackcoord_1 = refactorTrackcoord(cropgray_c, newvs, vertex_ignore)
                trackcoord_2 = refactorTrackcoord(cropgray_cc, newvs, vertex_ignore)
                #
                img_cc = img.copy()
                img_ss = img.copy()
                #
                # trackcoord_2_ = getTrackcoord(trackcoord_)
                #
                confirmVertex(img_cc, starts_tracker_auth, trackcoord_2, R)
                (answ2, centre2) = trackerShoot(img_ss, newvs, vertex_tracker, R, 2)
                print('(answ2, centre2): ', (answ2, centre2))
                #
                recur_iters += 1
                if recur_iters > 3:
                    break
                print(recur_iters)

            n = 0
            print(recur_iters)
            print(n)


def findStart_Tracker(image):
    img = cv2.imread(image)

    img_c = img.copy()

    gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 2)

    edges = cv2.Canny(gray, 100, 255)

    vs = module.loaddat(path)

    newvs = module.getDistanceVertex(vs)

    # img_rect = img.copy()
    cropgray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    # crop_gray = cv2.GaussianBlur(cropgray, (kernel_size, kernel_size), -1)

    # trackcoord, vertex_tracker, detcoord
    (trackcoord, vertex_tracker, start_line) = find_start(cropgray, newvs)

    # TRACKER
    tracker(gray, img_c, trackcoord, vertex_tracker, newvs)

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

        d = module.updateDvertex(d)
        cv2.rectangle(img_c, (x - d, y - d), (x + w + d, y + h + d), (0, 165, 255), 2)

        iter += 1

    for i in start_line:
        cv2.circle(img_c, (i[1], i[0]), 10, (255, 0, 255), 3)

    cv2.imwrite(folder + "result.jpg", img_c)


if __name__ == '__main__':
    # create folder
    folder = './../result/start_lines_1/'
    module.createFolder(folder)

    # errors and debug logs
    # sys.stdout = open('./result/' + 'output.log', 'w')

    # image
    image = sys.argv[1]

    # path to folder
    path = sys.argv[2] #"./vs2/vertex_search.graphvs"

    findStart_Tracker(image)

    sys.exit(0)