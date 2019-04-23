# python ./src/tracker.py ./data/vs1/exe1.jpg ./data/vs1/vertexFS.graphvs
# python ./src/tracker.py ./data/vs/exe6.jpg ./data/vs/vertexFS.graphvs

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
iterator = 0

# exit recursivery
recur = []


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
        elif (trackerInBox(indices[1][i], indices[0][i], vs) == False):
            if ([indices[0][i], indices[1][i]] not in trackcoord):
                trackcoord.append([indices[0][i], indices[1][i]])  # y x
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


def trackerInBox(xd, yd, vs):
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


def trackerFilterDots_1(dotcoord, trackcoord):
    if len(dotcoord) == 1:
        return dotcoord
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

            # if len xs == 1
            # if len(xs) == 1:
            #     x = xs[0]
            #     for y in ys:
            #         if (((y, x) != ds[-1]) & ((y, x) in trackcoord)):
            #             trackcoord.remove([x, y])
            # elif len(ys) == 1:
            #     y = ys[0]
            #     for x in xs:
            #         if (((y, x) != ds[-1]) & ((y, x) in trackcoord)):
            #             trackcoord.remove([x, y])

            xs.clear()
            ys.clear()

            xs.append(dotcoord[i][1])
            ys.append(dotcoord[i][0])

    # print("Result: ys ", ys)
    # print("Result: xs ", xs)
    # print("Result: ", (math.ceil(sum(xs) / len(xs)), math.ceil(sum(ys) / len(ys))))
    ds.append((math.ceil(sum(ys) / len(ys)), math.ceil(sum(xs) / len(xs))))

    if len(ds) == 1:
        return ds

    return ds


def trackerFilterDots_2(trackcoord_prefiltering, trackcoord):
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

    check = (math.ceil(sum(x) / len(x)), math.ceil(sum(y) / len(y)))
    check_ = (math.ceil(sum(y) / len(y)), math.ceil(sum(x) / len(x)))


    filt_dotcoord.clear()
    filt_dotcoord.append(check_)
    return filt_dotcoord


#def trackerShoot():


def intersectionCircles(x_c1, y_c1, r1, x_c2, y_c2, r2):
    if (math.ceil(module.dis(x_c1,x_c2,y_c1,y_c2,0,0)) - (r1 + r2) <= 1):
        return True
    return False


'''
centre=
map = {
(centre[1], centre[0]): [(centre[1], centre[0]), ...]

}
'''


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
                trackcoord_filtering = trackerFilterDots_2(trackcoord_prefiltering, trackcoord)
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



'''
centre=
map = {
(centre[1], centre[0]): [(centre[1], centre[0]), ...]

}
'''


def trackerStepSingle(img_c, start_tracker, trackcoord, R):

    lines = {}

    for i in range(0, len(start_tracker)):
        lines.update({start_tracker[i] : []})

    print(lines)




    # lines = {}
    #
    # steps = []
    #
    # centre = step[0]
    # cv2.circle(img_c, (centre[1], centre[0]), R, (0, 0, 255), 1)
    #
    # trackcoord_prefiltering = []
    #
    # print(len(trackcoord))
    #
    # for coord in trackcoord:
    #     if inCircleCont(coord[1], coord[0], centre[1], centre[0], R, 1) == True:
    #         if (coord not in trackcoord_prefiltering):
    #             print(len(trackcoord))
    #             trackcoord_prefiltering.append([coord[1], coord[0]])
    #         else:
    #             exit()
    #             trackcoord.remove(coord)
    #
    # trackcoord_filtering = []
    # #
    # if len(trackcoord_prefiltering) > 0:
    #     # trackcoord_filt = trackerFilterDots_1(trackcoord_prefiltering, trackcoord)
    #     # trackcoord_filtering = trackerFilterDots_2(trackcoord_filt, trackcoord)
    #     trackcoord_filtering = trackcoord_prefiltering
    #
    #     steps.append(tuple((trackcoord_filtering[0][1], trackcoord_filtering[0][0])))
    #
    #     print("INFO: step loading...")
    #
    # for coord in trackcoord:
    #     if ((inCircle(coord[1], coord[0], centre[1], centre[0], R) == True) &
    #             (inCircleCont(coord[1], coord[0], centre[1], centre[0], R, 1) == False)):
    #         trackcoord.remove(coord)
    #
    # for coord in trackcoord_filtering:
    #     cv2.circle(img_c, (coord[0], coord[1]), R, (0, 0, 255), 1)
    #
    # print("INFO: next step!")
    # print("INFO: len(steps): ", len(steps))
    #
    # global iterator
    # cv2.imwrite(folder + str(iterator) + '_tracker.jpg', img_c)
    # iterator += 1
    #
    # recur.append(len(trackcoord))
    #
    # # exit
    # if ((len(steps) != 0) & (recur.count(recur[-1]) == 1)):
    #     trackerStepSingle(img_c, steps, trackcoord, R)
    # elif ((len(steps) == 0) | (recur.count(recur[-1]) > 1)):
    #     trackcoord.clear()
    #
    #
    # print("INFO: END of steps")
    # cv2.imwrite(folder + 'tracker.jpg', img_c)


def getRTrack(newvs):
    rad = []
    for vertex in newvs:
        d = int(vertex[7]) // 4
        rad.append(module.updateDvertex(d))

    return math.ceil(sum(rad) / len(rad))


def tracker(img, trackcoord, vertex_tracker, newvs):
    trackcoord_ = trackcoord
    img_c = img.copy()
    # for i in trackcoord_:
    #     cv2.circle(img_c, (i[1], i[0]), 1, (255, 255, 255), 1)

    start_tracker = []


    #R = 20
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

        # R = math.ceil(d / 4 + 0.2*(d / 3))

        start = vertex_tracker.get(tuple(vertex))


        for i in start:
            start_tracker.append(i)

    trackerStepALL(img_c, start_tracker, trackcoord_, R)

        # for centre in start:
        #     cv2.circle(img_c, (centre[1], centre[0]), R, (255, 0, 255), 1)
        #
        #     trackcoord_prefiltering = []
        #
        #     print(len(trackcoord))
        #
        #     for coord in trackcoord:
        #         if inCircleCont(coord[1], coord[0], centre[1], centre[0], R, 1) == True:
        #             if (coord not in trackcoord_prefiltering):
        #                 print(len(trackcoord))
        #                 # print(type(coord))
        #                 # print(coord[0])
        #                 trackcoord_prefiltering.append([coord[1], coord[0]])
        #             else:
        #                 trackcoord.remove(coord)
        #         elif inCircle(coord[1], coord[0], centre[1], centre[0], R) == True:
        #             trackcoord.remove(coord)
        #
        #     trackcoord_filtering = []
        #     #
        #     if len(trackcoord_prefiltering) > 0:
        #         trackcoord_filt = trackerFilterDots_1(trackcoord_prefiltering)
        #         trackcoord_filtering = trackerFilterDots_2(trackcoord_filt)
        #         print('AFTER: len(trackcoord_filtering) = ', len(trackcoord_filtering),
        #               trackcoord_filtering)
        #
        #     for coord in trackcoord_filtering:
        #         cv2.circle(img_c, (coord[0], coord[1]), R, (0, 255, 0), 1)

    # cv2.imwrite(folder + 'tracker.jpg', img_c)


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
    tracker(img_c, trackcoord, vertex_tracker, newvs)

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
    folder = './result/start_lines/'
    module.createFolder(folder)

    # # errors and debug logs
    # sys.stdout = open('./result/' + 'output.log', 'w')

    # image
    image = sys.argv[1]

    # path to folder
    path = sys.argv[2] #"./vs2/vertex_search.graphvs"

    findStart_Tracker(image)

    sys.exit(0)