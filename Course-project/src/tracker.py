# python ./src/tracker.py ./result/vs/debug_Haar_NN/preprocessing.jpg ./result/vs/vertex_lines.graphvs

import numpy as np
import shutil
import math
import sys
import cv2
import os

libdir = os.path.dirname('./src/')
sys.path.append(os.path.split(libdir)[0])
from src.libs import lib_lineStart_Tracker as module
# from src import lib_connect_lines_hough as hough
from src.libs import canny


'''FIND START'''
def find_start(img, vs):
    # filters
    # gaus = cv2.GaussianBlur(img, (3, 3), 0)

    # PART 1
    # Canny
    edges = cv2.Canny(img, 100, 255)    # canny.auto_canny(img)

    kernel = np.ones((5, 5), np.uint8)
    edges_k = cv2.dilate(edges, kernel, iterations=3)

    cv2.imwrite(folder_debug + 'Canny_k.jpg', edges_k)
    img_k = cv2.imread(folder_debug + 'Canny_k.jpg')

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

    # PART 2
    # Find start coordinate and road for Tracker
    coorRoadTracker = []
    coordStartTracker = []
    # images to show
    img_c = img.copy()
    imgcc = img.copy()
    height, width = imgcc.shape
    img_hough = np.zeros((height, width, 1), np.uint8)

    for i in range(0, len(indices[0])):
        '''Find start place for tracker'''
        if (module.inBoxContVertex(indices[1][i], indices[0][i], vs) == True):
            if ([indices[0][i], indices[1][i]] not in coordStartTracker):
                '''coordStartTracker - list of coordinates line contour of rectangle'''
                coordStartTracker.append([indices[0][i], indices[1][i]]) # y x
                # draw circles
                cv2.circle(img_c, (indices[1][i], indices[0][i]), 3, (255, 255, 255), 1)
            '''Another coordinates is a trcaker road, add to coorRoadTracker list'''
        elif (module.trackerInBox(indices[1][i], indices[0][i], vs) == False):
            if ([indices[0][i], indices[1][i]] not in coorRoadTracker):
                coorRoadTracker.append([indices[0][i], indices[1][i]])  # y x
                cv2.circle(imgcc, (indices[1][i], indices[0][i]), 1, (255, 0, 255), 1)
                cv2.circle(img_hough, (indices[1][i], indices[0][i]), 1, (255, 255, 255), -1)

    # Images with results
    cv2.imwrite(module.folder + 'tracker_road.jpg', imgcc)
    cv2.imwrite(module.folder + 'tracker_road_black.jpg', img_hough)
    cv2.imwrite(folder_debug + 'line_start_before_filtering.jpg', img_c)

    # Filtering
    predCST = module.filterDots_1(coordStartTracker)
    newCoordStartTracker = module.filterDots_2(predCST)

    # PART 3
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

        # Tracker start lines
        img_steps = img.copy()
        start_lines = []
        for stl in newCoordStartTracker:
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

    # save canny image with edges and start coordinates for Tracker
    cv2.imwrite(folder_debug + 'canny.jpg', edges)

    # PART 4
    # Save coordinates in file
    datcoord = module.folder + 'Startlines_y_x.txt'
    dat = open(datcoord, 'w')
    # dat.write('y x\n')
    for i in newCoordStartTracker:
        dat.write(str(i[0]) + ' ' + str(i[1]) + '\n')
        cv2.circle(img, (i[1], i[0]), 10, (255, 255, 255), 1)

    dat.close()

    # PART 5
    # Save image with line starts and return list
    coordStartTracker = newCoordStartTracker
    cv2.imwrite(module.folder + 'line_start.jpg', img)
    return [coorRoadTracker, vertex_tracker, coordStartTracker]


'''Control function for Tracker start running and Tracker shooting'''
def runTracker(img, coorRoadTracker, vertex_tracker, newvs):
    '''
    Control function for Tracker start running and Tracker shooting

    :param img: image for copy and drawing results
    :param coorRoadTracker: coordinates of road Tracker
    :param vertex_tracker: for start running Tracker
    :param newvs: dictionary with information about vertexes
    :return: None
    '''

    module.coordinatesRoadTracker = coorRoadTracker

    start_tracker = []

    R = module.getRTrack(newvs)

    for vertex in newvs:
        start = vertex_tracker.get(tuple(vertex))
        for i in start:
            start_tracker.append(i)

    for vertex in newvs:
        module.graphPreReady = [vertex, ":"]

        start = vertex_tracker.get(tuple(vertex))

        for oneTracker in start:
            imgOneTracker = img.copy()
            imgShootTracker = img.copy()

            module.trackerStepSingle(imgOneTracker, oneTracker, start_tracker, R)

            (answ1, centre1) = module.trackerShoot(imgShootTracker, newvs, vertex_tracker, R)

            print("     " * module.rec_iter, "ACTION!")
            action = open('./result/start_lines/action.txt', 'w')
            action.write("-1")
            action.close()

            print("     " * module.rec_iter, '(answ1, centre1): ', (answ1, centre1))
            if (centre1 in start_tracker):
                start_tracker.remove(centre1)

            module.n = 0

            module.deleteCoordinates = []

            print("     " * module.rec_iter, "INFORMATION  graphPreReady: ", module.graphPreReady)

        print("     " * module.rec_iter, 'INFO: graphReady = ', module.graphReady)


'''The main function of steps detection lines!'''
def findStartTracker(image):
    '''
    The main function of steps detection lines!

    :param image: image for copy and drawing
    :return: None
    '''
    img = cv2.imread(image)
    img_c = img.copy()
    cropgray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

    '''STEP 1: vertex search (vs)
    Load data from file ./result/vs/vertex_lines.graphvs'''
    vs = module.loaddat(path)

    '''STEP 2: Update vertex search, get distance'''
    newvs = module.getDistanceVertex(vs)

    '''STEP 3: Find start'''
    (coorRoadTracker, vertex_tracker, start_line) = find_start(cropgray, newvs)

    '''STEP 4: Tracker'''
    runTracker(img_c, coorRoadTracker, vertex_tracker, newvs)

    '''STEP 5: Draw results'''
    height, width, channels = img_c.shape
    tracker_predict = np.zeros((height, width, channels), np.uint8)

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
        k = math.ceil(0.1 * (r // 4))
        cv2.rectangle(img_c, (x - k - d // 4, y - k - d // 4),
                      (x + w + k + d // 4, y + h + k + d // 4), (0, 215, 255), 2)

        d = module.updateDvertex(d)
        cv2.rectangle(img_c, (x - d, y - d), (x + w + d, y + h + d), (0, 165, 255), 2)

        # Tracker_predict
        cv2.circle(tracker_predict, (x_c, y_c), r, (0, 0, 255), 2)

    for i in start_line:
        cv2.circle(img_c, (i[1], i[0]), 10, (255, 0, 255), 3)

    for line_draw in module.graphReady:
        vertexF = line_draw[0]
        x_c_F = int(vertexF[4])
        y_c_F = int(vertexF[5])

        vertexL = line_draw[1]
        x_c_L = int(vertexL[4])
        y_c_L = int(vertexL[5])
        cv2.line(tracker_predict, (x_c_F, y_c_F), (x_c_L, y_c_L),
                 (0, 0, 255), 2)

    cv2.imwrite(module.folder + "Start_lines.jpg", img_c)
    cv2.imwrite(module.folder + "image_predict.jpg", tracker_predict)
    return


if __name__ == '__main__':
    # create folder
    module.folder = './result/start_lines/'
    module.createFolder(module.folder)
    folder_debug = './result/start_lines/debug_tracker/'
    # create dir
    module.createFolder(folder_debug)

    # image
    image = sys.argv[1]

    # path to folder
    path = sys.argv[2] #"./vs2/vertex_search.graphvs"

    findStartTracker(image)

    sys.exit(0)