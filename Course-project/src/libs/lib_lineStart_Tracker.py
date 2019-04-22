import numpy as np
import shutil
import math
import sys
import cv2
import os


'''---------------------------------------------------------------------------------------------
                                      FIND START FOR TRACKER
   ---------------------------------------------------------------------------------------------
'''
'''---------------------------------------------------------------------------------------------
                                        Standart functions
'''
'''Standart functions'''
def createFolder(directory):
    '''
    Create directory, if find with sample name -> remove -> create

    :param directory: path to directory with name
    :return: None
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


'''Standart functions'''
def loaddat(data_file):
    '''
    Load information about rectangle from file vertex_lines.graphvs

    :param data_file: path to file
    :return: list of
    '''
    vertex_search = []
    vs = open(data_file, 'r')

    for line in vs:
        cleanedLine = line.strip()
        x = cleanedLine.split(" ")
        vertex_search.append(x)

    return vertex_search


'''---------------------------------------------------------------------------------------------
                                        Math functions
'''
'''Update distance between two rectangles'''
def updateDvertex(d):
    '''
    Update distance

    :param d: distance
    :return: Updated distance
    '''
    return math.ceil(d // 2 - 0.4 * (d // 4))


'''Find distance between two dots (center of rectangles)'''
def dis(x1, x2, y1, y2, r1, r2):
    '''
    Distance on the plane between 2 dots

    :param x1: x coordinate of 1 rectangle
    :param x2: x' coordinate of 2 rectangle
    :param y1: y coordinate of 1 rectangle
    :param y2: y' coordinate of 2 rectangle
    :param r1: radius of 1 circle-rectangle
    :param r2: radius of 2 circle-rectangle
    :return: distance between 2 dots (centre of the rectangles)
    '''
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2) - r1 - r2


'''Find distance between two rectangles'''
def d(x1, x2, y1, y2, r1, r2):
    '''
    Find distance between two rectangles

    :param x1: x coordinate of 1 rectangle
    :param x2: x' coordinate of 2 rectangle
    :param y1: y coordinate of 1 rectangle
    :param y2: y' coordinate of 2 rectangle
    :param r1: radius of 1 circle-rectangle
    :param r2: radius of 2 circle-rectangle
    :return: max of difference rectangles' x coordinates and y coordinates
    '''
    return max(abs(x2 - x1), abs(y2 - y1)) - r1 - r2


'''Update list of vertex search'''
def getDistanceVertex(vs):
    '''
    Update list of vertex search

    :param vs: list of vertex search
    :return: new list of vertex search with distance
    '''
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


'''Check is coordinates in rectangle or not'''
def inPolygon(xd, yd, x, y, w, h, f):
    '''
    Check is the coordinates (xd, yd) in rectangle

    :param xd: checking coordinate
    :param yd: checking coordinate
    :param x: coordinate of rectangle
    :param y: coordinate of rectangle
    :param w: width of rectangle
    :param h: height of rectangle
    :param f: fault
    :return: True or False
    '''
    if (((xd >= (x - f)) & (xd <= (x + w + f))) & ((yd >= (y - f)) & (yd <= (y + h + f)))):
        return True
    else:
        return False


'''Check is coordinates in rectangle or not'''
def inPolygonContur(xd, yd, x, y, w, h, f, pix=0):
    '''
    Check is the coordinates (xd, yd) in rectangle

    :param xd: checking coordinate
    :param yd: checking coordinate
    :param x: coordinate of rectangle
    :param y: coordinate of rectangle
    :param w: width of rectangle
    :param h: height of rectangle
    :param f: fault
    :param pix: extern fault
    :return: True or False
    '''
    if ( ((abs(xd - x + f) <= pix) & (y - f - pix <= yd <= y + h + f + pix)) | ((abs(xd - x - w - f) <= pix) & (y - f - pix <= yd <= y + h + f + pix)) |
           ((abs(yd - y + f) <= pix) & (x - f - pix <= xd <= x + w + f + pix)) | ((abs(yd - y - h - f) <= pix) & (x - f - pix <= xd <= x + w + f + pix))):
        return True
    else:
        return False


'''Check is coordinates in contour of rectangle or not'''
def inBoxContVertex(xd, yd, vs):
    '''
    Check is coordinates in contour of rectangle or not

    :param xd: checking coordinate
    :param yd: checking coordinate
    :param vs: vertex search
    :return: True or False
    '''
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


'''---------------------------------------------------------------------------------------------
                                Filtering coordinates of start lines
'''
'''Count similar coordinates'''
def countUnfilteringDots(predCST, i):
    '''
    Count similar coordinates

    :param predCST: coordinate start Tracker
    :param i: element for check
    :return: list of elements for remove from predCST
    '''
    filtx = [i]
    filty = [i]

    for j in predCST:
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


'''Filtering dots for start coordinate Tracker'''
def filterDots_1(coordStartTracker):
    '''
    The set of intersection points is collapsed into one point

    :param coordStartTracker: start coordinates for filtering
    :return: filtering dots
    '''
    xs = [coordStartTracker[0][1]]
    ys = [coordStartTracker[0][0]]
    ds = []

    for i in range(1, len(coordStartTracker)):
        if ((coordStartTracker[i][1] in xs) & (len(xs) == 1) &
                ((max(ys + [coordStartTracker[i][0]]) - min(ys + [coordStartTracker[i][0]])) <= 9)):
            # print("max(ys) - min(ys)", max(ys) - min(ys), sep=" ")
            ys.append(coordStartTracker[i][0])
            # print("add ys ", ys)

        elif ((coordStartTracker[i][0] in ys) & (len(ys) == 1) &
              ((max(xs + [coordStartTracker[i][1]]) - min(xs + [coordStartTracker[i][1]])) <= 9)):
            # print("max(xs) - min(xs)", max(xs) - min(xs), sep=" ")
            xs.append(coordStartTracker[i][1])
            # print("add xs ", xs)

        elif ((coordStartTracker[i][1] not in ys) | (coordStartTracker[i][0] not in xs)):
            # print("Result: ys ", ys)
            # print("Result: xs ", xs)
            # print("Result: ", (math.ceil(sum(xs) / len(xs)), math.ceil(sum(ys) / len(ys))))

            ds.append((math.ceil(sum(ys) / len(ys)), math.ceil(sum(xs) / len(xs))))

            xs.clear()
            ys.clear()
            xs.append(coordStartTracker[i][1])
            ys.append(coordStartTracker[i][0])

    # print("Result: ys ", ys)
    # print("Result: xs ", xs)
    # print("Result: ", (math.ceil(sum(xs) / len(xs)), math.ceil(sum(ys) / len(ys))))
    ds.append((math.ceil(sum(ys) / len(ys)), math.ceil(sum(xs) / len(xs))))

    if len(ds) == 1:
        return ds[0]

    return ds


'''Filtering dots for start coordinate Tracker'''
def filterDots_2(preCST):
    '''
    Count similar coordinates, remove them

    :param preCST: list for filtering
    :return: filtering list of coordinates
    '''
    for j in preCST:
        filt = countUnfilteringDots(preCST, j)
        if (len(filt) > 1):
            dis = filterDots_1(filt)
            for i in filt:
                if i in preCST:
                    preCST.remove(i)
            if dis not in preCST:
                preCST.append(dis)

    # print(preCST, len(preCST))

    for j in preCST:
        filt = countUnfilteringDots(preCST, j)
        if len(filt) != 0:
            print("Not filtering:", filt)

    return preCST


'''---------------------------------------------------------------------------------------------
                                        TRACKER
   ---------------------------------------------------------------------------------------------
'''
'''---------------------------------------------------------------------------------------------
                                    Global parametrs
'''
'''Folder to save image-results'''
folder = ""

'''Iterator for save steps of Tracker running and shooting'''
iterator_images = 0

'''ignore start of Tracker'''
ignoreStartTracker = []

'''Coordinates of road Tracker'''
coordinatesRoadTracker = []

'''List with coordinates for deleting'''
deleteCoordinates = []

'''list for circle start Tracker'''
circleStartTaracker = []

'''Preready graph list'''
graphPreReady = []

'''List with tuples of vertex connection'''
graphReady = []

'''Iter for simple output information'''
rec_iter = 0

'''Dictionary for write coordinates in file'''
dist = {}

'''Count of openning file (n <= 2)'''
n = 0


'''---------------------------------------------------------------------------------------------
                                        Standart functions
'''
'''Load data Tracker Action from file'''
def loaddatTracker(data_file):
    '''
    Load data of Tracker from file

    :param data_file: path to file
    :return: list of actions
    '''
    actions = []
    vs = open(data_file, 'r')
    for line in vs:
        cleanedLine = line.strip()
        x = cleanedLine.split(" ")
        actions.append(x)

    return actions


'''Find iterator for Tracker'''
def findIterTrackVertex(newvs, vertex_tracker):
    '''
    Find index for Tracker

    :param newvs: new vertex search after filtering and sorting
    :param vertex_tracker: dictionary with start Tracker
    :return: index for Tracker
    '''
    actions = loaddatTracker('./result/start_lines/action.txt')
    action = actions[0]
    if len(action) == 1:
        return 1
    centre = (int(action[-2]), int(action[-1]))

    i = 0
    for vertex in newvs:
        start = vertex_tracker.get(tuple(vertex))
        if (centre in start):
            return i
        i += 1
    return -1

'''---------------------------------------------------------------------------------------------
                                        Math functions
'''
'''Is Tracker in rectangle?'''
def trackerInBox(xd, yd, vs):
    '''
    Is Tracker in rectangle?

    :param xd: coordinate x
    :param yd: coordinate y
    :param vs: list of coordinates
    :return: True or False
    '''
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

        d = updateDvertex(d)

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


'''Is Tracker in circle?'''
def inCircle(xd, yd, x_c, y_c, r):
    '''
    Is Tracker in circle?

    :param xd: current coordinate x
    :param yd: current coordinate y
    :param x_c: coordinate x of centre
    :param y_c: coordinate y of centre
    :param r: radius of circle
    :return: True of False
    '''
    d = dis(xd, x_c, yd, y_c, 0, 0) - r
    if ((d < 0)): # & (inCircleCont(xd, x_c, yd, y_c, r, 0) == True)):
        return True
    return False


'''Is Tracker in counter of circle?'''
def inCircleCont(xd, yd, x_c, y_c, r, pix = 0):
    '''
    Is Tracker in counter of circle?

    :param xd: current coordinate x
    :param yd: current coordinate y
    :param x_c: coordinate x of centre
    :param y_c: coordinate y of centre
    :param r: radius of circle
    :param pix: loss (default = 0)
    :return: True of False
    '''
    if (math.ceil(abs(dis(xd, x_c, yd, y_c, 0, 0) - r)) <= pix):
        return True
    return False


'''Intersection circles'''
def intersectionCircles(x_c1, y_c1, r1, x_c2, y_c2, r2):
    '''
    Is circle intersecting?

    :param x_c1: x coordinate 1 circle
    :param y_c1: y coordinate 1 circle
    :param r1: radius 1 circle
    :param x_c2: x coordinate 2 circle
    :param y_c2: y coordinate 2 circle
    :param r2: radius 2 circle
    :return: True or False
    '''
    if (math.ceil(dis(x_c1, x_c2, y_c1, y_c2, 0, 0)) - (r1 + r2) <= 1):
        di = math.ceil(dis(x_c1, x_c2, y_c1, y_c2, 0, 0)) - (r1 + r2)
        global rec_iter
        print("     "*rec_iter, "DIS: ", di)
        dist[di] = [x_c1, y_c1, x_c2, y_c2]
        return True
    return False


'''Radius tracker'''
def getRTrack(newvs):
    '''
    Get radius tracker

    :param newvs: new vertex search
    :return: middle radius Tracker
    '''
    rad = []
    for vertex in newvs:
        d = int(vertex[7]) // 4
        rad.append(updateDvertex(d))

    return math.ceil(sum(rad) / len(rad))


'''find K: k = (f(x) - b) / x'''
def getCoefFunct(centre1, centre2):
    '''
    find K: k = (f(x) - b) / x

    :param centre1: the first coordinate
    :param centre2: the second coordinate
    :return:
    '''
    if (centre1[1] - centre2[1] != 0):
        return (round(((centre1[0] - centre2[0]) / (centre1[1] - centre2[1])), 2))
    else:
        return 0


'''Draw function for main rectangle'''
def getKRect(img_c, centre, centreRect, vertex):
    '''
    Draw function for main rectangle

    :param img_c: copy image for draw
    :param centre: center
    :param centreRect: center of rectangle
    :param vertex: list with infomation about vertex
    :return: None
    '''
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


'''---------------------------------------------------------------------------------------------
                                 Special functions for Tracker
'''
'''Filter dots'''
def trackerFilterDots_2(trackcoord_prefiltering):
    '''
    Function-filter

    :param trackcoord_prefiltering: list of coordinates to filtering
    :return: 2 filtering coordinates
    '''
    if len(trackcoord_prefiltering) == 1:
        return trackcoord_prefiltering

    dotcoord = trackcoord_prefiltering
    print("     " * rec_iter, dotcoord)

    filt_dotcoord = []

    for i in range(0, len(dotcoord)):
        # (y, x)
        xs = int(dotcoord[i][1])
        ys = int(dotcoord[i][0])

        for j in range(i + 1, len(dotcoord)):
            if ((abs(xs - dotcoord[i][1]) < 9) & (abs(ys - dotcoord[i][0]) < 9) &
                    ((abs(xs - dotcoord[i][1]) >= 0) | (abs(ys - dotcoord[i][0]) >= 0))):
                if (((dotcoord[j][0], dotcoord[j][1]) not in filt_dotcoord)):
                    filt_dotcoord.append((dotcoord[j][0], dotcoord[j][1]))
                    if ((ys, xs) not in filt_dotcoord):
                        filt_dotcoord.append((ys, xs))

    # print(filt_dotcoord)
    # print("     " * rec_iter, filt_dotcoord)
    global ignoreStartTracker
    print("     " * rec_iter, "ignoreStartTracker: ", ignoreStartTracker)

    x = []
    y = []
    for i in filt_dotcoord:
        x.append(i[1])
        y.append(i[0])

    check_ = [math.ceil(sum(y) / len(y)), math.ceil(sum(x) / len(x))]
    print("     " * rec_iter, "Check: ", (check_[1], check_[0]), ignoreStartTracker[-1])
    if ((check_[1], check_[0]) not in ignoreStartTracker):
        filt_dotcoord.clear()
        filt_dotcoord.append(tuple(check_))

    print("     " * rec_iter, "filt_dotcoord: ", filt_dotcoord)

    return filt_dotcoord


'''Intersection circles for Tracker with removing excess'''
def intersectionCirclesStartTracker(centre, starts_tracker, R):
    '''
    Intersection circles for Tracker, in function removing starts Tracker in list startTracker and
    add ignoring coordinates

    :param centre: coordinates of centre (x, y)
    :param starts_tracker: start coordinates for Tracker
    :param R: radius of centre
    :return: True - (if find Action and write coordinates Action in file), else - False (write -1
    in file)
    '''
    startTracker = starts_tracker
    # removing excess
    global ignoreStartTracker
    for i in ignoreStartTracker:
        if i in startTracker:
            startTracker.remove(i)

    for cen in startTracker:
        if (intersectionCircles(centre[0], centre[1], R, cen[0], cen[1], R) == True):
            # removing excess
            if cen not in ignoreStartTracker:
                ignoreStartTracker.append(cen)
                startTracker.remove(cen)
            if centre not in ignoreStartTracker:
                ignoreStartTracker.append(centre)
            if centre in startTracker:
                startTracker.remove(centre)

            d = min(dist.keys())    # load min key from dictionary
            res_d = dist[d]
            global n
            global rec_iter
            n += 1
            if n <= 1:
                print("     "*rec_iter, "ACTION!")
                action = open('./result/start_lines/action.txt', 'w')
                action.write(str(res_d[0]) + ' ' + str(res_d[1]) + ' ' + str(res_d[2]) + ' ' + str(res_d[3]) + '\n')
                action.close()
                del dist[d]
            return True

    # if no find Action => write in file -1
    print("     " * rec_iter, "NO ACTION!")
    action = open('./result/start_lines/action.txt', 'w')
    action.write("-1")
    action.close()
    return False


'''---------------------------------------------------------------------------------------------
                                       TRACKER RUN
'''
'''Tracker reccursive step-function'''
def trackerStepSingle(img_c, centerTracker, starts_tracker, R):
    '''
    Tracker reccursive step-function that move Tracker from start to end of line.

    :param img_c: image copy for draw results of moving Tracker
    :param centerTracker: center for start moving Tracker
    :param starts_tracker: another centers to stop the Tracker moving
    :param R: radius of circle Tracker
    :return: None
    '''
    steps = []
    centre = centerTracker
    global ignoreStartTracker
    global rec_iter
    global iterator_images
    global coordinatesRoadTracker
    global deleteCoordinates
    global circleStartTaracker
    global n

    circleStartTaracker.append(centre)

    if (centre in ignoreStartTracker):
        print("     " * rec_iter, "INFO: END of steps")
        cv2.imwrite(folder + str(iterator_images) + '_END_tracker.jpg', img_c)
        return

    ignoreStartTracker.append(centerTracker)

    print("     " * rec_iter, "INFO: Start coordinate:", centre)
    print("     " * rec_iter, "INFO: ignoreStartTracker:", ignoreStartTracker)

    if ((intersectionCirclesStartTracker(centre, starts_tracker, R) == False) &
                                                            (centre not in coordinatesRoadTracker)):
        cv2.circle(img_c, (centre[1], centre[0]), R, (0, 0, 255), 1)
        deleteCoordinates.append([centre[1], centre[0]])

        trackcoord_prefiltering = []

        # print("     "*rec_iter, "len(coordinatesRoadTracker) = ", len(coordinatesRoadTracker))

        for coord in coordinatesRoadTracker:
            if inCircleCont(coord[1], coord[0], centre[1], centre[0], R, 1) == True:
                if (coord not in trackcoord_prefiltering):
                    trackcoord_prefiltering.append([coord[1], coord[0]])
                    deleteCoordinates.append([coord[1], coord[0]])

        trackcoord_filtering = []

        if len(trackcoord_prefiltering) > 0:
            # print("     "*rec_iter, "trackcoord_prefiltering: ", trackcoord_prefiltering)
            trackcoord_filtering = trackerFilterDots_2(trackcoord_prefiltering)
            # print("     "*rec_iter, "len(trackcoord_filtering) ", len(trackcoord_filtering),
            #                                                             trackcoord_filtering)
            steps = (trackcoord_filtering[0][1], trackcoord_filtering[0][0])
            deleteCoordinates.append([trackcoord_filtering[0][1], trackcoord_filtering[0][0]])

        print("     "*rec_iter, "INFO: step loading...")

        for coord in coordinatesRoadTracker:
            if ((inCircle(coord[1], coord[0], centre[1], centre[0], R + 1) == True) &
                    (inCircleCont(coord[1], coord[0], centre[1], centre[0], R, 0) == False)):
                deleteCoordinates.append(coord)


        # print("     " * rec_iter, "len(trackcoord_filtering) ", len(trackcoord_filtering),
        #       trackcoord_filtering)

        for coord in coordinatesRoadTracker:
            if coord in deleteCoordinates:
                # print("Deliteble: ", coord)
                coordinatesRoadTracker.remove(coord)

            if coord in ignoreStartTracker:
                coordinatesRoadTracker.remove(coord)


        for coord in trackcoord_filtering:
            cv2.circle(img_c, (coord[0], coord[1]), R, (0, 0, 255), 1)

        print("     "*rec_iter, "INFO: next step!")
        print("     "*rec_iter, "INFO: len(steps): ", len(steps), type(steps), steps)

        cv2.imwrite(folder + str(iterator_images) + '_tracker.jpg', img_c)
        iterator_images += 1

        # global n
        # n = 0
        print("     " * rec_iter, "INFO: n = ", n)
        print("     "*rec_iter, steps)
        # exit(0)

        # exit
        rec_iter += 1
        # if rec_iter == 30:
        #     exit(0)

        if (len(steps) != 0):
            trackerStepSingle(img_c, steps, starts_tracker, R)

    n = 0
    print("     "*rec_iter, "INFO: END of steps")
    cv2.imwrite(folder + str(iterator_images) + '_END_tracker.jpg', img_c)
    return


'''---------------------------------------------------------------------------------------------
                                      TRACKER SHOOT
'''
'''Tracker Shoot'''
def trackerShoot(img_c, newvs, vertex_tracker, R):
    '''
    Tracker Shoot for check: is the vertex connecting?

    :param img_c: image for draw result Tracker Shooting
    :param newvs: dictionary with vertexes' lists with information
    :param vertex_tracker: for function findIterTrackVertex
    :param R: radius of Tracker
    :return: (True / False, last center Tracker stop)
    '''
    global iterator_images
    global circleStartTaracker
    global graphPreReady
    global graphReady
    global ignoreStartTracker

    actions = loaddatTracker('./result/start_lines/action.txt')

    print("     " * rec_iter, "ACTIONS: ", actions)
    action = actions[0]

    if len(action) > 1:
        di = math.ceil(
            dis(int(action[0]), int(action[-2]), int(action[1]), int(action[-1]), 0, 0)) - (R + R)
        print("     " * rec_iter, "INFO: DI = ", di)
        if (di > 1):
            return (False, tuple())

        numV = findIterTrackVertex(newvs, vertex_tracker)
        # print("     " * rec_iter, "numV = ", numV)
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
            d = updateDvertex(d)
            cv2.rectangle(img_c, (x - d, y - d), (x + w + d, y + h + d), (0, 165, 255), 2)

            # centreRect
            # cv2.circle(img_c, (x_c, y_c), 10, (255, 255, 255), 1)
            # print("     "*rec_iter, ignoreStartTracker)

            # centre = (int(action[0]), int(action[1]))
            # cv2.circle(img_c, (centre[1], centre[0]), R, (255, 255, 255), 1)
            centre1 = (int(action[-2]), int(action[-1]))
            cv2.circle(img_c, (centre1[1], centre1[0]), R, (255, 255, 255), 1)

            centre2 = (int(action[0]), int(action[1]))
            cv2.circle(img_c, (centre2[1], centre2[0]), R, (0, 0, 255), 1)

            cv2.line(img_c, (centre1[1], centre1[0]), (centre2[1], centre2[0]), (0, 255, 0), 2)

            k_find = getCoefFunct(centre1, centre2)

            if (((graphPreReady[0], vertex) not in graphReady) &
                    ((vertex, graphPreReady[0]) not in graphReady) &
                    ((vertex, graphPreReady[0]) != (vertex, vertex)) &
                    ((vertex, graphPreReady[0]) != (graphPreReady[0], graphPreReady[0]))):
                    graphReady.append((graphPreReady[0], vertex))
            graphPreReady.append(vertex)
            print("     "*rec_iter, "graphPreReady = ", graphPreReady)


            b_find = centre1[0] - k_find * centre1[1]
            print("     "*rec_iter, "K = ", k_find)
            print("     "*rec_iter, "b = ", b_find)
            cv2.line(img_c, (centre1[1], centre1[0]), (x_c, int(k_find * x_c + b_find)),
                                                                            (0, 255, 0), 2)

            getKRect(img_c, centre1, (x_c, y_c), vertex)

            iterator_images += 1

            print("     " * rec_iter, "INFORMATION: iterator_images = ", iterator_images)

            iterator_images += 1

            cv2.imwrite(folder + str(iterator_images) + '_SHOOT.jpg', img_c)

            if (inPolygon(x_c, int(k_find * x_c + b_find), x, y, w, h, r)) == True:
                return (True, centre1)
            else:
                return (False, centre1)
        else:
            return (False, tuple())

    else:
        iterator_images += 1

        print("     " * rec_iter, "SHOOT WITHOUT END!")
        print("     " * rec_iter, circleStartTaracker[-1], circleStartTaracker[-2])

        centre1 = (circleStartTaracker[-1][0], circleStartTaracker[-1][1])
        centre2 = (circleStartTaracker[-2][0], circleStartTaracker[-2][1])

        di = math.ceil(
            dis(int(centre1[1]), int(centre2[1]), int(centre1[0]), int(centre2[0]), 0, 0)) - (R + R)
        print("     " * rec_iter, "INFO: DI = ", di)
        if (di > 1):
            return (False, tuple())

        cv2.circle(img_c, (centre1[1], centre1[0]), R, (255, 255, 255), 1)

        cv2.circle(img_c, (centre2[1], centre2[0]), R, (0, 0, 255), 1)

        cv2.line(img_c, (centre1[1], centre1[0]), (centre2[1], centre2[0]), (0, 255, 0), 2)

        # k_find = getCoefFunct(centre1, centre2)
        # b_find = centre1[0] - k_find * centre1[1]

        centre2 = [circleStartTaracker[-2][0], circleStartTaracker[-2][1]]
        deltaX = (centre1[0] - centre2[0])
        # print("delta X = ", deltaX)
        deltaY = (centre1[1] - centre2[1])
        # print("delta Y = ", deltaY)

        for lots in range(len(newvs)):
            centre2[0] += deltaX
            centre2[1] += deltaY

            # print("(X, Y): ", centre2)
            cv2.circle(img_c, (centre2[1], centre2[0]), R, (0, 0, 255), 1)

            for num_V in range(len(newvs)):
                vert = newvs[num_V]

                if (inPolygon(centre2[0], centre2[1], int(vert[0]), int(vert[1]),
                                            int(vert[2]), int(vert[3]), int(vert[6]))) == True:
                    x_v = int(vert[0])
                    y_v = int(vert[1])
                    w_v = int(vert[2])
                    h_v = int(vert[3])
                    x_c_v = int(vert[4])
                    y_c_v = int(vert[5])
                    r_v = int(vert[6])
                    d_v = int(vert[7])

                    cv2.rectangle(img_c, (x_v - r_v // 4, y_v - r_v // 4),
                                  (x_v + w_v + r_v // 4, y_v + h_v + r_v // 4),
                                  (0, 0, 255), 1)
                    k_v = math.ceil(0.1 * (r_v // 4))
                    cv2.rectangle(img_c, (x_v - k_v - d_v // 4, y_v - k_v - d_v // 4),
                                  (x_v + w_v + k_v + d_v // 4, y_v + h_v + k_v + d_v // 4),
                                                                                (0, 215, 255), 2)
                    d_v = updateDvertex(d_v)
                    cv2.rectangle(img_c, (x_v - d_v, y_v - d_v),
                                            (x_v + w_v + d_v, y_v + h_v + d_v), (0, 165, 255), 2)

                    getKRect(img_c, centre1, (x_c_v, x_c_v), vert)

                    cv2.line(img_c, (centre1[1], centre1[0]), (x_c_v, y_c_v),
                             (0, 255, 0), 2)

                    if (((graphPreReady[0], vert) not in graphReady) &
                            ((vert, graphPreReady[0]) not in graphReady) &
                            ((vert, graphPreReady[0]) != (vert, vert)) &
                            ((vert, graphPreReady[0]) != (graphPreReady[0], graphPreReady[0]))):
                        graphReady.append((graphPreReady[0], vert))
                    graphPreReady.append(vert)
                    print("     " * rec_iter, "graphPreReady = ", graphPreReady)

                    iterator_images += 1

                    print("     " * rec_iter, "INFORMATION: iterator_images = ", iterator_images)

                    iterator_images += 1

                    cv2.imwrite(folder + str(iterator_images) + '_SHOOT.jpg', img_c)

                    global n
                    n = 0

                    return (True, centre1)


        print("     "*rec_iter, "graphPreReady = ", graphPreReady)

        iterator_images += 1

        print("     " * rec_iter, "iterator_images = ", iterator_images)

        iterator_images += 1

        cv2.imwrite(folder + str(iterator_images) + '_SHOOT.jpg', img_c)

        n = 0

        return (False, centre1)



