import matplotlib.pyplot as plt
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


def inPolygon(xd, yd, x, y, w, h):
    if (((xd >= x) & (xd <= (x + w))) & ((yd >= y) & (yd <= (y + h)))):
        return True
    else:
        return False


def find_start(image):
    img = cv2.imread(image)

    img_c = img.copy()

    gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 2)

    edges = cv2.Canny(gray, 100, 255)

    vs = loaddat(path)

    lines_start = []

    it = 0
    for vertex in vs:
        # vertex_sort.append([x, y, w, h, (x_c, y_c)])
        x = int(vertex[0])
        y = int(vertex[1])
        w = int(vertex[2])
        h = int(vertex[3])
        x_c = int(vertex[4])
        y_c = int(vertex[5])
        r = math.ceil((math.ceil(w/2) + math.ceil(h/2))/2)

        img_c2 = img.copy()
        # cv2.rectangle(img_c2, (x, y), (x + w, y + h), (255, 0, 255), 2)
        # cv2.rectangle(img_c2, (x - r, y - r), (x + w + r, y + h + r), (0, 255, 0), 2)

        gray2 = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

        edges2 = cv2.Canny(gray2, 100, 255)

        # # 1
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        # crop_img = img_c[y:y + h, x:x + w]
        #
        # # 2
        # cv2.rectangle(img, (x - r, y - r), (x + w + r, y + h + r), (0, 255, 0), 2)
        # crop_img2 = img_c2[y - r:y + h + r, x - r:x + w + r]
        #
        # gray2 = cv2.cvtColor(crop_img2, cv2.COLOR_BGR2GRAY)
        #
        # edges2 = cv2.Canny(gray2, 100, 255)

        # next step
        # HoughLinesP

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 2  # minimum number of pixels making up a line
        max_line_gap = 3  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img_c2) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges2, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

        i = 0
        for line in lines:
            for x1, y1, x2, y2 in line:
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                # print((inPolygon(x1, y1, x, y, w, h)), inPolygon(x2, y2, x, y, w, h))
                if ((inPolygon(x1, y1, x - r//4, y - r//4, w + r//4, h + r//4) != True) &
                        (inPolygon(x2, y2, x - r//4, y - r//4, w + r//4, h + r//4) != True) &
                        (inPolygon(x1, y1, x - r, y - r, w + 2*r, h + 2*r) == True) &
                        (inPolygon(x2, y2, x - r, y - r, w + 2*r, h + 2*r) == True)):
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

                    lines_start.append([x1, y1, x2, y2])

                i += 1

        lines_edges = cv2.addWeighted(img_c2, 0.8, line_image, 1, 0)

        cv2.imwrite(folder + str(it) + '.jpg', lines_edges)

        it += 1

        # cv2.imshow("1 rect", img)
        # cv2.imshow("crop_img", img_c2)
        # cv2.imshow("crop_img2", lines_edges)

        #cv2.waitKey(0)

        # k = cv2.waitKey(33)
        # if k == 27:
        #     continue


        # k = cv2.waitKey(33)
        # if k == 27:  # Esc key to stop
        #     break
        # elif k == -1:  # normally -1 returned,so don't print it
        #     continue
        # else:
        #     print(k)  # else print its value
    # # HoughLinesP
    # rho = 1  # distance resolution in pixels of the Hough grid
    # theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    # min_line_length = 9  # minimum number of pixels making up a line
    # max_line_gap = 9  # maximum gap in pixels between connectable line segments
    # line_image = np.copy(img) * 0  # creating a blank to draw lines on
    #
    # # Run Hough on edge detected image
    # # Output "lines" is an array containing endpoints of detected line segments
    # lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
    #                         min_line_length, max_line_gap)

    # i = 0
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    #
    #         # steps
    #         # lines_edge = cv2.addWeighted(img_c, 0.8, line_image, 1, 0)
    #         # cv2.imwrite(folder + str(i) + '.jpg', lines_edge)
    #
    #         i += 1
    #
    # print(i)
    # # Draw the lines on the  image
    # lines_edges = cv2.addWeighted(img_c, 0.8, line_image, 1, 0)
    #
    # cv2.imwrite(folder + 'lines_edges.jpg', lines_edges)



    # vertex_sort.append([x, y, w, h, (x_c, y_c)])
    ## cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # filters
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # gaus = cv2.GaussianBlur(gray, (3, 3), 0)
    # # Canny
    # edges = cv2.Canny(gray, 100, 255)
    # # Find coordinates of edges
    # indices = np.where(edges != [0])
    # print(indices)
    # #     print('y: ', indices[0], '\nx: ', indices[1])
    # # coordinates = zip(indices[1], indices[0])
    #
    # # make coordinates in simple form
    # detcoord = []
    #
    # datcoord = folder + 'SobelOutput_y_x' + '.txt'
    # dat = open(datcoord, 'a')
    #
    # # dat.write('y x\n')
    #
    # for i in range(0, len(indices[0])):
    #     dat.write(str(indices[0][i]) + ' ' + str(indices[1][i]) + '\n')
    #     detcoord.append([indices[0][i], indices[1][i]])
    #
    # dat.close()
    #
    # cv2.imwrite(folder + 'Sobel.jpg', edges)
    #
    # return (detcoord)

    for st in lines_start:
        x1 = st[0]
        y1 = st[1]
        x2 = st[2]
        y2 = st[3]

        cv2.line(img_c, (x1, y1), (x2, y2), (255, 0, 0), 5)

    for vertex in vs:
        x = int(vertex[0])
        y = int(vertex[1])
        w = int(vertex[2])
        h = int(vertex[3])
        x_c = int(vertex[4])
        y_c = int(vertex[5])
        r = math.ceil((math.ceil(w / 2) + math.ceil(h / 2)) / 2)

        cv2.rectangle(img_c, (x - r//4, y - r//4), (x + w + r//4, y + h + r//4), (255, 0, 255), 2)
        cv2.rectangle(img_c, (x - r, y - r), (x + w + r, y + h + r), (0, 255, 0), 2)

    cv2.imwrite(folder + "result.jpg", img_c)



if __name__ == '__main__':
    # create folder
    folder = './result/start_lines/'
    createFolder(folder)

    # image
    image = sys.argv[1]

    # path to folder
    path = sys.argv[2] #"./vs2/vertex_search.graphvs"

    find_start(image)

    sys.exit(0)