import cv2
import numpy as np
import imutils
import math
from PIL import Image
from matplotlib import pyplot as plt


# create class Tracker-pix
class Tracker_pix:
    '''
    Class Tracker_pix:
        coord: function return coordinates of edge

        getstartcoord: return start coordinates for Tracker_pix

        direct: function return the direction of Tracker_pix

        linecoord: return coordinates of line

        isline: return is it line

        circlecoord: return coordinates of circle

        iscircle: return is it circle
    '''

    '''
    Write in list coordinates
    '''
    Coordinates = []
    Rectpix = []

    def __init__(self, image, size=(5, 5)):
        """Constructor"""
        self.image = cv2.imread(image, 0)
        self.size = size

    def coord(self):
        """
        function return coordinates of edge
        """
        img = self.image
        edges = cv2.Canny(img, 100, 255)

        # Find coordinates of edges
        indices = np.where(edges != [0])
        # print('y: ', indices[0], '\nx: ', indices[1])
        self.Coordinates = indices

        return indices

    def getstartcoord(self):
        """
        return start coordinates for Tracker_pix
        """
        indices = self.Coordinates

        ystart = indices[0][0]
        xstart = indices[1][0]
        dmin = math.ceil(math.sqrt(indices[1][0] ** 2 + indices[0][0] ** 2))
        #     print(dmin)
        for i in range(1, len(indices[0])):
            d = math.ceil(math.sqrt(indices[1][i] ** 2 + indices[0][i] ** 2))
            if d < dmin:
                ystart = indices[0][i]
                xstart = indices[1][i]
                dmin = d
                # print(dmin)
        return ([xstart, ystart])

    def direct(self):
        

