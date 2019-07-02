import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import cv2

from src.libs import lib_lineStart_Tracker as module


'''---------------------------------------------------------------------------------------------
                                      ADJECNY MATRIX
   ---------------------------------------------------------------------------------------------
'''
def GraphMatrix(graphReady, img_c, save_like_original=False):
    '''
    This function convert graphReady to adjency matrix, save it and save graph.

    :param graphReady: list of actions vertexes'
    :img_c: image for saving like original image
    :save_like_original: True - saving coordinates of vertex. False - save using adjency matrix.
    :return: None
    '''
    path = './result/adjency_matrix/'
    module.createFolder(path)

    if save_like_original == False:
        list_graph = []
        act_graph = []

        for line_draw in graphReady:
            vertexF = line_draw[0]
            x_c_F = int(vertexF[4])
            y_c_F = int(vertexF[5])

            vertexL = line_draw[1]
            x_c_L = int(vertexL[4])
            y_c_L = int(vertexL[5])

            if (x_c_F, y_c_F) not in list_graph:
                list_graph.append((x_c_F, y_c_F))
            if (x_c_L, y_c_L) not in list_graph:
                list_graph.append((x_c_L, y_c_L))

            act_graph.append(((x_c_F, y_c_F), (x_c_L, y_c_L)))

        # sorted
        list_graph.sort()
        act_graph.sort(key=lambda x: x[1])

        # numpy adjency matrix
        matrix = np.zeros((len(list_graph), len(list_graph)))

        for i in range(len(list_graph)):
            for j in range(len(list_graph)):
                if (list_graph[i], list_graph[j]) in act_graph or (
                list_graph[j], list_graph[i]) in act_graph:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0

        np.savetxt(path + 'adjency_matrix.matrix', matrix, delimiter=',')
        G = nx.from_numpy_matrix(matrix)
        nx.draw(G)
        plt.savefig(path + "Graph_adjency_matrix.png", format="PNG")

    else:

        height, width, channels = img_c.shape
        tracker_predict = np.zeros((height, width, channels), np.uint8)

        for line_draw in module.graphReady:
            vertexF = line_draw[0]
            x_c_F = int(vertexF[4])
            y_c_F = int(vertexF[5])
            r_F = int(vertexF[6])
            cv2.circle(tracker_predict, (x_c_F, y_c_F), r_F, (0, 0, 255), 2)

            vertexL = line_draw[1]
            x_c_L = int(vertexL[4])
            y_c_L = int(vertexL[5])
            r_L = int(vertexL[6])
            cv2.circle(tracker_predict, (x_c_L, y_c_L), r_L, (0, 0, 255), 2)

            cv2.line(tracker_predict, (x_c_F, y_c_F), (x_c_L, y_c_L), (0, 0, 255), 2)

        cv2.imwrite(module.folder + "image_predict.jpg", tracker_predict)
    return