import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class Divider(object):
    def __init__(self):
        # self.dense_region = (0, 0, 0, 0)
        # self.dense_region = (0,0,576,320)
        # self.dense_region = (0,0,864,480)
        # self.dense_region = (512,288,1088,608)
        self.dense_region = [0,0,1088,608]
        # self.dense_region = (300,0,1388,1080)
        # self.dense_region = (320, 320, 384, 416)
        self.showing = False

    def divide(self, dets=None):
        # det: [l, t, w, h]
        centers_x = []
        centers_y = []
        for det in dets:
            centers_x.append(det[0]+(det[2]/2))
            centers_y.append(det[1]+(det[3]/2))
        x_array = np.array(centers_x)
        y_array = np.array(centers_y)
        data = np.array(list(zip(x_array, y_array))).reshape(len(x_array),2)
        # pred = DBSCAN(eps=150, min_samples = 8).fit_predict(data)
        pred = DBSCAN(eps=100, min_samples = 3).fit_predict(data)
        
        ##
        if self.showing:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.cla()
            plt.ion()
            plt.scatter(data[:,0], data[:, 1], c=pred)

        max_N = max(pred)
        center_dict_x = {}
        center_dict_y = {}
        len_N = []
        require_N = 1
        for l in range(0, max_N+1):
            if l != -1:
                center_dict_x[l] = []
                center_dict_y[l] = []
        for i,l in enumerate(pred):
            if l != -1:
                center_dict_x[l].append(x_array[i])
                center_dict_y[l].append(y_array[i])
        for l in range(0, max_N+1):
            len_N.append(len(center_dict_x[l]))
        if len(len_N)==0:
            ##
            if self.showing:
                plt.xlim((0, 1088))
                plt.ylim((0, 608))
                plt.show()
                plt.pause(0.5)
            return self.dense_region, 0
        require_l = []
        while require_N > 0:
            require_l.append(len_N.index(max(len_N)))
            require_N -= 1
        for l in require_l:
            x = round(np.mean(center_dict_x[l]))
            y = round(np.mean(center_dict_y[l]))
            
            ##
            if self.showing:
                rect = plt.Rectangle((x-288,y-160),576,320,fill=False,color='r')
                ax.add_patch(rect)

            self.dense_region = [x-288,y-160, x+288, y+160] #(576, 320)
            
            # Secure
            if self.dense_region[0] < 0:
                self.dense_region[0] = 0
                self.dense_region[2] = 576
            if self.dense_region[1] < 0:
                self.dense_region[1] = 0
                self.dense_region[3] = 320
            if self.dense_region[2] > 1088:
                self.dense_region[2] = 1088
                self.dense_region[0] = 1088 - 576
            if self.dense_region[3] > 608:
                self.dense_region[3] = 608
                self.dense_region[1] = 608 - 320

            # self.dense_region = (0,0,1088,608)
        if self.showing:
            plt.xlim((0, 1088))
            plt.ylim((0, 608))
            plt.show()
            plt.pause(0.5)
        # self.dense_region = (0,0,0,0)
        return self.dense_region, 1

    