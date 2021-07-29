import cv2
import numpy as np
import torch
import os
import time

from .opts import opts
from .tracker_trt import FairTracker
from .fairmot.utils.transformation import *
from .fairmot.tracking_utils import visualization as vis
from .fairmot.tracking_utils.log import logger
from .test_utils import write_results


"""
This is the kernel of system and do the tracking task.
It can accept tow intput format. One is the image sequence direction, the other is a video file.
For the image sequence direction, it should be the format similar to the following:
    - DIRECTION "img1": the image set of all image files named like "000397"
    - INI FILE "seqinfo.ini": the file has some information about sequence, for instance:

        [Sequence]
        name=MOT17-01-FRCNN
        imDir=img1
        frameRate=30
        seqLength=450
        imWidth=1920
        imHeight=1080
        imExt=.jpg

And for the video file, it should be a file type which can be loaded by cv2.videocapture.
"""
class TrackingKernel:
    def __init__(self, enableFP16):
        self.enableFP16 = enableFP16

    def init_kernel(self, frame_rate, image_size, target_size, opt):
        """ Create kernel only. """
        self.image_size = image_size
        self.target_size = target_size
        # tracker_init
        # setup the MOT_Tracker
        self.tracker = FairTracker(opt, frame_rate, self.enableFP16)

    def pre_processing(self, img0):
        img = np.array(img0)
        # Padded resize
        img, _, _, _ = letterbox(img, height=self.target_size[0], width=self.target_size[1])
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return img
    
    def call_once(self, img0, dense_region=None, dense_region0=None):
        """ Kernel Executor in a single step. Current only support for video file. """
        st = time.time()
        #### PRE PROCESSING ####
        img = self.pre_processing(img0)
        if dense_region is not None:
            img = img[:, dense_region[1]:dense_region[3], dense_region[0]:dense_region[2]]
        blob = np.expand_dims(img, axis=0)
        #### UPDATE TRACKER ####
        #### ********* ####
        ret_trks = []
        # blob = torch.from_numpy(img).cuda().unsqueeze(0)

        #### TRACKING ####
        trks = self.tracker.update(blob, self.image_size, dense_region0)
        et = time.time()
        return trks, et-st

    def infer_preprocessing(self, img):
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return img

    def infer(self, img):
        """ Kernel Executor in a single step. Current only support for video file. """
        #### PRE PROCESSING ####
        st = time.time()
        img = self.infer_preprocessing(img)
        #### UPDATE TRACKER ####
        #### ********* ####
        ret_trks = []
        blob = np.expand_dims(img, axis=0)

        #### TRACKING ####
        dets = self.tracker.infer(blob, self.image_size)
        et = time.time()
        return dets, et-st