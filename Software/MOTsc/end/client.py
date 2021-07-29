from sklearn.cluster import DBSCAN
import sys
import os
import cv2
import time
import copy
import zmq
import sys
import PIL.Image as Image
import numpy as np
import io
import torch

from TrackingKernel.tracker import STrack
from fairmot.tracking_utils import visualization as vis
from fairmot.tracking_utils.log import logger
from fairmot.utils.transformation import letterbox_parameter_computing, letterbox_bbx_inversemap, letterbox_bbx_map
from fairmot.utils.drawing import rectangle
from fairmot.utils.transformation import *
from tracker import STrack, FairTracker
from test_utils import write_results
from opts import opts
from divider import Divider
from allocator import allocate, joint
from offloader import offload


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
class TrackingKernel_infer:
    def init_kernel(self, frame_rate, image_size, target_size, opt):
        """ Create kernel only. """
        self.image_size = image_size
        self.target_size = target_size
        
        # tracker_init
        # setup the MOT_Tracker
        self.tracker = FairTracker(opt, frame_rate)

    def pre_processing(self, img):
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return img
    
    def call_once(self, img):
        """ Kernel Executor in a single step. Current only support for video file. """
        #### PRE PROCESSING ####
        st = time.time()
        print(img.shape)
        img = self.pre_processing(img)
        #### UPDATE TRACKER ####
        #### ********* ####
        ret_trks = []
        blob = torch.from_numpy(img).cuda().unsqueeze(0)

        #### TRACKING ####
        dets = self.tracker.infer(blob, self.image_size)
        et = time.time()
        return dets, et-st

    def track(self, online_dets):
        online_trks = self.tracker.track(online_dets)
        return online_trks

def pre_processing(img0, target_size):
    img = np.array(img0)
    # Padded resize
    img, _, _, _ = letterbox(img, height=target_size[0], width=target_size[1])
    return img

if __name__ == "__main__":
    #### edge-end ####
    context = zmq.Context()

    print("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://192.168.1.105:5555")
    edge_show = True
    
    #### basic ####
    opt = opts().init()
    input_file = '../../../data/demo.avi'
    show_image = False
    # save_dir = '../../outputs'
    save_dir = None
    result_filename = 'demo.txt'
    data_type = 'mot'

    vc = cv2.VideoCapture(input_file)
    frame_rate = vc.get(cv2.CAP_PROP_FPS)
    raw_shape_wh = (int(vc.get(3)), int(vc.get(4)))
    max_frame_idx = vc.get(cv2.CAP_PROP_FRAME_COUNT)

    raw_shape_hw = [raw_shape_wh[1], raw_shape_wh[0]]
    target_shape_hw = [608, 1088]

    tk = TrackingKernel_infer()
    tk.init_kernel(frame_rate, raw_shape_hw, target_shape_hw, opt)

    #### ATS ####
    divider_cycle = 1
    detection_cycle = 1
    divider = Divider()             # create divider
    dense_region = (0, 0, 1088, 608)
    dense_region0 = (0, 0, raw_shape_hw[1], raw_shape_hw[0])
    divider_cycle_counter = 0
    detection_cycle_counter = 0

    print("end is ready")
    
    socket.send(b'')
    socket.recv()
    
    
    #### network init ####
    socket.send(b"init")
    message = socket.recv()
    print(f"Received reply init [ {message} ]")
    socket.send(bytes(str(frame_rate).encode('utf-8')))
    message = socket.recv()
    print(f"Received reply {frame_rate} [ {message} ]")
    socket.send(bytes(str(max_frame_idx).encode('utf-8')))
    message = socket.recv()
    print(f"Received reply {max_frame_idx} [ {message} ]")
    socket.send(bytes(str(raw_shape_hw[0]).encode('utf-8')))
    message = socket.recv()
    print(f"Received reply {raw_shape_hw[0]} [ {message} ]")
    socket.send(bytes(str(raw_shape_hw[1]).encode('utf-8')))
    message = socket.recv()
    print(f"Received reply {raw_shape_hw[1]} [ {message} ]")
    


    # parameters configure
    socket.send(b'')
    enableTimeTrick = socket.recv_pyobj()
    socket.send(b'')
    enableSpaceTrick = socket.recv_pyobj()
    socket.send(b'')
    detection_cycle = socket.recv_pyobj()
    socket.send(b'')
    divider_cycle = socket.recv_pyobj()
    socket.send(b'')
    K = socket.recv_pyobj()
    print(f"K {K}")
    socket.send(b'')
    enableTRT = socket.recv_pyobj()
    print(f"enableTRT {enableTRT}")
    testpkg = np.random.randn(1000).tobytes()

    target_size = [608, 1088]

    speed_lst = []
    # test bandwidth
    _, img0 = vc.read()
    img = pre_processing(img0, target_size)
    encoded, buffer = cv2.imencode('.jpg', img[..., ::-1])

    # K = 10
    noise = 0
    for i in range(K):
        st = time.time()
        socket.send(buffer)
        socket.recv()
        datasize = sys.getsizeof(buffer)
        speed = datasize / (time.time() - st)
        print(f"Datasize: {datasize} B; Speed: {speed} B/s")
        speed_lst.append(speed)
    current_speed = sum(speed_lst) / K - noise
    if current_speed < 0:
        current_speed = 1
    print(f"Init bandwidth: {current_speed} B/s")

    # init 1 frame to test
    print("sent test")
    socket.send(buffer)
    socket.recv()
    online_dets_end, timetick = tk.call_once(img)
    print("local test finished")
    socket.send(b'')
    targets = socket.recv_pyobj()
    print("get test")

    if enableSpaceTrick and enableTimeTrick:
        divider_cycle = divider_cycle
        detection_cycle = detection_cycle
        print(f"divider_cycle: {divider_cycle}, detection_cycle: {detection_cycle}")
    elif enableSpaceTrick:
        divider_cycle = divider_cycle
        detection_cycle = 0
        print(f"divider_cycle: {divider_cycle}")
    elif enableTimeTrick:
        divider_cycle = 0
        detection_cycle = detection_cycle
        print(f"detection_cycle: {detection_cycle}")
    else:
        divider_cycle = 0
        detection_cycle = 0
        print("no tricks")
    divider_cycle_counter = 0
    detection_cycle_counter = 0
    
    ratio, new_shape, dw, dh = letterbox_parameter_computing(raw_shape_hw, target_shape_hw)
    s_tracked_stracks = []
    f_tracked_stracks = []
    s_lost_stracks = []
    f_lost_stracks = []
    results = [] 
    frame_idx = 1
    time_all = 0
    ## Execute start
    while frame_idx <= max_frame_idx:
        # Load image
        print("Processing frame %05d" % frame_idx)
        #### LOADING IMAGE ####
        if frame_idx != 1:
            _, img0 = vc.read()
        time_st = time.time()
        if enableSpaceTrick and detection_cycle_counter == 0 and divider_cycle_counter == 0:
            width = 1088
            heights = [int(i * 608 / 19) for i in range(19 + 1)]
            ## Aggregate stracks
            tk.tracker.tracked_stracks = joint(tk.tracker.tracked_stracks, f_tracked_stracks)
            tk.tracker.lost_stracks = joint(tk.tracker.lost_stracks, f_lost_stracks) 
            ## Global detection
            img = pre_processing(img0, target_size)
            # Offloading evaluation
            heights_choice = offload(current_speed,False,enableTRT)
            end_height = heights[heights_choice]
            edge_height = heights[19 - heights_choice]
            rend = end_height * width * 3
            redge = edge_height * width * 3
            print(f"The optimized choice is: end {end_height}, edge {edge_height}")

            # offloading
            img_end = img[edge_height:, :, :]
            if heights_choice != 19:
                img_edge = img[:edge_height, :, :]
                encoded, buffer = cv2.imencode('.jpg', img_edge[..., ::-1])
                st = time.time()
            else:
                buffer = testpkg
                socket.send(b'test')
                socket.recv()
            datasize = sys.getsizeof(buffer)
            socket.send(buffer)
            socket.recv()
            speed = datasize / (time.time() - st)
            print(f"Datasize: {datasize} B; Speed: {speed} B/s")

            speed_lst.append(speed)
            current_speed = sum(speed_lst[-K:]) / K - noise
            if current_speed < 0:
                current_speed = 1
            print(f"Current avg speed: {current_speed} B/s")
            
            online_dets_end = []
            if heights_choice != 0:
                # Infer locally
                online_dets_end, timetick = tk.call_once(img_end)
                for det in online_dets_end:
                    det._tlwh[1] += (edge_height-dh) / ratio

            online_dets_edge = []
            if heights_choice != 19:
                # Get results from edge
                socket.send(b'')
                online_dets_edge = socket.recv_pyobj()
            else:
                socket.send(b'')
                socket.recv_pyobj()

            online_dets = online_dets_end + online_dets_edge
            online_targets = tk.track(online_dets)
        
            ## Divide region
            ddets = []
            for trk in online_targets:
                bbx = trk.tlwh
                ddet = letterbox_bbx_map(bbx, ratio, dw, dh)
                ddets.append(ddet)
            dense_region, divide_ok = divider.divide(ddets) # anchored on target_shape
            dense_region0 = letterbox_bbx_inversemap(dense_region, ratio, dw, dh)
            if divide_ok:
                ## Allocate stracks by region
                tk.tracker.tracked_stracks, f_tracked_stracks = allocate(tk.tracker.tracked_stracks, dense_region0)
                tk.tracker.lost_stracks, f_lost_stracks = allocate(tk.tracker.lost_stracks, dense_region0)
            else:
                dense_region0 = (0, 0, raw_shape_hw[1], raw_shape_hw[0])
            detection_cycle_counter = detection_cycle
            divider_cycle_counter = divider_cycle
        elif enableSpaceTrick and detection_cycle_counter == 0:
            width = 576
            heights = [int(i * 320 / 10) for i in range(10 + 1)]
            ## Dense region detection
            img = pre_processing(img0, target_size)
            if enableSpaceTrick:
                img = img[dense_region[1]:dense_region[3], dense_region[0]:dense_region[2], :]
            # Offloading evaluation
            heights_choice = offload(current_speed,True,enableTRT)
            end_height = heights[heights_choice]
            edge_height = heights[10 - heights_choice]
            rend = end_height * width * 3
            redge = edge_height * width * 3
            print(f"The optimized choice is: end {end_height}, edge {edge_height}")

            # offloading
            img_end = img[edge_height:, :, :]
            if heights_choice != 10:
                img_edge = img[:edge_height, :, :]
                encoded, buffer = cv2.imencode('.jpg', img_edge[..., ::-1])
                st = time.time()
            else:
                buffer = testpkg
                socket.send(b'test')
                socket.recv()
            datasize = sys.getsizeof(buffer)
            socket.send(buffer)
            socket.recv()
            speed = datasize / (time.time() - st)
            print(f"Datasize: {datasize} B; Speed: {speed} B/s")

            speed_lst.append(speed)
            current_speed = sum(speed_lst[-K:]) / K - noise
            if current_speed < 0:
                current_speed = 1
            print(f"Current avg speed: {current_speed} B/s")
            
            online_dets_end = []
            if heights_choice != 0:
                # Infer locally
                online_dets_end, timetick = tk.call_once(img_end)
                for det in online_dets_end:
                    det._tlwh[1] += (edge_height-dh) / ratio

            online_dets_edge = []
            if heights_choice != 10:
                # Get results from edge
                socket.send(b'')
                online_dets_edge = socket.recv_pyobj()
            else:
                socket.send(b'')
                socket.recv_pyobj()

            online_dets = online_dets_end + online_dets_edge
            # tune by dense_region0
            for det in online_dets:
                det._tlwh[0] += dense_region0[0]
                det._tlwh[1] += dense_region0[1]
            s_online_targets = tk.track(online_dets)
            if enableSpaceTrick:
                ## Sparse region detecction
                STrack.multi_predict(f_tracked_stracks)
            online_targets = joint(s_online_targets, f_tracked_stracks)
            detection_cycle_counter = detection_cycle
            divider_cycle_counter -= 1
        elif detection_cycle_counter == 0:
            width = 1088
            heights = [int(i * 608 / 19) for i in range(19 + 1)]
            ## Aggregate stracks
            tk.tracker.tracked_stracks = joint(tk.tracker.tracked_stracks, f_tracked_stracks)
            tk.tracker.lost_stracks = joint(tk.tracker.lost_stracks, f_lost_stracks) 
            ## Global detection
            img = pre_processing(img0, target_size)
            # Offloading evaluation
            heights_choice = offload(current_speed,False,enableTRT)
            end_height = heights[heights_choice]
            edge_height = heights[19 - heights_choice]
            rend = end_height * width * 3
            redge = edge_height * width * 3
            print(f"The optimized choice is: end {end_height}, edge {edge_height}")

            # offloading
            img_end = img[edge_height:, :, :]
            if heights_choice != 19:
                img_edge = img[:edge_height, :, :]
                encoded, buffer = cv2.imencode('.jpg', img_edge[..., ::-1])
                st = time.time()
            else:
                buffer = testpkg
                socket.send(b'test')
                socket.recv()
            datasize = sys.getsizeof(buffer)
            socket.send(buffer)
            socket.recv()
            speed = datasize / (time.time() - st)
            print(f"Datasize: {datasize} B; Speed: {speed} B/s")

            speed_lst.append(speed)
            current_speed = sum(speed_lst[-K:]) / K - noise
            if current_speed < 0:
                current_speed = 1
            print(f"Current avg speed: {current_speed} B/s")
            
            online_dets_end = []
            if heights_choice != 0:
                # Infer locally
                online_dets_end, timetick = tk.call_once(img_end)
                for det in online_dets_end:
                    det._tlwh[1] += (edge_height-dh) / ratio

            online_dets_edge = []
            if heights_choice != 19:
                # Get results from edge
                socket.send(b'')
                online_dets_edge = socket.recv_pyobj()
            else:
                socket.send(b'')
                socket.recv_pyobj()

            online_dets = online_dets_end + online_dets_edge
            online_targets = tk.track(online_dets)
            detection_cycle_counter = detection_cycle
            divider_cycle_counter -= 1
        else:
            width = 0
            # Dense region detection
            STrack.multi_predict(tk.tracker.tracked_stracks)
            # Sparse region detecction
            STrack.multi_predict(f_tracked_stracks)
            online_targets = joint(tk.tracker.tracked_stracks, f_tracked_stracks)
            detection_cycle_counter -= 1
            socket.send(b'f_turn')
            socket.recv()

        timetick = time.time() - time_st
        time_all += timetick
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        # save results
        results.append((frame_idx, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_idx)
        if show_image:
            cv2.imshow('online_im', online_im)
            cv2.waitKey(1)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_idx)), online_im)
        if edge_show:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_idx)
            rectangle(online_im,dense_region0[0],dense_region0[1],dense_region0[2]-dense_region0[0],dense_region0[3]-dense_region0[1],(0,140,255),label="DenseRegion",thickness=2)
            
            if width == 576:
                cv2.line(online_im,(dense_region0[0],dense_region0[1]+int((edge_height-dh)/ratio)), (dense_region0[0]+int((width-dw)/ratio),dense_region0[1]+int((edge_height-dh)/ratio)), (0,255,0),3)
            elif width == 1088:
                cv2.line(online_im,(0,int((edge_height-dh)/ratio)), (int((width-dw)/ratio),int((edge_height-dh)/ratio)), (0,255,0),3)
            encoded, buffer = cv2.imencode('.jpg', online_im[..., ::-1])
            socket.send(buffer)
            socket.recv()
            fps = frame_idx / time_all
            socket.send_pyobj(fps)
            socket.recv()
        logger.info('Processing frame {} ({:.2f} fps)'.format(frame_idx, frame_idx / time_all))

        frame_idx += 1
    # save results
    write_results(result_filename, results, data_type)
