import sys
import os
import cv2
import time
import threading
import copy
import zmq
import sys
import PIL.Image as Image
import numpy as np
import io
import PySide2
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from UI.MainWindow import Ui_MainWindow
from UI.toItemWidget import toItemWidget
from TrackingKernel.TrackingKernel import TrackingKernel as TK_basic
from TrackingKernel.TrackingKernel_trt import TrackingKernel as TK_trt
from TrackingKernel.fairmot.tracking_utils import visualization as vis
from TrackingKernel.fairmot.utils.transformation import letterbox_parameter_computing, letterbox_bbx_inversemap, letterbox_bbx_map
from TrackingKernel.fairmot.utils.drawing import rectangle
from TrackingKernel.tracker import STrack
from test_utils import write_results
from opts import opts
from divider import Divider
from allocator import allocate, joint

class Communicate(QObject):
    display_singal = Signal(threading.Event)
    idupdated_singal = Signal(threading.Event)
    fpsupdated_signal = Signal(threading.Event)


class Worker(QThread):
    def __init__(self, parent):
        super().__init__()
        self.signals = Communicate()
        self.parent = parent
        self.signals.display_singal.connect(parent.Display)
        self.signals.idupdated_singal.connect(parent.UpdateId)
        self.signals.fpsupdated_signal.connect(parent.UpdateFps)
        # self.signals.result_signal.connect(self.parent.showResult)
        # self.signals.configure_signal.connect(parent.setConfigre)
        self.enableTRT = parent.enableTRT
        self.enableFP16 = parent.enableFP16
        self.enableTimeTrick = parent.enableTimeTrick
        self.enableSpaceTrick = parent.enableSpaceTrick
        self.divider_cycle = parent.divider_cycle
        self.detection_cycle = parent.detection_cycle

    def run(self):
        print("this")
        opt = opts().init()
        vc = cv2.VideoCapture(self.parent.input_file)
        frame_rate = vc.get(cv2.CAP_PROP_FPS)
        raw_shape_wh = (int(vc.get(3)), int(vc.get(4)))
        max_frame_idx = vc.get(cv2.CAP_PROP_FRAME_COUNT)

        raw_shape_hw = [raw_shape_wh[1], raw_shape_wh[0]]
        target_shape_hw = [608, 1088]
        print("video loaded")
        if self.enableTRT:
            tk = TK_trt(self.enableFP16)
        else:
            tk = TK_basic()
        tk.init_kernel(frame_rate, raw_shape_hw, target_shape_hw, opt)
        resolution = str(tk.image_size[1]) + " x " + str(tk.image_size[0])
        self.parent.resolutionLabel.setText(resolution)
        self.parent.playing = True
        self.parent.opened = True

        # create divider
        divider = Divider()
        results = []
        frame_idx = 1
        time_all = 0
        ### ratio compute ###
        ratio, new_shape, dw, dh = letterbox_parameter_computing(raw_shape_hw, target_shape_hw)
        dense_region = (0, 0, 1088, 608)
        dense_region0 = (0, 0, raw_shape_hw[1], raw_shape_hw[0])
        if self.enableSpaceTrick and self.enableTimeTrick:
            divider_cycle = self.divider_cycle
            detection_cycle = self.detection_cycle
            print(f"divider_cycle: {divider_cycle}, detection_cycle: {detection_cycle}")
        elif self.enableSpaceTrick:
            divider_cycle = self.divider_cycle
            detection_cycle = 0
            print(f"divider_cycle: {divider_cycle}")
        elif self.enableTimeTrick:
            divider_cycle = 0
            detection_cycle = self.detection_cycle
            print(f"detection_cycle: {detection_cycle}")
        else:
            divider_cycle = 0
            detection_cycle = 0
            print("no tricks")
        divider_cycle_counter = 0
        detection_cycle_counter = 0
        s_tracked_stracks = []
        f_tracked_stracks = []
        s_lost_stracks = []
        f_lost_stracks = []
        while True:
            if self.parent.opened:
                if self.parent.playing:
                    if frame_idx <= max_frame_idx:
                        #### LOADING IMAGE ####
                        _, img0 = vc.read()
                        if self.enableSpaceTrick and detection_cycle_counter == 0 and divider_cycle_counter == 0:
                            time_st = time.time()
                            # Aggregate stracks
                            tk.tracker.tracked_stracks = joint(tk.tracker.tracked_stracks, f_tracked_stracks)
                            tk.tracker.lost_stracks = joint(tk.tracker.lost_stracks, f_lost_stracks) 
                            # Global detection
                            online_targets,_ = tk.call_once(img0)
                            # Divide region
                            ddets = []
                            for trk in online_targets:
                                bbx = trk.tlwh
                                ddet = letterbox_bbx_map(bbx, ratio, dw, dh)
                                ddets.append(ddet)
                            dense_region, divide_ok = divider.divide(ddets) # anchored on target_shape
                            dense_region0 = letterbox_bbx_inversemap(dense_region, ratio, dw, dh)
                            if divide_ok:
                                # Allocate stracks by region
                                tk.tracker.tracked_stracks, f_tracked_stracks = allocate(tk.tracker.tracked_stracks, dense_region0)
                                tk.tracker.lost_stracks, f_lost_stracks = allocate(tk.tracker.lost_stracks, dense_region0)
                            else:
                                dense_region0 = (0, 0, raw_shape_hw[1], raw_shape_hw[0])
                            detection_cycle_counter = detection_cycle
                            divider_cycle_counter = divider_cycle
                            
                            timetick = time.time() - time_st
                        elif detection_cycle_counter == 0:
                            time_st = time.time()
                            if self.enableSpaceTrick:
                                # Dense region detection
                                s_online_targets, timetick = tk.call_once(img0, dense_region, dense_region0)
                                # Sparse region detecction
                                STrack.multi_predict(f_tracked_stracks)
                            else:
                                s_online_targets, timetick = tk.call_once(img0)
                            online_targets = joint(s_online_targets, f_tracked_stracks)
                            detection_cycle_counter = detection_cycle
                            divider_cycle_counter -= 1
                            timetick = time.time() - time_st
                        else:
                            time_st = time.time()
                            # Dense region detection
                            STrack.multi_predict(tk.tracker.tracked_stracks)
                            # Sparse region detecction
                            STrack.multi_predict(f_tracked_stracks)
                            online_targets = joint(tk.tracker.tracked_stracks, f_tracked_stracks)
                            detection_cycle_counter -= 1
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
                        fps = frame_idx / time_all
                        online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_idx, fps=fps)
                        rectangle(online_im,dense_region0[0],dense_region0[1],dense_region0[2]-dense_region0[0],dense_region0[3]-dense_region0[1],(0,140,255),label="DenseRegion",thickness=2)
                        
                        self.frame = online_im
                        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)


                        # img = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888)
                        # self.parent.videoLabel.setPixmap(QPixmap.fromImage(img))
                        self.frame_idx = frame_idx
                        self.fps = fps
                        self.update_video()
                        self.update_frame_idx()
                        self.update_fps()
                        # self.parent.framenumberLabel.setText(str(frame_idx))
                        # self.parent.fpsLabel.setText(str(fps))

                        # cv2.waitKey(int(1000 / frame_rate))
                        frame_idx += 1
                    else:
                        if self.enableTRT:
                            tk.tracker.release_memory()
                        self.parent.videoLabel.setPixmap(QPixmap("./finished.jpeg"))
                        self.parent.playing = False
                        self.parent.opened = False
                        self.parent.actionOpen.setEnabled(True)
                        self.parent.actionContinue.setEnabled(False)
                        self.parent.actionPause.setEnabled(False)
                        self.parent.actionClose.setEnabled(False)
                        self.parent.enableedgeChb.setEnabled(True)
                        self.parent.KSpb.setEnabled(True)
                        # save results
                        write_results(self.parent.output_file, results, 'mot')
                        break
                else:
                    continue
            else:
                if self.enableTRT:
                    tk.tracker.release_memory()
                self.parent.videoLabel.setPixmap(QPixmap("./finished.jpeg"))
                self.parent.playing = False
                self.parent.opened = False
                break

    def send_results(self, current_results):
        event = threading.Event()
        self.signals.result_signal.emit(event, current_results)
        event.wait()

    def update_video(self):
        event = threading.Event()
        self.signals.display_singal.emit(event)
        event.wait()

    def update_frame_idx(self):
        event = threading.Event()
        self.signals.idupdated_singal.emit(event)
        event.wait()

    def update_fps(self):
        event = threading.Event()
        self.signals.fpsupdated_signal.emit(event)
        event.wait()

class EdgeWorker(QThread):
    def __init__(self, parent):
        super().__init__()
        self.signals = Communicate()
        self.parent = parent
        # self.signals.display_singal.connect(parent.Display)
        # self.signals.result_signal.connect(self.parent.showResult)
        # self.signals.configure_signal.connect(parent.setConfigre)
        self.enableTRT = parent.enableTRT
        self.enableFP16 = parent.enableFP16
        self.enableTimeTrick = parent.enableTimeTrick
        self.enableSpaceTrick = parent.enableSpaceTrick
        self.divider_cycle = parent.divider_cycle
        self.detection_cycle = parent.detection_cycle
    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        print("Socket Created")
        
        socket.recv()
        socket.send(b'start')

        opt = opts().init()

        # init
        raw_shape_hw = [1080, 1920]

        init_msg = socket.recv()
        print(f"Received request: {init_msg}")
        socket.send(b"Got")
        frame_rate = float(socket.recv())
        print(f"Frame rate: {frame_rate}")
        socket.send(b"Got")
        max_frame_idx = float(socket.recv())
        print(f"Frame rate: {max_frame_idx}")
        socket.send(b"Got")
        raw_shape_hw[0] = int(socket.recv())
        print(f"Frame rate: {raw_shape_hw[0]}")
        socket.send(b"Got")
        raw_shape_hw[1] = int(socket.recv())
        print(f"Frame rate: {raw_shape_hw[1]}")
        socket.send(b"Got")
        

        # configure parameters
        socket.recv()
        socket.send_pyobj(self.enableTimeTrick)
        socket.recv()
        socket.send_pyobj(self.enableSpaceTrick)
        socket.recv()
        socket.send_pyobj(self.detection_cycle)
        socket.recv()
        socket.send_pyobj(self.divider_cycle)
        socket.recv()
        K = self.parent.K
        socket.send_pyobj(K)
        socket.recv()
        socket.send_pyobj(self.enableTRT)
        
        

        target_shape_hw = [608, 1088]

        if self.enableTRT:
            tk = TK_trt(self.enableFP16)
        else:
            tk = TK_basic()
        tk.init_kernel(frame_rate, raw_shape_hw, target_shape_hw, opt)

        # test bandwidth
        for i in range(K):
            frame = socket.recv()
            socket.send(b'')
            print(f"Received")
            
        # test first frame
        frame = socket.recv()
        socket.send(b'')
        print(f"Received")
        online_dets = []
        if frame != b'':
            img = np.array(Image.open(io.BytesIO(frame)))
            # Execute
            online_dets, timetick = tk.infer(img)
        else:
            tk.tracker.frame_id += 1
        socket.recv()
        socket.send_pyobj(online_dets)
        data_size = sys.getsizeof(online_dets) / 1024

        print(f"Result sent, size: {data_size} KB")
        time_all = 0
        while True:
            # time_st = time.time()
            frame = socket.recv()
            socket.send(b'')
            print(f"Received")
            online_dets = []
            if frame == b'f_turn':
                frame = socket.recv()
                online_im = np.array(Image.open(io.BytesIO(frame)))
                print(f"Result sent, size: {data_size} KB")
                self.frame = online_im
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

                img = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888)
                self.parent.videoLabel.setPixmap(QPixmap.fromImage(img))
                self.parent.framenumberLabel.setText(str(tk.tracker.frame_id))
                cv2.waitKey(int(1000 / frame_rate))
                socket.send(b'')
                fps = socket.recv_pyobj()
                socket.send(b'')
                self.parent.fpsLabel.setText(str(fps))
                continue
            if frame == b'test':
                frame = socket.recv()
                socket.send(b'')
                tk.tracker.frame_id += 1
            else:
                img = np.array(Image.open(io.BytesIO(frame)))
                # Execute
                online_dets, timetick = tk.infer(img)
            socket.recv()
            socket.send_pyobj(online_dets)
            data_size = sys.getsizeof(online_dets) / 1024
            frame = socket.recv()
            online_im = np.array(Image.open(io.BytesIO(frame)))
            
            print(f"Result sent, size: {data_size} KB")
            self.frame = online_im
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)


            img = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888)
            self.parent.videoLabel.setPixmap(QPixmap.fromImage(img))
            self.parent.framenumberLabel.setText(str(tk.tracker.frame_id))
            cv2.waitKey(int(1000 / frame_rate))
            socket.send(b'')
            fps = socket.recv_pyobj()
            socket.send(b'')
            self.parent.fpsLabel.setText(str(fps))

            

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.connect_actions()
        self.itemDict = {}
        self.itemWidgetDict = {}
        self.currentTO = []
        self.enableTimeTrick = False
        self.enableSpaceTrick = False
        self.divider_cycle = 0
        self.detection_cycle = 0
        self.enableTRT = False
        self.enableFP16 = False
        self.enableEdge = False
        self.K = 1
        self.input_file = "edge"
        self.opened = False
        self.playing = False

    @Slot(threading.Event)
    def Display(self, event):
        img = QImage(self.worker.frame.data, self.worker.frame.shape[1], self.worker.frame.shape[0], QImage.Format_RGB888)
        self.videoLabel.setPixmap(QPixmap.fromImage(img))
        event.set()

    @Slot(threading.Event)
    def UpdateId(self, event):
        self.framenumberLabel.setText(str(self.worker.frame_idx))
        event.set()

    @Slot(threading.Event)
    def UpdateFps(self, event):
        self.fpsLabel.setText(str(self.worker.fps))
        event.set()
        

    def exploreVideo(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(None,  
                                    "Open..",  
                                    str(os.getcwd()),
                                    "All Files (*);;Video Files (*.avi)")
        if fileName_choose == "":
            print("\nCancel Select")
            return

        print("\nFile name:")
        print(fileName_choose)
        print("File filter type: ",filetype)
        self.input_file = fileName_choose
        typedetail = fileName_choose.split('.')[-1]
        self.videotypeLabel.setText("." + typedetail + " file")
        self.locationurlLabel.setText(fileName_choose)
        videoname = fileName_choose.split('/')[-1].split('.')[0]
        self.videonameLabel.setText(videoname)
        self.trackingmethodLabel.setText("FairMOT")
        self.groupBox_4.setEnabled(True)
        self.actionOpen.setEnabled(False)
        self.actionContinue.setEnabled(True)
        self.actionPause.setEnabled(False)
        self.actionClose.setEnabled(True)
        self.enableedgeChb.setEnabled(False)
        self.KSpb.setEnabled(False)
        #explore video file function

    def exploreSaveLocation(self):
        direc_name = QFileDialog.getExistingDirectory(None,"Choose save location..",".")
        if direc_name == "":
            print("\nCancel Select")
            return
        print("\nSave location:")
        print(direc_name)
        self.save_direc_name = direc_name
        self.saveLocationLineEdit.setText(direc_name)

    def pauseVideo(self):
        self.playing = False
        self.actionPause.setEnabled(False)
        self.actionContinue.setEnabled(True)

    def continueVideo(self):
        if self.opened:
            self.playing = True
        else:
            self.playVideo()
            self.groupBox_4.setEnabled(False)
            self.playing = True
        self.actionPause.setEnabled(True)
        self.actionContinue.setEnabled(False)

    def closeVideo(self):
        self.opened = False
        self.actionOpen.setEnabled(True)
        self.enableedgeChb.setEnabled(True)
        self.actionPause.setEnabled(False)
        self.actionContinue.setEnabled(False)
        self.actionClose.setEnabled(False)
        self.groupBox_4.setEnabled(False)
        
    def txtShowingControl(self):
        if self.txtShowing:
            self.txtShowing = False
        else:
            self.txtShowing = True

    def connect_actions(self):
        self.actionOpen.triggered.connect(lambda:self.exploreVideo())
        self.actionPause.setEnabled(False)
        self.actionContinue.setEnabled(False)
        self.actionClose.setEnabled(False)
        self.actionPause.triggered.connect(lambda:self.pauseVideo())
        self.actionContinue.triggered.connect(lambda:self.continueVideo())
        self.actionClose.triggered.connect(lambda:self.closeVideo())
        self.groupBox_4.setEnabled(False)
        # configures
        self.enabletrtChb.stateChanged.connect(lambda:self.trtEnableChanged())
        self.enablefp16Chb.stateChanged.connect(lambda:self.trt16EnableChanged())
        self.timetrickChb.stateChanged.connect(lambda:self.timeTrickChanged())
        self.spacetrickChb.stateChanged.connect(lambda:self.spaceTrickChanged())
        self.detcycleSpb.valueChanged.connect(lambda:self.detCycleChanged())
        self.globaldetcycleSpb.valueChanged.connect(lambda:self.globalDetCycleChanged())
        self.KSpb.valueChanged.connect(lambda:self.KChanged())
        self.enableedgeChb.stateChanged.connect(lambda:self.edgeEnableChanged())

    def playVideo(self):
        # save_location = self.saveLocationLineEdit.text()
        save_location = ""
        if save_location == "":
            save_location = "."
        self.output_file = save_location + '/' + self.input_file.split('/')[-1] + '_' + str(int(time.time())) + ".txt"
        self.fo = open(self.output_file, 'w')
        print("Opened", self.output_file)
        self.videoLabel.setScaledContents(True)
        # configure
        
        if self.enableEdge:
            self.worker = EdgeWorker(self)
        else:
            print("create worker")
            self.worker = Worker(self)
        print("start worker")
        self.worker.start()

    def trtEnableChanged(self):
        if self.enabletrtChb.isChecked():
            self.enableTRT = True
        else:
            self.enableTRT = False

    def trt16EnableChanged(self):
        if self.enablefp16Chb.isChecked():
            self.enableFP16 = True
        else:
            self.enableFP16 = False

    def timeTrickChanged(self):
        if self.timetrickChb.isChecked():
            self.enableTimeTrick = True
            self.detcycleSpb.setMinimum(0)
            self.detcycleSpb.setMaximum(25)
            self.detcycleSpb.setValue(0)
        else:
            self.enableTimeTrick = False

    def spaceTrickChanged(self):
        if self.spacetrickChb.isChecked():
            self.enableSpaceTrick = True
            self.globaldetcycleSpb.setMinimum(0)
            self.globaldetcycleSpb.setMaximum(25)
            self.globaldetcycleSpb.setValue(0)
        else:
            self.enableSpaceTrick = False

    def detCycleChanged(self):
        self.detection_cycle = self.detcycleSpb.value()

    def globalDetCycleChanged(self):
        self.divider_cycle = self.globaldetcycleSpb.value()

    def KChanged(self):
        self.K = self.KSpb.value()

    def edgeEnableChanged(self):
        if self.enableedgeChb.isChecked():
            self.enableEdge = True
            self.groupBox_4.setEnabled(True)
            self.actionContinue.setEnabled(True)
            self.input_file = "edge"
            self.KSpb.setEnabled(True)
            self.KSpb.setMinimum(0)
            self.KSpb.setValue(1)
        else:
            self.enableEdge = False
            self.groupBox_4.setEnabled(False)
            self.actionContinue.setEnabled(False)
            self.KSpb.setEnabled(False)

    @Slot(threading.Event, list)
    def showResult(self, event, current_results):
        for trk in current_results:
            trk_txt = ','.join(list(map(str,trk)))
        for trk_id in self.currentTO:
            if trk_id not in current_results:
                self.currentTO.remove(trk_id)
                row = self.trackingobjectsList.row(self.itemDict[trk_id])
                self.trackingobjectsList.takeItem(row)
            self.fo.write(trk_txt + '\n')
        event.set()
        

if __name__ == "__main__":
    dirname = os.path.dirname(PySide2.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())