import os, sys, traceback
import time
import queue, threading
import logging
import random
import json
import math

import numpy as np
import cv2
import tensorflow as tf
import torch
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
from deep_sort import nn_matching
from deep_sort import preprocessing
import deep_sort.generate_detections as gdet
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
# from shapely.geometry import LineString, Point
from models.models import Darknet
from models.utils import non_max_suppression

from videoCapture import VideoCapture
# from server import log

class GpuDevice(threading.Thread):
    def __init__(self, id, device, log, config):
        threading.Thread.__init__(self)
        self.id = id
        self.log = log
        self.device = device
        self.config = config
        self.cnt = 0
        self.streams = {}
        self.borders = {}
        self.img_size = config['img_size']
        self.nms_max_overlap = config['nms_max_overlap']
        self.body_min_w = config['body_min_w']

        self.conf_thres = config['conf_thres'] 
        self.nms_thres = config['nms_thres']
        self.max_hum_w = int(self.img_size/2)
        self.detector = Darknet(self.config['model_def'], img_size=self.config['img_size']).to(self.device)
        self.detector.load_darknet_weights(self.config['weights_path'])
        self.encoder = gdet.create_box_encoder(self.config['model_filename'], batch_size=self.config['batch_size'])
        self._stopevent = threading.Event()
        self.killed = False
    
    def startStream(self, id, url, borders):
        print('start video: ', url)
        #id, url, borders, skipFrames, max_cosine_distance, nn_budget
        self.streams[id] = VideoCapture(id, url, borders, self.config)
        if(len(self.streams.keys())==1):
            self.start()

    def stopStream(self, id, url):
        print('stop video: ', self.url)
        self.streams[id].exit()
        del self.streams[id]
        if(len(self.streams.keys())==0):
            self.kill()
    
    def drawBorderLine(self, a, b):
        length = 40
        vX0 = b[0] - a[0]; vY0 = b[1] - a[1]
        mag = math.sqrt(vX0*vX0 + vY0*vY0)
        vX = vX0 / mag; vY = vY0 / mag
        temp = vX; vX = -vY; vY = temp
        z0 = (int(a[0]+vX0/2), int(a[1]+vY0/2))
        z1 = (int(a[0]+vX0/2 - vX * length), int(a[1] +vY0/2- vY * length))
        cv2.line(frame, a, b, (255, 255, 0), 2)
        cv2.arrowedLine(frame, z0, z1, (0, 255, 0), 2)
        cv2.putText(frame, "Out", z1, 0, 1, (0, 255, 0), 1)

    def kill(self):
        self.log.debug("kill camera")
        for cam in self.streams:
            self.streams[cam].exit()
        self._stopevent.set()
        self.killed = True

    def run(self):
        print("start camera")
        cnt_people_in = []
        while not self._stopevent.isSet():
            frames = []
            for cam in self.streams:
                frame = self.streams[cam].read()
                cv2.imshow("preview", frame)
                frames.append(frame)
            start = time.time()
            self.cnt += 1
            #print("frmae",len(frames))
            # frame = cv2.resize(frame, show_fh_fw)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame = cv2.copyMakeBorder(frames[0], 0, 400, 0, 0, cv2.BORDER_CONSTANT)
            frame = frames[0]
            # input picture to Tensor        
            # frame_cuda = torch.from_numpy(frame).float().to(device) 
            frame_cuda = transforms.ToTensor()(frame).unsqueeze(0)
            # other way
            with torch.no_grad():    
                # input_imgs = input_imgs.unsqueeze(0).permute(0, 3, 1, 2)/255.0#.permute(2, 0, 1)
                obj_detec = self.detector(frame_cuda)
                # print(obj_detec.shape)
                obj_detec = non_max_suppression(obj_detec, self.conf_thres, self.nms_thres)
            if not obj_detec[0] is None:
                #print("torch=",len(obj_detec[0]))
                pass
            # print(boxes)
            boxs = []
            confs = []
            for item in obj_detec:
                if item is not None:
                    # print("item ", item)
                    i = 0
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in item: #classes[int(cls_pred)]
                        wb = y2-y1
                        if((cls_pred == 0) and (wb < self.max_hum_w) and (wb > self.body_min_w)):
                                # boxs.append([int(y1*ratio_h_w[1]), int(x1*ratio_h_w[0]), int(y2*ratio_h_w[1]), int(x2*ratio_h_w[0])])
                                # boxs.append([int(y1), int(x1), int(y2), int(x2)])
                                boxs.append([y1, x1, y2, x2])
                                # print(conf, cls_conf)
                                # confs.append(float(conf))
                                # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                                # person_photo = frame[y1:y2, x1:x2]
            # print("box=",boxs)
            # frame_tf = tf.convert_to_tensor(frame, dtype=tf.float32)
            if(len(boxs)):
                t_start2 = time.time()
                features = self.encoder(frame, boxs)
                start2 = t_start2-start
                # print("emd=",time.time() - t_start2)
                # emd= 0.004137516021728516
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
                # detections = [Detection(bbox, conf, feature) for bbox, conf, feature in zip(boxs, confs, features)]
                start21 = time.time()-start
                # print(detections)
                boxes = np.array([d.tlwh for d in detections]) # w and h replace by x2 y2
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
                detections = [detections[i] for i in indices]
                # print(detections)
                for cam in self.streams:
                    #self.streams[cam].track(detections, frame)
                    pass
                    

if __name__ == "__main__":
    log = logging.getLogger('app')
    log.setLevel(logging.DEBUG)
    f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(f)
    log.addHandler(ch)
    defaultConfig = {
        'model_name': "yolov3-tiny", 
        'model_def': "models/yolov3-tiny.cfg",
        'weights_path':  "models/yolov3-tiny.weights",
        'model_filename': 'models/mars-small128.pb',
        'save_video_flag': False,
        'display_video_flag': True,
        'skip_frames': 4,
        'conf_thres': 0.5, 
        'nms_thres': 0.4, 
        'img_size': 416, 
        'body_res':(256, 128), 
        'body_min_w': 64, 
        'threshold': 0.5, 
        'nms_max_overlap': 0.9, 
        'max_cosine_distance': 0.2,
        'batch_size':1,
        'img_size_start': (1600,1200),
        'path_track': 20,
        'save_video_res':(720, 540),
        'nn_budget':None,
        'frame_scale':3.84615384615, # 1600/416
        }
    frame_scale = defaultConfig['frame_scale']
    borders = {'border1':[[int(0/frame_scale), int(400/frame_scale)], [int(1200/frame_scale), int(400/frame_scale)]]}


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = GpuDevice("test", device, log, defaultConfig)
    gpu.startStream('test_1', "video/39.avi", borders)
    while True:
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    gpu.kill()



