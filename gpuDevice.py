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

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
from deep_sort import nn_matching
from deep_sort import preprocessing
import deep_sort.generate_detections as gdet
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
# from shapely.geometry import LineString, Point
from models.models import Darknet
from models.utils import non_max_suppression

import videoCapture
# from server import log

class GpuDevice(threading.Thread):
    def __init__(self, id, device, config, log):        
        self.id = id
        self.log = log
        self.frame = []
        self.device = device
        self.config = config
        self.cnt = 0
        self.cams = []
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
        print("created ok")
        threading.Thread.__init__(self)
        #t = threading.Thread(target=self._reader)
        #t.daemon = True
    
    def startCam(self, camConfig):
        print('start video: ', camConfig['id'])
        #id, url, borders, skipFrames, max_cosine_distance, nn_budget
        cam =  videoCapture.VideoCapture(camConfig, self.config)
        self.cams.append(cam)
        print(self.cams[0].id)
        if(len(self.cams)==1):
            self.start()

    def stopCam(self, id):
        print('stop video: ', id)
        self.cams[0].exit()
        time.sleep(0.1)
        del self.cams[0]
        if(len(self.cams)==0):
            time.sleep(0.5)
            self.kill()
    
    #def drawBorderLine(self, a, b):
    #    length = 40
    #    vX0 = b[0] - a[0]; vY0 = b[1] - a[1]
    #    mag = math.sqrt(vX0*vX0 + vY0*vY0)
    #    vX = vX0 / mag; vY = vY0 / mag
    #    temp = vX; vX = -vY; vY = temp
    #    z0 = (int(a[0]+vX0/2), int(a[1]+vY0/2))
    #    z1 = (int(a[0]+vX0/2 - vX * length), int(a[1] +vY0/2- vY * length))
    #    cv2.line(frame, a, b, (255, 255, 0), 2)
    #    cv2.arrowedLine(frame, z0, z1, (0, 255, 0), 2)
    #    cv2.putText(frame, "Out", z1, 0, 1, (0, 255, 0), 1)

    def kill(self):
        self.log.debug("kill cameras "+ str(len(self.cams)))
        self._stopevent.set()
        for cam in self.cams:
            cam.exit()
        print("kill ok")        
        self.killed = True

    def run(self):
        print("start camera")
        cnt_people_in = []
        tr = True 
        while not self._stopevent.isSet():
            start = time.time()
            self.cnt += 1
            frames = []
            for cam in self.cams:
                if self.cams[0]._stopevent.isSet():
                    #delete cam from cams list
                    pass
                else:
                    frames.append(cam.read())
                    if (tr): cv2.imwrite("video/39_2.jpg", frames[0])            
            frame = frames[0]
            frame_cuda = transforms.ToTensor()(frame).unsqueeze(0)
            with torch.no_grad():    
                obj_detec = self.detector(frame_cuda)
                obj_detec = non_max_suppression(obj_detec, self.conf_thres, self.nms_thres)
            boxs = []
            confs = []
            for item in obj_detec:
                if item is not None:
                    i = 0
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in item: #classes[int(cls_pred)]
                        wb = y2-y1
                        if((cls_pred == 0) and (wb < self.max_hum_w) and (wb > self.body_min_w)):
                                boxs.append([y1, x1, y2, x2])
            if(len(boxs)):
                t_start2 = time.time()
                features = self.encoder(frame, boxs)
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
                boxes = np.array([d.tlwh for d in detections]) # w and h replace by x2 y2
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
                detections = [detections[i] for i in indices]
                for i, cam in enumerate(self.cams):
                    cam.track(detections, frames[i])
                    #if (cam.display_video_flag):
                        #print(type(cam.outFrame))
                        #if cam.outFrame.any():
                            #cv2.imshow("preview_"+str(i), cam.outFrame)
                    #pass
                    

if __name__ == "__main__":
    log = logging.getLogger('app')
    log.setLevel(logging.DEBUG)
    f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(f)
    log.addHandler(ch)
    with open('config/GPU_default.yaml') as f:    
        defaultConfig = yaml.load(f, Loader=yaml.FullLoader)
    with open('config/Stream_default.yaml') as f:    
        camConfig = yaml.load(f, Loader=yaml.FullLoader)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu = GpuDevice("test", device, defaultConfig, log)
        gpu.startCam(camConfig)
        time.sleep(10)
        while True:
            #for i, cam in enumerate(gpu.cams):
            #    if (cam.display_video_flag):
            #        print(type(cam.outframe))
            #        if cam.outframe.any():
            #            cv2.imshow("preview_"+str(i), cam.outFrame)
            
            key = cv2.waitKey(100)
            if key & 0xFF == ord('q'):
                print("key=",key)
                break
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
        print("stop")
    print("try stop app")
    gpu.kill()
    print("try 2 stop app")
    cv2.destroyAllWindows()
    print("stop app")
    