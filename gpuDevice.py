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
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU

from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
from deep_sort import nn_matching
from deep_sort import preprocessing
import deep_sort.generate_detections as gdet
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

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
        self.body_min_w = config['body_min_w']
        self.max_hum_w = int(self.img_size/2)
        self.detector = YOLOv4(
            input_shape=(config["img_size"], config["img_size"], 3), 
            anchors=YOLOV4_ANCHORS, 
            num_classes=80, 
            training=False, 
            yolo_max_boxes=config["yolo_max_boxes"], 
            yolo_iou_threshold=config["yolo_iou_threshold"], 
            yolo_score_threshold=config["yolo_score_threshold"]) 
        self.detector.load_weights(self.config['detector_filename'])
        self.encoder = gdet.create_box_encoder(self.config['encoder_filename'], batch_size=self.config['batch_size'])
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
            print("frame rs = ", frame.shape)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # convert numpy opencv to tensor
            frame_tz = tf.convert_to_tensor(frame, dtype=tf.float32, dtype_hint=None, name=None)
            frame_tz = tf.expand_dims(frame_tz, axis=0) / 255.0
            boxes, scores, classes, valid_detections = detector.predict(frame_tz)            
            #    obj_detec = non_max_suppression(obj_detec, self.conf_thres, self.nms_thres)
            boxs = []
            confs = []
            for i in range(len(boxes[0])): 
                if scores[0][i] > 0:
                    if classes[0][i] == 0:
                        boxs.append((np.array(boxes[0][i])*self.img_size))
                        confs.append(score[0][i])
            if(len(boxs)):
                t_start2 = time.time()
                features = self.encoder(frame, boxs)
                detections = [Detection(bbox, conf, feature) for bbox, conf, feature in zip(boxs, confs, features)] 
                for i, cam in enumerate(self.cams):
                    cam.track(detections, frames[i])
                    

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
    