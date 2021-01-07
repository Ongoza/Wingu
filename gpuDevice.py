import os, sys, traceback
import time
import queue, threading
import logging
import random
import json
import yaml
import math
import numpy as np
import cv2
import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU

from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4

import videoCapture
# from server import log

class GpuDevice(threading.Thread):
    def __init__(self, id, device, configFileName, log):
        self.ready = False
        self.id = id
        self.log = log
        self.device = device
        try:
            with open(configFileName) as f:    
                self.config = yaml.load(f, Loader=yaml.FullLoader)        
            if device:
                with tf.device(self.device):
                    self.detector = YOLOv4(
                        input_shape=(self.config["img_size"], self.config["img_size"], 3), 
                        anchors=YOLOV4_ANCHORS, 
                        num_classes=80, 
                        training=False, 
                        yolo_max_boxes=self.config["yolo_max_boxes"], 
                        yolo_iou_threshold=self.config["yolo_iou_threshold"], 
                        yolo_score_threshold=self.config["yolo_score_threshold"]) 
                    self.detector.load_weights(self.config['detector_filename'])
            else:
                self.detector = YOLOv4(
                    input_shape=(self.config["img_size"], self.config["img_size"], 3), 
                    anchors=YOLOV4_ANCHORS, 
                    num_classes=80, 
                    training=False, 
                    yolo_max_boxes=self.config["yolo_max_boxes"], 
                    yolo_iou_threshold=self.config["yolo_iou_threshold"], 
                    yolo_score_threshold=self.config["yolo_score_threshold"]) 
                self.detector.load_weights(self.config['detector_filename'])
            self.cnt = 0
            self.frame = []
            self.cams = []
            self.img_size = self.config['img_size']
            self._stopevent = threading.Event()
            self.ready = True
            print("GPU created ok")
            threading.Thread.__init__(self)
            #t = threading.Thread(target=self._reader)
            #t.daemon = True
        except:
            print("Can not start GPU for " + id + " " + config)            
            # traceback.print_exception(*sys.exc_info()) 
            print("VideoStream err:", sys.exc_info())


    def startCam(self, camConfigFileName, iter):
        if iter < 100:
            print('try to start video: ', camConfigFileName, iter)
            iter += 1
            #id, url, borders, skipFrames, max_cosine_distance, nn_budget
            if self.ready:
                cam =  videoCapture.VideoCapture(camConfigFileName, self.config, self.log)
                if cam.id:
                    self.cams.append(cam)
                    print("Current num of caneras:", len(self.cams))
                    if(len(self.cams) > 0):
                        if self._stopevent.isSet(): 
                            self.start()
                else:
                   print("can not start Stream")
            else:                
                print("wait")
                time.sleep(1)
                self.startCam(camConfigFileName, iter)
        else:
            print("GPU is not ready for long time")

    def stopCam(self, id):
        print('stop video: ', id)
        self.cams[0].exit()
        time.sleep(0.1)
        del self.cams[0]
        if(len(self.cams)==0):
            print("Any cameras are not present!")
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
        self.log.debug("start GPU")
        while not self._stopevent.isSet():
            start = time.time()
            self.cnt += 1
            frames = []
            for cam in self.cams:
                if cam._stopevent.isSet():
                    self.log.debug("cam stopped: ", cam.id)
                    #delete cam from cams list
                    pass
                else:
                    frames.append(cam.read())
                    self.log.debug("cur_frame", cam.id, cam.cur_frame)
                    #if (tr): cv2.imwrite("video/39_2.jpg", frames[0])            
            if frames:
                # frame = frames[0]
                # print("frame rs = ", frames.shape)
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # convert numpy opencv to tensor
                frames_tf = tf.convert_to_tensor(frames, dtype=tf.float32, dtype_hint=None, name=None) / 255.0
                #frames_tf = tf.expand_dims(frames_tf, axis=0) / 255.0
                boxes, scores, classes, valid_detections = self.detector.predict(frames_tf)            
                #    obj_detec = non_max_suppression(obj_detec, self.conf_thres, self.nms_thres)
                for j in range(len(frames)):
                   self.cams[j].track(boxes[j], scores[j], classes[j], frames[j]) 
            else:
                self.log.info("Any available streams")
                self.kill()
                


if __name__ == "__main__":
    #tf.debugging.set_log_device_placement(True)
    log = logging.getLogger('app')
    log.setLevel(logging.DEBUG)
    f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(f)
    log.addHandler(ch)
    device = None
    log.debug(tf.config.experimental.list_physical_devices())
    # my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    # tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        log.debug("cuda is available " + str(len(gpus)))
        device = gpus[0].name
    else:
        print("cuda is not available")
        # device = tf.config.experimental.list_physical_devices('CPU')[0].name
    # print("device=", device)
    tr = False
    gpu = None
    try:
        gpu = GpuDevice("test", device, 'config/GPU_default.yaml', log)
        time.sleep(5)
        gpu.startCam('config/Stream_39.yaml', 0)
        time.sleep(5)
        gpu.startCam('config/Stream_43.yaml', 0)        
        tr = True
    except:
        log.debug(sys.exc_info())
    if tr:
        while True:
            time.sleep(3)
            log.debug("tik")
            try:
                log.debug("gpu.cams[0].outFrame "+ str(len(gpu.cams)))
                log.debug("frame "+ gpu.cams[0].id +" "+ str(len(gpu.cams[0].outFrame)))
                log.debug("frame "+ gpu.cams[0].id +" "+ str(gpu.cams[0].cur_frame))
                if gpu.cams[0].outFrame:
                    cv2.imshow('Avi_39', gpu.cams[0].outFrame)
                if gpu.cams[1].outFrame:
                    cv2.imshow('Avi_43', gpu.cams[1].outFrame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'): break
            except KeyboardInterrupt:
                log.debug("try to stop")
                gpu.kill()
                break
    cv2.destroyAllWindows()
    log.debug("Stoped - OK")
    