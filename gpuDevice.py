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
    def __init__(self, device, configFileName, log):
        self.ready = False
        self.id = device
        self.log = log
        self.proceedTime = 0
        self.device = device
        try:
            with open(os.path.join('config', configFileName+'.yaml')) as f:    
                self.config = yaml.load(f, Loader=yaml.FullLoader)        
            if device == "CPU":
                self.detector = YOLOv4(
                    input_shape=(self.config["img_size"], self.config["img_size"], 3), 
                    anchors=YOLOV4_ANCHORS, 
                    num_classes=80, 
                    training=False, 
                    yolo_max_boxes=self.config["yolo_max_boxes"],
                    # This means that all predicted that have a detection probability less than VALUE will be removed.
                    yolo_iou_threshold=self.config["yolo_iou_threshold"], 
                    yolo_score_threshold=self.config["yolo_score_threshold"]) 
                self.detector.load_weights(os.path.join('models', self.config['detector_filename']))
            else:
                with tf.device(self.device):
                    self.detector = YOLOv4(
                        input_shape=(self.config["img_size"], self.config["img_size"], 3), 
                        anchors=YOLOV4_ANCHORS, 
                        num_classes=80, 
                        training=False, 
                        yolo_max_boxes=self.config["yolo_max_boxes"], 
                        yolo_iou_threshold=self.config["yolo_iou_threshold"], 
                        yolo_score_threshold=self.config["yolo_score_threshold"]) 
                    self.detector.load_weights(os.path.join('models', self.config['detector_filename']))
            # print("GPU config", self.config)
            self.cnt = 0
            self.id = self.config['id']
            self.frame = []
            self.cams = []
            self.img_size = self.config['img_size']
            self._stopevent = threading.Event()
            self.ready = True
            print("GPU ", str(self.id), " created ok")
            threading.Thread.__init__(self)
            #t = threading.Thread(target=self._reader)
            #t.daemon = True
        except:
            print("Can not start GPU for " + str(self.id) + " ", self.config)            
            # traceback.print_exception(*sys.exc_info()) 
            print("GPU err:", sys.exc_info())
            self.kill()

    def startCam(self, camConfigFileName, iter):
        if iter < 100:
            print('try to start video: ', camConfigFileName, iter)
            iter += 1
            #id, url, borders, skipFrames, max_cosine_distance, nn_budget
            if self.ready:
                cam =  videoCapture.VideoCapture(camConfigFileName, self.config, self.device, self.log)
                if cam.id:
                    self.cams.append(cam)
                    self.log.debug("GPU "+str(self.id)+" Current num of caneras:" + str(len(self.cams)))
                    if(len(self.cams) == 1 ):
                        self.start()
                else:
                   self.log.info("GPU "+str(self.id)+" can not start Stream")
            else:                
                time.sleep(1)
                self.startCam(camConfigFileName, iter)
        else:
            self.log.info("GPU "+str(self.id)+" is not ready for start too long time!!")

    def stopCam(self, id):
        self.log.debug("GPU "+str(self.id)+'stop video: '+str(id))
        self.cams[0].exit()
        time.sleep(0.1)
        del self.cams[0]
        if(len(self.cams)==0):
            print("Any cameras are not present!")
            time.sleep(0.5)
            self.kill()

    def kill(self):
        try:
            self.log.debug("start to stop GPU "+ str(self.id))
            self._stopevent.set()
            for i in range(len(self.cams)):
                print("try stop cam" + self.cams[i].id)
                self.cams[i].kill()
                print("cams num:",len(self.cams))
            self.log.info("GPU "+str(self.id)+" stopped")        
            self.killed = True
        except:
            print("can not stop some cams")

    def run(self):
        self.log.debug("start GPU "+str(self.id))
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
                    #print("cur_frame", cam.id, cam.cur_frame)
                    #if (tr): cv2.imwrite("video/39_2.jpg", frames[0])            
            # print("fr",len(frames))
            if frames:
                try:
                    # frame = frames[0]
                    # print("frame rs = ", frames.shape)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # convert numpy opencv to tensor
                    frames_tf = tf.convert_to_tensor(frames, dtype=tf.float32, dtype_hint=None, name=None) / 255.0
                    # print("frames_tf type=", type(frames_tf))
                    #frames_tf = tf.expand_dims(frames_tf, axis=0) / 255.0
                    boxes, scores, classes, valid_detections = self.detector.predict(frames_tf)            
                    #    obj_detec = non_max_suppression(obj_detec, self.conf_thres, self.nms_thres)
                    for j in range(len(frames)):
                       self.cams[j].track(boxes[j], scores[j], classes[j], frames[j]) 
                    self.proceedTime = time.time() - start
                except:
                    self.log.error("GPU "+str(self.id)+" skip frame by exception")
                    print(sys.exc_info(), type(frames_tf))
            else:
                self.log.info("Any available streams in GPU "+str(self.id))
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
    device = "CPU"
    devices = ["CPU"]
    log.debug(tf.config.experimental.list_physical_devices())
    # my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    # tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU') 
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        log.debug("cuda is available " + str(len(gpus)))
        devices.append(gpu.name)
        device = gpus[0].name
    else:
        print("cuda is not available")
        # device = tf.config.experimental.list_physical_devices('CPU')[0].name
    print("devices: " + " ".join(devices))
    print("device=", device)
    tr = False
    gpu = None
    try:
        gpu = GpuDevice(device, 'GPU_0', log)
        time.sleep(5)
        gpu.startCam('Stream_39', 0)
        #time.sleep(5)
        #gpu.startCam('Stream_43', 0)        
        tr = True
    except:
        print("ecxept!!!!!")
        log.debug(sys.exc_info())
        if gpu:
            print("try stop gpu from main")
            gpu.kill()
    if False:
        while True:
            time.sleep(3)
            log.debug("tik")
            try:
                log.debug("gpu.cams[0].outFrame "+ str(len(gpu.cams)))
                for cam in gpu.cams:
                    # print("frame "+ gpu.cams[0].id +" ", gpu.cams[0].get_cur_frame())
                    # print("frame "+ cam.id +" ", cam.cur_frame_cnt)
                    if gpu.cams[0].outFrame.any():
                        cv2.imshow(str(cam.id), cam.get_cur_frame())
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'): break
            except KeyboardInterrupt:
                log.debug("try to stop from loop main")
                gpu.kill()
                break
    else:
        try:
            time.sleep(60)
        except:
            gpu.kill()
    print("gpu ", gpu)
    if gpu:
        print("try stop gpu from main")
        gpu.kill()
    cv2.destroyAllWindows()
    log.debug("Stoped - OK")
    