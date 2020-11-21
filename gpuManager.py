import os, sys, traceback
import time
import os, sys, traceback
import time
import queue, threading
# from multiprocessing import Pool, current_process, Queue
# https://stackoverflow.com/questions/53422761/distributing-jobs-evenly-across-multiple-gpus-with-multiprocessing-pool
import logging
import random
import json
import math

import numpy as np
import cv2
import tensorflow as tf
import torch

import gpuDevice
from models.models import Darknet
#from models.utils import non_max_suppression
#from deep_sort import nn_matching
#from deep_sort import preprocessing
import deep_sort.generate_detections as gdet
from deep_sort.detection import Detection

class Manager(threading.Thread):
    def __init__(self, log):
        threading.Thread.__init__(self)
        self.gpusList = {}
        self.streamsList = {}
        self.gpusActiveList = {}
        self.log = log
        self.encodersList = {}
        self.detectorsList = {}
        self.defaultConfig = {
            'model_name': "yolov3-tiny", 
            'model_def': "models/yolov3-tiny.cfg",
            'weights_path':  "models/yolov3-tiny.weights",
            'model_filename': 'models/mars-small128.pb',
            'saveVideoFlag': False,
            'displayVideoFlag': False,
            'conf_thres': 0.5, 
            'nms_thres': 0.4, 
            'img_size': 416, 
            'body_res':(256, 128), 
            'body_min_w': 64, 
            'threshold': 0.5, 
            'nms_max_overlap': 0.9, 
            'max_cosine_distance': 0.2
            }
        self.cntCudaDevices = 0
        self.gpusList['0'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.addModelsToGpu('0', self.gpusList['0'])
        
        

        #self._stopevent = threading.Event()
        #self.killed = False

    def startGpu(self, id, device):
        self.log.debug("start gpu: "+ id)
        self.gpusActiveList[id] = gpuDevice.GpuDevice(id, device, self.log, self.defaultConfig)
        self.gpusctiveList[id].start()
        #print(self.camActiveObjList)

    def stopGpu(self, id):
        self.log.debug("stop gpu: "+ id)
        self.gpusActiveList[id].kill()
        self.gpusActiveList[id].join()
        del self.gpusActiveList[id]


    def kill(self):
        self.log.debug("try to kill gpus")
        gpus = self.gpusctiveList.keys()
        for id in gpus:
            try:
                self.log.debug("try stop gpu: "+ id)
                self.gpusActiveList[id].kill()
                self.gpusActiveList[id].join()
            except:
                self.log.debug('Eroro stop gpu: '+ id)
        self.log.debug('No active gpus now...')

if __name__ == "__main__":
    log = logging.getLogger('app')
    log.setLevel(logging.DEBUG)
    f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(f)
    log.addHandler(ch)

    camManager = Manager(log)
    #camManager.start()
    time.sleep(10)
    #camManager.kill()
    #camManager.join()