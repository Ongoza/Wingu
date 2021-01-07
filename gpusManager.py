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
import yaml

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
        self.streamsList = []
        self.gpusActiveList = {}
        self.log = log
        self.encodersList = {}
        self.detectorsList = {}
        self.gpusList['0'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("GPU 0", self.gpusList['0'])
        # self.addModelsToGpu('0', self.gpusList['0'])        
        #self._stopevent = threading.Event()
        #self.killed = False

    def startGpu(self, id, config, device = None):
        self.log.debug("start gpu: "+ id +"  "+ str(self.gpusActiveList.keys()))
        if id in self.gpusActiveList.keys():
            self.log.info("GPU is already running")
        else:
            if device == None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.gpusActiveList[id] = gpuDevice.GpuDevice(id, device, config, self.log)
            self.log.info("GPU is running OK")
            self.log.debug("start gpu: "+ id +"  " + str(self.gpusActiveList.keys()))
            

    def stopGpu(self, id):
        self.log.info("stop gpu: "+ id)
        self.gpusActiveList[id].kill()
        del self.gpusActiveList[id]

    def startStream(self, gpuId, streamConfigFile):
        print(gpuId, self.streamsList, streamConfig)
        if gpuId in self.gpusActiveList.keys():
            if streamConfig['id'] in self.streamsList:
                self.log.info("This stream already exist in strems list ")
            else:
                #self.gpusActiveList[gpuId].startCam(streamConfig)
                self.streamsList.append(streamConfig['id'])
        else:
            self.log.info("This GPU is not starting")
        
    def stopStream(self, gpuId, id):
        if not self.streamsList[gpuId]:
            print("GPU does not exist")
        else:
            if id in self.streamsList[gpuId]:
                self.gpusActiveList[gpuId].stopCam(id)
                del self.streamsList[gpuId][id]
                print("Strean is stopped")
            else:                
                print("Strean does not exist in the strems list on this GPU")

    def kill(self):
        self.log.debug("try to kill gpus")
        gpus = self.gpusActiveList.keys()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager = Manager(log)
    with open('config/Gpu_default.yaml') as f:    
        defaultConfig = yaml.load(f, Loader=yaml.FullLoader)
    #with open('config/Stream_default.yaml') as f:    
    #    camConfig = yaml.load(f, Loader=yaml.FullLoader)

    manager.startGpu('0', defaultConfig, device)
    manager.startStream('0', 'config/Stream_default.yaml', defaultConfig, log)

    #with open('config/defaultConfig.yaml', 'w') as f:    
    #    data = yaml.dump(defaultConfig, f)
    #gpu.startCam("video/39.avi", borders)
    #camManager.start()
    time.sleep(10)
    #camManager.kill()
    #camManager.join()