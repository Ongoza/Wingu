import os, sys, traceback
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
# XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda" python ml_code.py
# TF_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda" python ml_code.py

import time
import queue, threading
import logging
import random
import json
import yaml
import math
import numpy as np
import cv2
# import gpu_trt_tool as trt_det
# import tensorflow as tf
import tensorrt as trt
import pycuda.driver as cuda
import asyncio
from requests_futures import sessions
import deep_sort.generate_detections_onnx as gdet

#from yolov3_tf2.models import YoloV3

#from tf2_yolov4.anchors import YOLOV4_ANCHORS
#from tf2_yolov4.model import YOLOv4

from videoCapture import *
from detector_trt import YoloDetector 
#import tensorflow as tf
# from server import log
# from wingu_server import ws_send_data, save_statistic

class GpuDevice(threading.Thread):
    def __init__(self, id, device_name, configFile, log):
        print("GpuDevice start init")
        threading.Thread.__init__(self)
        self.ready = False
        self.id = id
        self.batch_size = 2
        self.log = log
        self.cams = {}
        self.max_batch_size = 4
        self.config_mot = {'engine_path':'models/yolov4_-1_3_416_416_dynamic.engine','max_batch_size':self.max_batch_size}
        # with open('config/mot.json') as config_file:
        #     config_mot = json.load(config_file, cls=ConfigDecoder)
        #     self.config_mot = config['mot']
        self.detector = None
        self.server_URL = "http://localhost:8080/update?"
        self.session = sessions.FuturesSession(max_workers=2)
        self.proceedTime = 0
        self.device = str(device_name)
        try:
            self.config = configFile 
            self.cnt = 0
            self.frame = []
            self.img_size = self.config['img_size']
            self._stopevent = threading.Event()
            self.ready = True
            # self.isRunning = False
            self.log.debug(device_name +" with name "+ str(self.id)+ " created ok id:"+ str(self.device))
            self.session.get(self.server_URL + "cmd=GpuStart&name="+str(id)+"&status=OK")
            self.start()
        except:
            print("GpuDevice init Can not start GPU for " + str(self.id) + " ", self.device, self.config)            
            # traceback.print_exception(*sys.exc_info()) 
            self.session.get(self.server_URL + "cmd=GpuStart&name="+str(id)+"&status=error&module=Gpu")
            print(sys.exc_info())
            self.kill()

    def getCamsList(self):
        # check if cam exists then update cams list and retur it
        return self.cams

    def startCam(self, camConfig, cam_id, iter):
        try:
            print("GpuDevice iter", iter)
            if iter < 100:
                print('GpuDevice try to start video: ', camConfig, cam_id)
                iter += 1
                if self.ready:
                    #if(len(self.cams) == 1 ):
                    #    self.start()
                    #    time.sleep(3)
                    print("GpuDevice Try start videoCapture obj")
                    cam = VideoCapture(camConfig, self.config, self.id, cam_id, self.log)
                    if cam is not None:
                        self.cams[cam_id] = cam
                        self.log.debug("GpuDevice "+str(self.device)+" Current num of cameras:" + str(len(self.cams)))
                    else:
                       self.log.info("GpuDevice "+str(self.id)+" can not start Stream")
                else:                
                    time.sleep(1)
                    print("try one more time")
                    self.startCam(camConfig, cam_id, iter)
            else:
                self.log.info("GpuDevice "+str(self.id)+" is not ready for start too long time!! cam="+str(cam_id))
        except:
            print(sys.exc_info())
            self.log.debug("GpuDevice "+str(self.id)+" exception on stream start cam="+str(cam_id))

    def send_data(self, data):
        print("send data", data)
        try:
            self.session.get(self.server_URL+data) 
        except:
            print("GpuDevice error send data")

    def stopCam(self, cam_id):
        self.log.debug("GpuDevice "+str(self.id)+' stop video: '+str(cam_id))
        try:
            self.cams[cam_id].kill()
            del self.cams[cam_id]
        except:
            self.log.debug("GpuDevice can not stop cam "+ id)
            self.send_data("cmd=stopStream&name=" + cam_id + "&status=error&module=Gpu")

    def kill(self):
        try:
            self.log.debug("start to stop GPU "+ str(self.id))
            self._stopevent.set()
            if hasattr(self, 'engine'):                
                del self.engine
            if hasattr(self, 'ctx'):
                self.ctx.pop()
                del self.ctx
                print("context popped!!!")
            if hasattr(self, 'context'): del self.context
            for cam in list(self.cams):
                print("try stop cam" + cam)
                self.stopCam(cam)
                # self.cams[cam].kill()
                print("cams num:", len(self.cams))
            self.log.info("GPU "+str(self.id)+" stopped")        
            self.killed = True
            self.session.get(self.server_URL + "cmd=GpuStop&name="+str(id)+"&status=OK")
        except:
            print("can not stop some cams")
        finally:
            if hasattr(self, 'ctx'):
                self.ctx.pop()

    def removeCam(self, cam_id):
        print("GpuDevice removeCam", cam_id)
        if cam_id in self.cams:
            del self.cams[cam_id]

    def run(self):
        # init cuda
        cuda.init()
        dev = cuda.Device(0)
        self.ctx = dev.make_context()
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.frame_count = 0
        self.detector = YoloDetector(self.img_size, self.config_mot, self.TRT_LOGGER)

        self.log.debug("GpuDevice starting "+str(self.id))
        while not self._stopevent.isSet():
            # print("GpuDevice tik")
            start = time.perf_counter()
            self.cnt += 1
            frames = []
            features = []
            cams = []
            for cam in list(self.cams):
                if self.cams[cam]._stopevent.isSet():
                    self.cams[cam].kill()
                    del self.cams[cam]
                    print("del camera ", cam)
                    #delete cam from cams list
                    pass
                else:
                    try:
                        frame = self.cams[cam].read()
                        if frame is not None:
                            frames.append(frame)
                            cams.append(self.cams[cam])
                        else:
                            print("skip data for cam:", cam)
                    except:
                        print("error frame!!!!!!!!!!!!!!!!!!!!!")
                        print("frame", type(frame), frame.shape, frame)
                        print(sys.exc_info())
            #start2 = time.perf_counter()
            #print("read time=",len(frames), start2 - start)
            bbxs = 0
            if frames:
                self.detector.detect_async(frames)
                boxes = self.detector.postprocess()
                try:
                    start2 = time.perf_counter()
                    print("detect", time.perf_counter()- start2, len(boxes))
                    print("boxes", len(boxes), time.perf_counter()- start2)
                    for j in range(len(cams)):
                        cams[j].track(boxes[j], frames[j], features)
                except:
                    self.log.error("GpuDevice "+str(self.id)+" skip frame by exception")
                    print(sys.exc_info(), type(frames_tf))
            else:
                time.sleep(1)
                print("Any active streams")
            # self.log.info("GpuDevice Any available streams in GPU "+str(self.id))
            # self.kill()
            #print("TRT frame{} cams({}) ims,size({}) bbxs({}) FPS={}".format(self.cnt, len(cams), self.img_size, bbxs, int(1/(time.time() - start))))
        #self.kill()


if __name__ == "__main__":
    log = logging.getLogger('app')
    log.setLevel(logging.DEBUG)
    f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(f)
    log.addHandler(ch)
    device_id = 0    
    device_name = "GPU:0"
    with open('config/Stream_file_39.yaml', encoding='utf-8') as f:    
        configStream = yaml.load(f, Loader=yaml.FullLoader)  
    with open(os.path.join('config', 'Gpu_device0.yaml'), encoding='utf-8') as f:    
        configGpu = yaml.load(f, Loader=yaml.FullLoader)        
    gpu = GpuDevice(device_id, device_name, configGpu, log)
    gpu.startCam(configStream, "file_39", 0)

