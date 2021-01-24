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
import asyncio
from requests_futures import sessions
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU

from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4

import videoCapture
# from server import log
# from wingu_server import ws_send_data, save_statistic

class GpuDevice(threading.Thread):
    def __init__(self, id, device_name, configFile, log):
        print("GpuDevice start init")
        threading.Thread.__init__(self)
        self.ready = False
        self.id = id
        self.log = log
        self.cams = {}
        self.server_URL = "http://localhost:8080/update?"
        self.session = sessions.FuturesSession(max_workers=2)
        self.proceedTime = 0
        self.device = str(device_name)
        try:
            self.config = configFile 
            print("GpuDevice start init model")
            with tf.device(self.device):
                self.detector = YOLOv4(
                    input_shape=(self.config["img_size"], self.config["img_size"], 3), 
                    anchors=YOLOV4_ANCHORS, 
                    num_classes=80, 
                    training=False, 
                    yolo_max_boxes=self.config["yolo_max_boxes"], 
                    yolo_iou_threshold=self.config["yolo_iou_threshold"], 
                    yolo_score_threshold=self.config["yolo_score_threshold"])
                print("GpuDevice model loaded!")
                self.detector.load_weights(os.path.join('models', self.config['detector_filename']))
                print("GpuDevice weights loaded!")
            print("GpuDevice init GPU config", self.config)
            self.cnt = 0
            self.frame = []
            self.img_size = self.config['img_size']
            self._stopevent = threading.Event()
            self.ready = True
            # self.isRunning = False
            self.log.debug(device_name +" with name "+ str(self.id)+ " created ok id:"+ str(self.device))
            self.session.get(self.server_URL + "cmd=startGPU&name="+str(id))
            self.start()
        except:
            print("GpuDevice init Can not start GPU for " + str(self.id) + " ", self.device, self.config)            
            # traceback.print_exception(*sys.exc_info()) 
            print(sys.exc_info())
            self.kill()

    def run(self):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._run())
        loop.close()

    def getCamsList(self):
        # check if cam exists then update cams list and retur it
        return self.cams

    def ws_send_data(self, cmd):
        try:
            self.session.get(self.server_URL+'cmd='+cmd)
            #future.result()
            #print("ok")
        except:
             pass

    def startCam(self, camConfig, cam_id, iter, client=None):
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
                    cam = videoCapture.VideoCapture(camConfig, self.config, self.id, cam_id, self.log, client)
                    if cam is not None:
                        self.cams[cam_id] = cam
                        self.log.debug("GpuDevice "+str(self.device)+" Current num of cameras:" + str(len(self.cams)))
                    else:
                       self.log.info("GpuDevice "+str(self.id)+" can not start Stream")
                else:                
                    time.sleep(1)
                    print("try one more time")
                    self.startCam(camConfig, cam_id, iter, client)
            else:
                self.log.info("GpuDevice "+str(self.id)+" is not ready for start too long time!! cam="+str(cam_id))
        except:
            print(sys.exc_info())
            self.log.debug("GpuDevice "+str(self.id)+" exception on stream start cam="+str(cam_id))

    def stopCam(self, id):
        self.log.debug("GpuDevice "+str(self.id)+'stop video: '+str(id))
        try:
            self.cams[id].kill()
            del self.cams[id]
            # self.session.get(self.server_URL)
            # self.ws_send_data("delCam")
        except:
            self.log.debug("GpuDevice can not stop cam "+ id)

    def kill(self):
        try:
            self.log.debug("start to stop GPU "+ str(self.id))
            self._stopevent.set()
            cams = list(self.cams.keys())
            for cam in cams:
                print("try stop cam" + cam)
                self.stopCam(cam)
                # self.cams[cam].kill()
                print("cams num:", len(self.cams))
            time.sleep(4)
            self.log.info("GPU "+str(self.id)+" stopped")        
            self.killed = True
        except:
            print("can not stop some cams")

    async def _run(self):
        self.log.debug("GpuDevice starting "+str(self.id))
        while not self._stopevent.isSet():
            # print("GpuDevice tik")
            start = time.time()
            self.cnt += 1
            frames = []
            cams = []
            for cam in self.cams:
                if self.cams[cam]._stopevent.isSet():
                    # self.log.debug("GpuDevice cam stopped: " + cam)
                    self.cams[cam].kill()
                    del self.cams[cam]
                    #delete cam from cams list
                    pass
                else:
                    frames.append(self.cams[cam].read())
                    cams.append(self.cams[cam])
                    #print("cur_frame", cam.id, cam.cur_frame)
                    #if (tr): cv2.imwrite("video/39_2.jpg", frames[0])            
            # print("fr",len(frames), cams)
            if frames:
                try:
                    # frame = frames[0]
                    # print("frame rs = ", frames.shape)
                    # convert numpy opencv to tensor
                    frames_tf = tf.convert_to_tensor(frames, dtype=tf.float32, dtype_hint=None, name=None) / 255.0
                    # print("frames_tf type=", type(frames_tf))
                    with tf.device(self.device):
                        boxes, scores, classes, valid_detections = self.detector.predict(frames_tf)
                    self.proceedTime = time.time() - start
                    for j in range(len(frames)):
                       await cams[j].track(boxes[j], scores[j], classes[j], frames[j]) 
                except:
                    self.log.error("GpuDevice "+str(self.id)+" skip frame by exception")
                    print(sys.exc_info(), type(frames_tf))
            # else:
                # self.log.info("GpuDevice Any available streams in GPU "+str(self.id))
                # self.kill()
                


if __name__ == "__main__":
    #tf.debugging.set_log_device_placement(True)
    log = logging.getLogger('app')
    log.setLevel(logging.DEBUG)
    f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(f)
    log.addHandler(ch)
    devices = ["CPU"]
    print(tf.config.experimental.list_physical_devices())
    # my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    # tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU') 
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        print("cuda is available " + str(len(gpus)))
        devices.append(gpu.name)
        device_name = gpus[0].name
    else:
        print("cuda is not available")
        # device = tf.config.experimental.list_physical_devices('CPU')[0].name
    print("devices: " + " ".join(devices))
    print("device=", devices[0])
    device_id = 0    
    device_name = devices[0]
    tr = False
    gpu = None
    try:
        with open('config/Stream_file_39.yaml', encoding='utf-8') as f:    
            configStream = yaml.load(f, Loader=yaml.FullLoader)  
        with open(os.path.join('config', 'Gpu_device0.yaml'), encoding='utf-8') as f:    
            configGpu = yaml.load(f, Loader=yaml.FullLoader)        
        gpu = GpuDevice(device_id, device_name, configGpu, log)
        time.sleep(5)
        gpu.startCam(configStream, "file_39", 0)
        #time.sleep(5)
        #gpu.startCam('Stream_43', 0)        
        tr = True
    except:
        print("ecxept!!!!!")
        print(sys.exc_info())
        if gpu:
            print("try stop gpu from main")
            gpu.kill()
    if True:
        print("cams "+ str(len(gpu.cams)))
        while True:
            try:
                start = time.time()
                for i, cam in enumerate(gpu.cams):
                    # print("frame "+ gpu.cams[0].id +" ", gpu.cams[0].get_cur_frame())
                    # print("frame "+ cam.id +" ", cam.cur_frame_cnt)
                    if gpu.cams[cam].outFrame.any():
                        cv2.imshow(str(cam), gpu.cams[cam].get_cur_frame())
                    if gpu.cams[cam].proceedTime[0]:
                        print("fpsRead_"+str(i), 1.0/(gpu.cams[cam].proceedTime[0]))
                    if gpu.cams[cam].proceedTime[1]:
                        print("fpsTrack_"+str(i), 1.0/(gpu.cams[cam].proceedTime[1]))
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'): break
            except:
                print("stop by exception")
                print(sys.exc_info()) 
                break
    else:
        try:
            time.sleep(60)
        except:
            print(sys.exc_info())
            gpu.kill()
    print("gpu ", gpu)
    if gpu:
        print("try stop gpu from main")
        gpu.kill()
    cv2.destroyAllWindows()
    print("Stoped - OK")
    