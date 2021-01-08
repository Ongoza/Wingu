import os, sys, traceback, time
import queue, threading

import logging

import tensorflow as tf
import yaml
import cv2

import gpuDevice

class Manager(threading.Thread):
    def __init__(self, log):
        threading.Thread.__init__(self)
        self.gpusActiveList = {}
        self.log = log
        self.gpusList = ["CPU"]  # Init for CPU  
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
               self.gpusList.append(gpu.name)
        self.log.debug("GPUs list: "+" ".join(self.gpusList))

    def startGpu(self, device_id, configFileName):
        # id 0 - CPU, 1 - the first GPU,  etc 
        self.log.debug("start gpu: "+ str(device_id) +" ".join([str(key) for key in self.gpusActiveList.keys()]))
        if device_id in self.gpusActiveList.keys():
            self.log.debug("GPU "+str(device_id)+" is already running")
        else:
            try:
                print("GPU config: ", self.gpusList[device_id], configFileName)
                #self.gpusActiveList[device_id] = gpuDevice.GpuDevice(device_id, self.gpusList[device_id], configFileName, self.log)
                self.gpusActiveList[0] = gpuDevice.GpuDevice(0, "CPU", configFileName, self.log)
                self.log.info("GPU "+ str(device_id) +" is running OK ")
                self.log.debug("added gpu "+ str(device_id) +" to list: "+" ".join([str(key) for key in self.gpusActiveList.keys()]))
            except:
                self.log.error("GPU "+str(device_id)+" is not running OK ")
                print(sys.exc_info())
            
    def stopGpu(self, id):
        if id in self.gpusActiveList:
            self.gpusActiveList[id].kill()
            del self.gpusActiveList[id]
            self.log.info("stoped gpu: "+ str(id))
        else:
            self.log.info("skip stoped gpu: "+ str(id))

    def kill(self):
        self.log.debug("try to kill gpus") 
        for id in self.gpusActiveList.keys():
            try:
                self.log.debug("try stop gpu: "+ str(id))
                self.gpusActiveList[id].kill()
            except:
                self.log.debug('Erorr stop gpu: '+ str(id))
        time.sleep(10)
        for id in self.gpusActiveList.keys():
            try:
                ready = self.gpusActiveList[id].id
            except:
                self.log.debug('Eroro stop gpu: '+ str(id))
                del self.gpusActiveList[id]
        if len(self.gpusActiveList) == 0:
            self.log.debug('No active gpus now...')
        else:
            self.log.error('Still active gpus: '+ " ".join([str(key) for key in self.gpusActiveList.keys()]))


if __name__ == "__main__":
    log = logging.getLogger('app')
    log.setLevel(logging.DEBUG)
    f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(f)
    log.addHandler(ch)
    manager = Manager(log)
    # id 0 - CPU, 1 - the first GPU,  etc
    manager.startGpu(0, 'config/Gpu_default.yaml')
    time.sleep(10)
    gpu = manager.gpusActiveList[0]
    gpu.startCam('config/Stream_39.yaml', 0)
    time.sleep(5)
    gpu.startCam('config/Stream_43.yaml', 0) 
    while True:
        time.sleep(3)
        log.debug("tik")
        try:
            log.debug("gpu.cams[0].outFrame "+ str(len(gpu.cams)))
            # print("frame "+ gpu.cams[0].id +" ", gpu.cams[0].outFrame)
            print("frame "+ gpu.cams[0].id +" ", gpu.cams[0].cur_frame_cnt)
            if gpu.cams[0].outFrame.any():
                cv2.imshow('Avi_39', gpu.cams[0].get_cur_frame())
            if gpu.cams[1].outFrame.any():
                cv2.imshow('Avi_43', gpu.cams[1].get_cur_frame())
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'): break
        except KeyboardInterrupt:
            log.debug("try to stop by hot keys")
            manager.kill()
            break
        except:
            log.error("try to stop by exception")
            manager.kill()
            break
    cv2.destroyAllWindows()
    log.debug("Stoped - OK")

   
   