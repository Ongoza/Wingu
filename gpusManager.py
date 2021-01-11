# TODO
# # add gpu load balanser
import os, sys, traceback, time
import queue, threading

import logging

import tensorflow as tf
import yaml
import cv2
import nvidia_smi
#import psutil

import gpuDevice

class Manager(threading.Thread):
    def __init__(self, configFileName):
        threading.Thread.__init__(self)
        self.gpusActiveList = {}  # running devices
        self.gpusList = [] # available devices
        self.camsList = {} # available cams on gpus
        self.log = logging.getLogger('app')
        self.log.setLevel(logging.DEBUG)
        f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(f)
        self.log.addHandler(ch)

        self.isGPU = False
        try:
            with open(os.path.join('config', configFileName+'.yaml')) as f:    
                self.config = yaml.load(f, Loader=yaml.FullLoader) 
            print("gpus manager config", self.config)
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                self.isGPU = True
                nvidia_smi.nvmlInit()
                if len(gpus) >= len(self.config['gpus_configs_list']):
                    for key in self.config['gpus_configs_list'].keys():               
                        self.gpusList.append(gpus[key].name)                        
                        if key in self.config['gpus_list_active']:
                            self.startGpu(self.config['gpus_configs_list'][key])
                    time.sleep(2)                        
            else:
                self.gpusList.append("CPU")  # Init for CPU
                if self.config['gpus_list_active']:
                    self.startGpu(self.config['cpu_config'])
            time.sleep(5)
            self.log.debug("Active GPUs list: "+" ".join(self.gpusActiveList.keys()))
            if self.config['autotart_streams']:
                for stream in self.config['autotart_streams']:
                    self.log.debug("try to autostart "+ stream)
                    self.startStream(stream)
        except:
            self.log.error("Can not start gpus manager")
            print(sys.exc_info())
            self.kill()            

    def getHardwareStatus(self):
        res = {}
        if self.isGPU:
            for i in self.camsList:
                r = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                res["GPU_"+str(i)] = {"Gpu":res.gpu, "Mem":res.memory}
        res["CPU"] = psutil.cpu_percent()
        res["Mem"] = psutil.virtual_memory().percent
        res["Mem %"] = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
        # res["Disk %"] = psutil
        return res

    def startGpu(self, configFileName, device_id="CPU"):
        # id 0 - CPU, 1 - the first GPU,  etc 
        self.log.debug("start gpu: "+ str(device_id) + " ".join(self.getActiveGpusList()))
        if device_id in self.gpusActiveList.keys():
            self.log.debug("GPU "+str(device_id)+" is already running")
        else:
            try:
                print("GPU config: ", self.gpusList, configFileName)
                # gpu = GpuDevice(device, 'GPU_0', log)
                self.gpusActiveList[device_id] = gpuDevice.GpuDevice(device_id, configFileName, self.log)
                self.log.info("GPU "+ str(device_id) +" is running OK ")
                self.log.debug("added gpu "+ str(device_id) +" to list: "+" ".join(self.getActiveGpusList()))
            except:
                self.log.error("GPU "+str(device_id)+" is not running OK ")
                print(sys.exc_info())
            
    def getActiveGpusList(self):
        res = []
        for gpu_id in self.gpusActiveList.keys():
            try:
                if self.gpusActiveList[gpu_id].id:
                    res.append(gpu_id)
                    self.log.debug("ok " + gpu_id)
            except:
                print("except in getActiveGpusList")
                del self.gpusActiveList[gpu_id]
        return res

    def getCamFrame(self, cam_id):
        res = None
        if cam_id in self.camsList.keys():
            gpu_id = self.camsList[cam_id]
            if gpu_id in self.gpusActiveList.keys():
                try:
                    print("check gpu", self.gpusActiveList[gpu_id].id)
                    if self.gpusActiveList[gpu_id].id:
                        for cam in self.gpusActiveList[gpu_id].getCamsList():
                            res = res.append(cam.get_status)
                except:
                    self.error.log("Can not take status GPU in Manager! GPU:" + gpu_id)
            return res
            
    def getCamsStatus(self):
        res =[]
        for gpu_id in self.getActiveGpusList():
            try:
                print("check cams in gpu", gpu_id)
                for cam in self.gpusActiveList[gpu_id].getCamsList():
                    res.append(cam.get_status)
            except:
                self.error.log("Can not take status GPU in Manager! GPU:" + gpu_id)
        return res

    def startStream(self, config, device=None):
        self.log.info("Start stream "+str(config)+" on device "+str(device))
        if device:
            if device in self.getActiveGpusList():
                self.gpusActiveList[device].startCam(config, device)
            else:
                self.log.info("Device "+str(device)+" is not availble")
        else:
            devices_cnt = len(self.gpusList)
            devices_cnt_act = len(self.gpusActiveList)
            act_ids = self.getActiveGpusList()
            if devices_cnt >= devices_cnt_act:
                if devices_cnt == 1:
                    if devices_cnt_act == 1:
                        self.gpusActiveList[act_ids[0]].startCam(config, 0)
                    else:
                        self.startGpu(self.gpusList[0])
                        time.sleep(5)
                        self.gpusActiveList[act_ids[0]].startCam(config, 0)
                else:
                    # add GPUs balanser
                    #self.startGpu(self.gpusList)
                    #time.sleep(5)
                    self.gpusActiveList[act_ids[0]].startCam(config, 0)
            else:
                print("Active gpus list is more then ")

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
    manager = Manager("Gpus_manager_default")
    time.sleep(10)
    gpu_keys = manager.getActiveGpusList()
    gpu = manager.gpusActiveList[gpu_keys[0]]
    print("GPU", " ".join(gpu_keys))
    # print("gpu.cams[0]", gpu.cams[0].id)
    # manager.startStream(""
    # )
    while True:
        time.sleep(3)
        log.debug("tik")
        try:
            log.debug("gpu.cams[0].outFrame "+ str(len(gpu.cams)))
            for i, cam in enumerate(gpu.cams):
                # print("frame "+ gpu.cams[0].id +" ", gpu.cams[0].outFrame)
                print("frame "+ cam.id +" ", cam.cur_frame_cnt)
                if cam.outFrame.any():
                    cv2.imshow(str(cam.id), cam.get_cur_frame())
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
    manager.kill()    
    cv2.destroyAllWindows()
    log.debug("Stoped - OK")

   
   