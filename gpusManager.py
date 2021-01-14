# TODO
# # add gpu load balanser
import os, sys, traceback, time
import queue, threading

import logging
import numpy as np
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
        self.gpusList = {} # available devices
        self.gpusConfigList = {} #  devices configs
        self.streamsConfigList = {}
        self.camsList = {} # running streams on gpus
        self.log = logging.getLogger('app')
        self.log.setLevel(logging.DEBUG)
        f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(f)
        self.log.addHandler(ch)
        self.isGPU = False
        self.isGpuStarted = False
        try:
            self.config = self.loadConfig(configFileName, 'Gpus_manager_')
            if self.config:
                self.config['gpu_configs'] = {}
                self.config['streams_configs'] = {}
                print("gpus manager config", self.config)
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    self.isGPU = True
                    nvidia_smi.nvmlInit()
                    if len(gpus) >= len(self.config['gpus_configs_list']):
                        for index, key in enumerate(self.config['gpus_configs_list']):
                            key = str(key)
                            self.gpusList[gpus[key].name] = index
                            cfg = self.loadConfig(['gpus_configs_list'][key], "Gpu_")
                            if cfg != None:
                                cfg['device_id'] = index
                                cfg['fileName'] = key
                                self.gpusConfigList['Gpu_'+index] = cfg
                                if key in self.config['autostart_gpus_list']:
                                    self.startGpu(self.config['gpus_configs_list'][key])
                                    time.sleep(3)                        
                else:
                    self.gpusList["CPU"] = 0  # Init for CPU
                    cfg = self.loadConfig(self.config['cpu_config'], "GPU_")
                    if cfg != None:
                        cfg['device_id'] = 0
                        cfg['fileName'] = self.config['cpu_config']
                        self.gpusConfigList['cpu'] = cfg
                        if self.config['autostart_gpus_list'] != None:
                            self.startGpu(cfg)
                            time.sleep(3)
                self.log.debug("GPUsmanager Active GPUs list: "+" ".join(self.gpusActiveList.keys()))
                for stream in self.config['streams']:
                    stream = str (stream)
                    cfg = self.loadConfig(stream, "Stream_")
                    if cfg != None:
                        self.streamsConfigList[stream] = cfg
                        # print("cfg",self.streamsConfigList)
                        if stream in self.config['autotart_streams']:
                            self.log.debug("GPUsmanager try to autostart "+ stream)
                            self.startStream(stream)
            else:
                self.log.error("GPUsmanager Can not load config gor GPUs Maanger")
                raise
        except:
            self.log.error("GPUsmanager Can not start gpus manager")
            print(sys.exc_info())
            self.kill()            

    def updateConfig(self, cfg):
        res = False
        try:
            if cfg['tp'] == "Stream_":
               if cfg['name'] in self.camsList:
                   print("GPUsmanager alreday runnning need to stop before updating")
               else:
                   print("GPUsmanager start update")
                   self.streamsConfigList[cfg['name']] = cfg['data'] 
            elif cfg['tp'] == "Gpu_":
                print("GPUsmanager update gpu config")
            elif cfg['tp'] == "Gpu_":
                print("GPUsmanager update gpu config")
            else:
                print("GPUsmanager error")
        except:
            print(sys._ext_info())
            self.log.error("GPUsmanager can not updateConfig for " + cfg['tp']+' '+cfg['name'] )
        return res

    def loadConfig(self, fileName, tp):
        res = None
        fileName = os.path.join('config', tp + str(fileName)+'.yaml')
        # print("GPUsmanager fileName", fileName)
        if os.path.isfile(fileName):
            try:
                with open(fileName, encoding='utf-8') as f:    
                    res = yaml.load(f, Loader=yaml.FullLoader)
            except:
                self.log.error("GPUsmanager Can not load config for " + fileName )
        return res

    def getStreamsConfig(self):
        return {'streamsConfigList': self.streamsConfigList}

    def getConfig(self):
        #np.append(uid, np.uint8(device_id))
        return {'managerConfig':self.config, 'gpusConfigList':self.gpusConfigList}

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

    def startGpu(self, configFile, device="CPU"):
        # id 0 - CPU, 1 - the first GPU,  etc 
        self.log.debug("GPUsmanager start gpu: "+ str(device) + " ".join(self.getActiveGpusList()))
        if device in self.gpusActiveList:
            self.log.debug("GPUsmanager "+str(device)+" is already running")
        else:
            try:
                device_id = self.gpusList[device]
                print("GPUsmanager config: ", self.gpusList, device_id, configFile)
                # gpu = GpuDevice(0, device, 'GPU_0', log)
                # id, device, device_id, configFile, log
                self.gpusActiveList[device] = gpuDevice.GpuDevice(device_id, device, configFile, self.log)
                self.log.info("GPUsmanager "+ str(device) +" is running OK device_id: " + str(device_id))
                self.log.debug("GPUsmanager added gpu "+ str(device))
            except:
                self.log.error("GPUsmanager "+str(device)+" is not running OK ")
                print(sys.exc_info())
            
    def getActiveGpusList(self):
        print("getActiveGpusList")
        #print("GPUsmanager self.streamsConfigList", self.gpusActiveList)
        res = []
        for gpu_id in self.gpusActiveList.keys():
            print("key", gpu_id, self.gpusActiveList[gpu_id])
            try:
                if self.gpusActiveList[gpu_id].device:
                    res.append(gpu_id)
                    self.log.debug("ok " + gpu_id)
            except:
                print("GPUsmanager except in getActiveGpusList")
                print(sys.exc_info())
                del self.gpusActiveList[gpu_id]
        # print("res", res)
        return res

    def getCamFrames(self):
        res = []
        for cam_id in self.camsList:
            fr = self.getCamFrame(cam_id)
            if fr:
                res.append(fr)
            else:
                self.log.info("GPUsmanager can not take frame for stream: " + cam_id)
        return res

    def getCamFrame(self, cam_id):
        res = None
        if cam_id in self.camsList:
            try:
                gpu_id = self.camsList[cam_id]
                # print("check gpu 2", cam_id, gpu_id, self.gpusActiveList[gpu_id].cams)
                if gpu_id in self.getActiveGpusList():
                    uid = self.gpusActiveList[gpu_id].cams[cam_id].uid
                    frame = self.gpusActiveList[gpu_id].cams[cam_id].get_cur_frame()
                    tr, frame_jpg = cv2.imencode('.jpg', frame)
                    if tr:
                        res = np.append(frame_jpg, uid)
                        #with open("video/dd__00.jpg",'wb') as f:
                        #    f.write(res)
                        # print("write ok")
                        # res = frame
            except:
                self.error.log("GPUsmanager Can not take frame in Manager! Stream:" + cam_id)
        return res
            
    def getCamsStatus(self):
        res =[]
        for gpu_id in self.getActiveGpusList():
            try:
                print("GPUsmanager check cams in gpu", gpu_id)
                for cam in self.gpusActiveList[gpu_id].getCamsList():
                    res.append(cam.get_status)
            except:
                self.error.log("GPUsmanager Can not take status GPU in Manager! GPU:" + gpu_id)
        return res

    def startStream(self, configName, device=None):
        self.log.info("GPUsmanager Start stream "+str(configName)+" on device "+str(device))
        if configName in self.streamsConfigList:
            config = self.streamsConfigList[configName]
            if device:
                if device in self.getActiveGpusList():
                    self.gpusActiveList[device].startCam(config, configName, device)
                    self.camsList[configName] = device
                    self.log.debug("cam add to camsList "+ str(device)+" "+configName)
                    print("cam add to camsList ", self.camsList)
                else:
                    self.log.info("GPUsmanager Device "+str(device)+" is not availble")
            else:
                devices_cnt = len(self.gpusList)
                act_ids = self.getActiveGpusList()
                devices_cnt_act = len(act_ids)
                print("act_ids", act_ids)
                if devices_cnt >= devices_cnt_act:
                    if devices_cnt == 1:
                        if devices_cnt_act == 1:
                            self.gpusActiveList[act_ids[0]].startCam(config, configName, 0)
                            self.camsList[configName] = act_ids[0]
                            self.log.debug("cam add to camsList "+ str(act_ids[0])+" "+configName)
                            print("cam add to camsList ", self.camsList)
                        else:
                            print("GPUsmanager add device management")
                            #self.startGpu(self.gpusList[0])
                            # self.startGpu(self.gpusList[0])
                            #time.sleep(5)
                            #self.gpusActiveList[act_ids[0]].startCam(config, 0)
                    else:
                        # add GPUs balanser
                        #self.startGpu(self.gpusList)
                        #time.sleep(5)
                        self.gpusActiveList[act_ids[0]].startCam(config, configName, 0)
                        self.camsList[configName] = act_ids[0]
                else:
                    self.log.debug("GPUsmanager Active gpus list is more then ")
        else:
            self.log.debug("GPUsmanager any config for " + configName)
                
    def stopGpu(self, id):
        if id in self.gpusActiveList:
            self.gpusActiveList[id].kill()
            del self.gpusActiveList[id]
            self.log.info("GPUsmanager stoped gpu: "+ str(id))
        else:
            self.log.info("GPUsmanager skip stoped gpu: "+ str(id))

    def kill(self):
        self.log.debug("GPUsmanager try to kill gpus") 
        for id in self.gpusActiveList.keys():
            try:
                self.log.debug("GPUsmanager try stop gpu: "+ str(id))
                self.gpusActiveList[id].kill()
            except:
                self.log.debug('GPUsmanager Erorr stop gpu: '+ str(id))
        time.sleep(10)
        for id in self.gpusActiveList.keys():
            try:
                ready = self.gpusActiveList[id].id
            except:
                self.log.debug('GPUsmanager Eroro stop gpu: '+ str(id))
                del self.gpusActiveList[id]
        if len(self.gpusActiveList) == 0:
            self.log.debug('GPUsmanager No active gpus now...')
        else:
            self.log.error('GPUsmanager Still active gpus: '+ " ".join([str(key) for key in self.gpusActiveList.keys()]))


if __name__ == "__main__":
    manager = Manager("default")
    time.sleep(10)
    if manager:
        print("GPUs:", manager.gpusActiveList)
    # print("config",manager.getConfig())
    gpu_id = list(manager.gpusActiveList.keys())[0]
    print("gpu_id", gpu_id)
    gpu = manager.gpusActiveList[gpu_id]
    time.sleep(20)
    print("frame",manager.getCamFrame('file_39'))
    if gpu:
        while False:
            try:
                start = time.time()
                print("GPUsmanager cams "+ str(len(gpu.cams)))
                for i, cam in enumerate(gpu.cams):
                    # print("frame "+ gpu.cams[0].id +" ", gpu.cams[0].outFrame)
                    print("GPUsmanager frame "+ str(cam))
                    if manager.getCamFrame(cam).any():
                        cv2.imshow(str(cam), manager.getCamFrame(cam))
                    #if cam.proceedTime[0]:
                    #    print("fpsRead_"+str(i), 1.0/(cam.proceedTime[0]))
                    #if cam.proceedTime[1]:
                    #    print("fpsTrack_"+str(i), 1.0/(cam.proceedTime[1]))
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'): break
            except KeyboardInterrupt:
                print("GPUsmanager try to stop by hot keys")
                manager.kill()
                break
            except:
                print("GPUsmanager try to stop by exception")
                print(sys.exc_info())
                manager.kill()
                break
    if manager:
        manager.kill()    
    cv2.destroyAllWindows()
    print("Stoped - OK")

   
   