# TODO
# # add gpu load balanser
import os, sys, traceback, time
import queue, threading
import asyncio
import logging
import numpy as np
import tensorflow as tf
import yaml
import cv2
import nvidia_smi
import psutil
from requests_futures import sessions

import gpuDevice

#from wingu_server import ws_send_data
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Manager(threading.Thread):
    def __init__(self, configFileName):
        threading.Thread.__init__(self)
        self.gpusActiveList = {}  # running devices
        self.gpusList = {} # available devices
        self.gpusConfigList = {} #  devices configs
        self.streamsConfigList = {}
        self.camsList = {} # running streams on gpus
        f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
        # Add file rotating handler, with level DEBUG
        fileLog = logging.handlers.RotatingFileHandler('manager.log', 'a', 100000, 5)
        fileLog.setLevel(logging.DEBUG)
        fileLog.setFormatter(f)
        logging.getLogger().addHandler(fileLog)

        self.log = logging.getLogger('appManager')
        self._stopevent = threading.Event()
        self.isGPU = False
        self.server_URL = "http://localhost:8080/update?"
        self.session = sessions.FuturesSession(max_workers=2)
        # self.camIdFrame = []
        self.isGpuStarted = False
        try:
            self.config = self.loadConfig(configFileName, 'Gpus_manager_')
            if self.config:
                self.db_path = "db.wingu.sqlite3"
                self.db_table_name = "stats"
                self.config['gpu_configs'] = {}
                self.config['streams_configs'] = {}
                print("gpus manager config", self.config)
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    print("gpus", gpus)
                    print("len=", len(self.config['gpus_configs_list']), len(gpus))
                    self.isGPU = True
                    nvidia_smi.nvmlInit()
                    self.gpuInfo = {} 
                    if len(gpus) <= len(self.config['gpus_configs_list']):
                        print("gpus_configs_list, gpus config is ok", self.config['gpus_configs_list'])
                        for index, gpu_id in enumerate(gpus):
                            print("key", index, gpu_id)
                            name = "/GPU:" + str(index)
                            device = str(self.config['gpus_configs_list'][index])
                            print("gpus name", name, device)
                            self.gpusList[name] = index
                            print("gpus 333", self.config['gpus_configs_list'][index])
                            cfg = self.loadConfig(self.config['gpus_configs_list'][index], "Gpu_")
                            print("gpus cfg", cfg)
                            if cfg is not None:
                                cfg['device_id'] = name
                                cfg['fileName'] = device
                                self.gpusConfigList[name] = cfg
                                if device in self.config['autostart_gpus_list']:
                                    self.startGpu(cfg, name)
                                    self.gpuInfo[name] = nvidia_smi.nvmlDeviceGetHandleByIndex(index)
                                    time.sleep(3)
                else:
                    name = "/CPU:0"
                    self.gpusList[name] = 0  # Init for CPU
                    cfg = self.loadConfig(self.config['cpu_config'], "GPU_")
                    if cfg != None:
                        cfg['device_id'] = name
                        cfg['fileName'] = self.config['cpu_config']
                        self.gpusConfigList['cpu'] = cfg
                        if self.config['autostart_gpus_list'] != None:
                            self.startGpu(cfg, name)
                self.log.debug("GPUsmanager Active GPUs list: "+" ".join(self.gpusActiveList.keys()))
                for stream in self.config['streams']:
                    stream = str (stream)
                    cfg = self.loadConfig(stream, "Stream_")
                    if cfg != None:
                        self.streamsConfigList[stream] = cfg
                        if self.config['autostart_streams']:
                            if stream in self.config['autostart_streams']:
                                self.startStream(stream)
                self.ready = True
                try:
                    self.log.debug("GPUsmanager try to autostart "+ stream)
                    time.sleep(3)
                except:
                    self.log.debug("GPUsmanager exception autostart "+ stream)

                self.session.get(self.server_URL+"cmd=startManager&status=OK&name=Init&module=Manager")
                # self.start()
            else:
                self.log.error("GPUsmanager Can not load config gor GPUs Maanger")
        except:
            self.log.error("GPUsmanager Can not start gpus manager")
            print(sys.exc_info())
            self.kill()            


    def getCamsStat(self):
        res = {}
        if self.camsList:
            for cam_id in list(self.camsList):
                try:
                     data = self.getCamStat(cam_id)
                     if data:
                        res[cam_id] = data
                except:
                    print("GpusManager error get cam stats")
                    print(sys.exc_info())
        return res

    def removeCam(self, cam_id):
        try:
            if cam_id in self.camsList:
                gpu_id = self.camsList[cam_id]
                del self.camsList[cam_id]
                self.gpusActiveList[gpu_id].removeCam(cam_id)
                print("GpusManager removeCam OK")
        except:
            print("GpusManager error get remove cam")
            print(sys.exc_info())

    def getCamStat(self, cam_id):
            res = []
            try:
                gpu_id = self.camsList[cam_id]
                res = self.gpusActiveList[gpu_id].cams[cam_id].get_cur_stat()
            except:
                print("GpusManager err get Cam stat")
                print(sys.exc_info())

            return res

    def getCamFrame(self, cam_id):
        res = None
        try:
            if cam_id in self.camsList:
                gpu_id = self.camsList[cam_id]
                if gpu_id in self.gpusActiveList:
                    if cam_id in self.gpusActiveList[gpu_id].cams:
                        res = self.gpusActiveList[gpu_id].cams[cam_id].get_cur_frame() 
        except:
            print("GpusManager getCamFrame exception")
            print(sys.exc_info())
        return res

    async def addConfig(self, name, type, cfg, client=None, auto=None):
        print("GPUsmanager start addConfig", name, type, cfg)
        if type=="Stream_":
            if auto is not None:
                if auto:
                    print("manager auto", self.config['autostart_streams'] )
                    if self.config['autostart_streams'] is None: self.config['autostart_streams'] = []
                    if name not in self.config['autostart_streams']:
                       self.config['autostart_streams'].append(name)
                else:
                    if self.config['autostart_streams'] is None: self.config['autostart_streams'] = []
                    if name in self.config['autostart_streams']:
                       self.config['autostart_streams'].remove(name)
            if name in self.camsList:
               # !!!!!!!!!!!!!!!!! restart cam!!!!!!!!!!!!
               # self.stopStream(name)
               print("GPUsmanager cam need restart after config update!!!!!!!!!!")
            else:
                self.streamsConfigList[name] = cfg
                #self.getStreamsConfig(self, client)
                self.session.get(self.server_URL+"cmd=configUpdated&type=" + type)                

    #async def _run(self):
    #    while not self._stopevent.isSet():
    #        try:
    #            print("GPUsmanager tik ", len(self.gpusActiveList))
    #            # print("data=", type, id, data)
    #            #await self.getHardwareStatus()
    #            await asyncio.sleep(3)
    #        except:
    #            self.log.debug("GPUsmanager run stoped by exception")


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

    #def getFrame(self):
    #    res = None
    #    try:
    #        print("camIdFrame", self.camIdFrame)
    #        res = self.gpusActiveList[self.camIdFrame[0]].cams[self.camIdFrame[1]].send_cur_frame()             
    #    except: print(sys.exc_info())
    #    return res

    async def getStreamsConfig(self, client):
        try:
            res = self.getCamsStatus()
            await client.send_json({'streamsConfigList': self.streamsConfigList, "camsList": res})
        except:
            await client.send_json({"error": ["GPUsmanager", "getStreamsConfig"]})
            print("GPUsmanager send status error")
            print(sys.exc_info())

    def send_data(self, data):
        try:
            self.session.get(self.server_URL+data)            
        except:
            print("GpusMaanger error send data")

    def startStream(self, configName, device=None):
        self.log.info("GPUsmanager Start stream "+str(configName)+" on device "+str(device))
        try:
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
        except:
            print(sys.exc_info())
            self.log.debug("GPUsmanager Can not start stream!")
            self.send_data("cmd=startStream&name=" + configName + "&status=error&module=Manager")

    def stopStream(self, cam_id):
        try:
            if cam_id in self.camsList:
                print("GPUsmanager stop stream ", cam_id, self.camsList[cam_id])
                if self.camsList[cam_id] in self.getActiveGpusList():
                    print("GPUsmanager stop stream 2", cam_id,  self.camsList[cam_id])
                    self.gpusActiveList[self.camsList[cam_id]].stopCam(cam_id)
                    del self.camsList[cam_id]
        except: 
            print(sys.exc_info())
            self.send_data("cmd=stopStream&name=" + cam_id + "&status=error&module=Manager")


    #def stopGetStream(self, cam_id, client=None):
    #    try:
    #        self.gpusActiveList[self.camIdFrame[0]].cams[self.camIdFrame[1]].stopGetStream(client)
    #        camIdFrame = []
    #        # await client.send_json({"OK":["stopGetStream", cam_id]})
    #    except: 
    #        print(sys.exc_info())
    #        if client is not None:
    #            self.send_data(client, {"error":["stopGetStream", cam_id, "exception on GPUsmanager"]})

    #def startGetStream(self, cam_id, client):
    #    self.log.debug("GPUsmanager start get stream " + cam_id)
    #    if cam_id in self.camsList:
    #        try:
    #            gpu_id = self.camsList[cam_id]
    #            if gpu_id in self.getActiveGpusList():
    #               print("check gpu 3", cam_id, gpu_id)
    #               self.gpusActiveList[gpu_id].cams[cam_id].startGetStream(client)
    #               self.camIdFrame = [gpu_id, cam_id]
    #        except:
    #            print(sys.exc_info())
    #            self.log.debug("GPUsmanager startGetStream exception!")
    #            if client is not None:
    #                await client.send_json({"error":["startGetStream", cam_id, "exception on GPUsmanager"]})

    def getSreamsStatus(self):
        res = self.getCamsStatus()
        return res

    def getConfig(self):
        #np.append(uid, np.uint8(device_id))
        return {'managerConfig':self.config}

    def getHardwareStatus(self):
        res = {}
        try:
            c_t = int(psutil.sensors_temperatures()['i350bb'][0].current)
            #c_t = 0
            res = {'cpu':[int(psutil.cpu_percent()), int(psutil.virtual_memory().percent), c_t, len(self.camsList)]}        
            if self.isGPU:
                if self.gpuInfo:
                    for name in self.gpuInfo:
                        handle = self.gpuInfo[name]
                        gpu = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                        temp = nvidia_smi.nvmlDeviceGetTemperature(handle, nvidia_smi.NVML_TEMPERATURE_GPU)
                        num = len(self.getActiveGpusList[name].cams)
                        res[name] = [int(gpu.gpu), int(gpu.memory), int(temp), num]
        except:
            print("get hardware")
            print(sys.exc_info())
        return res

    def startGpu(self, configFile, device="/CPU:0"):
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
        # print("GPUsmanager getActiveGpusList")
        #print("GPUsmanager self.streamsConfigList", self.gpusActiveList)
        res = []
        for gpu_id in list(self.gpusActiveList):
            # print("GPUsmanager key", gpu_id, self.gpusActiveList[gpu_id])
            try:
                if self.gpusActiveList[gpu_id].device:
                    res.append(gpu_id)
                    # self.log.debug("GPUsmanager CHECK gpu ok " + gpu_id)
            except:
                print("GPUsmanager except in getActiveGpusList")
                print(sys.exc_info())
                del self.gpusActiveList[gpu_id]
        # print("res", res)
        return res

    #def getCamFrames(self):
    #    res = []
    #    for cam_id in self.camsList:
    #        fr = self.getCamFrame(cam_id)
    #        if fr:
    #            res.append(fr)
    #        else:
    #            self.log.info("GPUsmanager can not take frame for stream: " + cam_id)
    #    return res

    #def getCamFrame(self, cam_id):
    #    res = None
    #    if cam_id in self.camsList:
    #        try:
    #            gpu_id = self.camsList[cam_id]
    #            # print("check gpu 2", cam_id, gpu_id, self.gpusActiveList[gpu_id].cams)
    #            if gpu_id in self.getActiveGpusList():
    #                uid = self.gpusActiveList[gpu_id].cams[cam_id].uid
    #                frame = self.gpusActiveList[gpu_id].cams[cam_id].get_cur_frame()
    #                tr, frame_jpg = cv2.imencode('.jpg', frame)
    #                if tr:
    #                    res = np.append(frame_jpg, uid)
    #                    #with open("video/dd__00.jpg",'wb') as f:
    #                    #    f.write(res)
    #                    # print("write ok")
    #                    # res = frame
    #        except:
    #            self.error.log("GPUsmanager Can not take frame in Manager! Stream:" + cam_id)
    #    return res
            
    def getCamsStatus(self):
        res = []
        try:
            gpu_ids = self.getActiveGpusList()
            for gpu_id in gpu_ids:
                cams = self.gpusActiveList[gpu_id].getCamsList()
                for cam in cams:
                    res.append(self.gpusActiveList[gpu_id].cams[cam].get_status())
        except:
            self.log.debug("GPUsmanager Can not take status GPU in Manager! GPU:" + gpu_id)
            print(sys.exc_info())
        return res

    #def stopGpu(self, id):
    #    if id in self.gpusActiveList:
    #        self.gpusActiveList[id].kill()
    #        del self.gpusActiveList[id]
    #        self.log.info("GPUsmanager stoped gpu: "+ str(id))
    #    else:
    #        self.log.info("GPUsmanager skip stoped gpu: "+ str(id))

    def kill(self):
        self.log.debug("GPUsmanager try to stop")
        self._stopevent.set()
        for id in list(self.gpusActiveList):
            try:
                self.log.debug("GPUsmanager try stop gpu: "+ str(id))
                self.gpusActiveList[id].kill()
            except:
                self.log.debug('GPUsmanager Erorr stop gpu: '+ str(id))
        time.sleep(4)
        for id in list(self.gpusActiveList):
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
    gpu = manager.gpusActiveList[gpu_id]
    time.sleep(10)
    print("frame", gpu.cams[cam].id)
    if gpu:
        while False:
            try:
                start = time.time()
                print("GPUsmanager cams "+ str(len(gpu.cams)))
                for i, cam in enumerate(gpu.cams):
                    # print("frame "+ gpu.cams[0].id +" ", gpu.cams[0].outFrame)
                    print("GPUsmanager frame "+ str(cam))
                    if gpu.cams[cam].outFrame.any():
                        cv2.imshow(str(cam), gpu.cams[cam].outFrame)
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

   
   