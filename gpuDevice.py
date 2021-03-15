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
import gpu_trt_tool as trt_det
# import tensorflow as tf
import tensorrt as trt
import pycuda.driver as cuda
import asyncio
from requests_futures import sessions
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU

#from yolov3_tf2.models import YoloV3

#from tf2_yolov4.anchors import YOLOV4_ANCHORS
#from tf2_yolov4.model import YOLOv4

import videoCapture
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
        self.server_URL = "http://localhost:8080/update?"
        self.session = sessions.FuturesSession(max_workers=2)
        self.proceedTime = 0
        #physical_devices = tf.config.experimental.list_physical_devices('GPU')
        #print("physical_devices", physical_devices)
        self.device = str(device_name)
        try:
            self.config = configFile 
            print("GpuDevice start init model")

            print("GpuDevice init GPU config", self.config)
            self.cnt = 0
            self.frame = []
            self.img_size = self.config['img_size']
            self.batch_size = 4
            self._LOG = trt.Logger()
            #cuda.init()
            #self.trt_device = cuda.Device(0)
            self.engine = trt_det.get_engine(os.path.join('models', 'yolov4_-20_3_416_416_dynamic.engine'), self._LOG)
            print("engine is OK")
            #self.context = self.trt_device.make_context()
            self.context = self.engine.create_execution_context()
            self.context.set_binding_shape(0, (self.batch_size, 3, self.img_size, self.img_size))
            #inputs = []
            #outputs = []
            #bindings = []
            #self.stream = cuda.Stream()
            #for binding in self.engine:
            #    size = trt.volume(self.engine.get_binding_shape(binding)) * self.batch_size
            #    dims = self.engine.get_binding_shape(binding)
            #    # in case batch dimension is -1 (dynamic)
            #    if dims[0] < 0: size *= -1
            #    dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            #    # Allocate host and device buffers
            #    host_mem = cuda.pagelocked_empty(size, dtype)
            #    device_mem = cuda.mem_alloc(host_mem.nbytes)
            #    print("device_mem",host_mem,device_mem)
            #    # Append the device buffer to device bindings.
            #    bindings.append(int(device_mem))
            #    # Append to the appropriate list.
            #    if self.engine.binding_is_input(binding):
            #        inputs.append(trt_det.HostDeviceMem(host_mem, device_mem))
            #    else:
            #        outputs.append(trt_det.HostDeviceMem(host_mem, device_mem))
            #self.buffers = (inputs, outputs, bindings, stream)
            self.buffers = trt_det.allocate_buffers(self.engine, self.batch_size)
            print("self.buffers", type(self.buffers))
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

    #def run(self):
    #    loop = asyncio.new_event_loop()
    #    loop.run_until_complete(self._run())
    #    loop.close()

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
                    cam = videoCapture.VideoCapture(camConfig, self.config, self.id, cam_id, self.log)
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
            if self.context:
                self.context.pop()
            del self.context
            self.log.debug("start to stop GPU "+ str(self.id))
            self._stopevent.set()
            for cam in list(self.cams):
                print("try stop cam" + cam)
                self.stopCam(cam)
                # self.cams[cam].kill()
                print("cams num:", len(self.cams))
            time.sleep(4)
            self.log.info("GPU "+str(self.id)+" stopped")        
            self.killed = True
            self.session.get(self.server_URL + "cmd=GpuStop&name="+str(id)+"&status=OK")
        except:
            print("can not stop some cams")

    def removeCam(self, cam_id):
        print("GpuDevice removeCam", cam_id)
        if cam_id in self.cams:
            del self.cams[cam_id]

    def run(self):
        self.log.debug("GpuDevice starting "+str(self.id))
        #with trt_det.get_engine(os.path.join('models', 'yolov4_-20_3_416_416_dynamic.engine'), self._LOG) as engine, engine.create_execution_context() as context:

        while not self._stopevent.isSet():
                # print("GpuDevice tik")
                start = time.time()
                self.cnt += 1
                frames = []
                cams = []
                for cam in list(self.cams):
                    if self.cams[cam]._stopevent.isSet():
                        # self.log.debug("GpuDevice cam stopped: " + cam)
                        self.cams[cam].kill()
                        del self.cams[cam]
                        print("del camera ", cam)
                        #delete cam from cams list
                        pass
                    else:
                        data = self.cams[cam].read()
                        if type(data) != bool:
                            print("data in", data.shape)
                            frames.append(data)
                            cams.append(self.cams[cam])
                        else:
                            print("skip data for cam:", cam)
                        #print("cur_frame", cam.id, cam.cur_frame)
                        #if (tr): cv2.imwrite("video/39_2.jpg", frames[0])
                # start2 = time.time()
                # print("read time=",len(frames), start2 - start)
                if frames:
                    start3 = time.time()
                    batch_size = len(frames)
                    frames_tf = np.stack(frames, axis=0)
                    #print("-Shape of the network input: ", frames_tf.shape)
                    #print("start detect")
                    frames_tf = np.ascontiguousarray(frames_tf)
                    print("Shape of the network input: ", frames_tf.shape, batch_size)
                    boxes = trt_det.detect(self.context, self.buffers, frames_tf, batch_size, self.img_size)
                    # print("detect", len(boxes[0]))
                    try:
                        # frame = frames[0]
                        # print("frame rs = ", frames.shape)
                        # convert numpy opencv to tensor
                        #frames_tf = tf.convert_to_tensor(frames, dtype=tf.float32, dtype_hint=None, name=None)
                        #print("to GPU=", time.time() - start3)
                        # start4 = time.time()
                        #boxes = self.detector.predict(frames_tf)
                        for j in range(len(cams)):
                           cams[j].track(boxes[j])
                        self.proceedTime = time.time() - start
                        print("detect time =", self.proceedTime, time.time() - start3)
                    except:
                        self.log.error("GpuDevice "+str(self.id)+" skip frame by exception")
                        print(sys.exc_info(), type(frames_tf))
                else:
                    time.sleep(1)
                # self.log.info("GpuDevice Any available streams in GPU "+str(self.id))
                # self.kill()
                print("TRT cams({}) proceed time ={}".format(len(cams), time.time() - start))


# if __name__ == "__main__":
    # tf.debugging.set_log_device_placement(True)
    # log = logging.getLogger('app')
    # log.setLevel(logging.DEBUG)
    # f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    # ch.setFormatter(f)
    # log.addHandler(ch)
    # devices = ["CPU"]
    # print(tf.config.experimental.list_physical_devices())
    # # my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    # # tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
    # for gpu in tf.config.experimental.list_physical_devices('GPU'):
    #     print("cuda is available " + str(len(gpus)))
    #     devices.append(gpu.name)
    #     device_name = gpus[0].name
    # else:
    #     print("cuda is not available")
    #     # device = tf.config.experimental.list_physical_devices('CPU')[0].name
    # print("devices: " + " ".join(devices))
    # print("device=", devices[0])
    # device_id = 0
    # device_name = devices[0]
    # tr = False
    # gpu = None
    # try:
    #     with open('config/Stream_file_39.yaml', encoding='utf-8') as f:
    #         configStream = yaml.load(f, Loader=yaml.FullLoader)
    #     with open(os.path.join('config', 'Gpu_device0.yaml'), encoding='utf-8') as f:
    #         configGpu = yaml.load(f, Loader=yaml.FullLoader)
    #     gpu = GpuDevice(device_id, device_name, configGpu, log)
    #     time.sleep(5)
    #     gpu.startCam(configStream, "file_39", 0)
    #     #time.sleep(5)
    #     #gpu.startCam('Stream_43', 0)
    #     tr = True
    # except:
    #     print("ecxept!!!!!")
    #     print(sys.exc_info())
    #     if gpu:
    #         print("try stop gpu from main")
    #         gpu.kill()
    # if True:
    #     print("cams "+ str(len(gpu.cams)))
    #     while True:
    #         try:
    #             start = time.time()
    #             for i, cam in enumerate(gpu.cams):
    #                 # print("frame "+ gpu.cams[0].id +" ", gpu.cams[0].get_cur_frame())
    #                 # print("frame "+ cam.id +" ", cam.cur_frame_cnt)
    #                 if gpu.cams[cam].outFrame.any():
    #                     cv2.imshow(str(cam), gpu.cams[cam].get_cur_frame())
    #                 if gpu.cams[cam].proceedTime[0]:
    #                     print("fpsRead_"+str(i), 1.0/(gpu.cams[cam].proceedTime[0]))
    #                 if gpu.cams[cam].proceedTime[1]:
    #                     print("fpsTrack_"+str(i), 1.0/(gpu.cams[cam].proceedTime[1]))
    #             key = cv2.waitKey(1)
    #             if key & 0xFF == ord('q'): break
    #         except:
    #             print("stop by exception")
    #             print(sys.exc_info())
    #             break
    # else:
    #     try:
    #         time.sleep(60)
    #     except:
    #         print(sys.exc_info())
    #         gpu.kill()
    # print("gpu ", gpu)
    # if gpu:
    #     print("try stop gpu from main")
    #     gpu.kill()
    # cv2.destroyAllWindows()
    # print("Stoped - OK")
    #