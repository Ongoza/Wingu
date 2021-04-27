import os, sys, traceback
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
import time
import threading
import logging
import math
import numpy as np
import numba as nb

import cv2
#import asyncio
from requests_futures import sessions
import deep_sort.generate_detections_onnx as gdet

from deep_sort import nn_matching

from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


class VideoCaptureFile:
    def __init__(self, name, frame_res, skip_frames=0):
        self.id = name
        print("start file reader")
        self.skip_frames = skip_frames
        self.cap = cv2.VideoCapture(name)
        self.totalFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 10
        self.proceed_frames_cnt = 0
        self.cur_frame_cnt = 0
        self.frame_res = frame_res
        print("start file reader 2")
        self.q = None
        self.takeOff = True
        self.isStop = False
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            #if self.takeOff:
                #print("read", self.proceed_frames_cnt, self.totalFrames)
                if self.proceed_frames_cnt < self.totalFrames:
                    ret, frame = self.cap.read()
                    self.cur_frame_cnt += 1
                    self.proceed_frames_cnt = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if self.skip_frames:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.proceed_frames_cnt + self.skip_frames)
                    #print("ret", ret, self.proceed_frames_cnt)
                    if (ret):
                        frame = cv2.resize(frame, self.frame_res)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        #frame_tf = np.transpose(frame, (2, 0, 1)).astype(np.float32) / 255.0
                        self.q = frame
                        #self.takeOff = False
                    else:
                        self.q = None
                        #self.takeOff = True
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.proceed_frames_cnt + 1)
                        time.sleep(0.01)
                else:
                    self.isStop = True
            #else:
                #time.sleep(0.001)

    def read(self): 
        self.takeOff = True
        return self.q

    def kill(self):
        self.cap.release()

class VideoCaptureStream:
    def __init__(self, name, frame_res):
        self.id = name
        self.isStop = False
        self.frame_res = frame_res
        #print("VideoCaptureStream", name)
        self.cap = cv2.VideoCapture(name)
        #print("VideoCaptureStream", self.cap)

        self.q = None
        t = threading.Thread(target=self._reader)
        t.daemon = True
        print("stream started!")
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
                ret, frame = self.cap.read()
                #print("stream ret", ret)
                if ret:
                    frame = cv2.resize(frame, self.frame_res)
                    #cv2.resize(image_src_0, image_size, interpolation=cv2.INTER_LINEAR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    #frame_tf = np.transpose(frame, (2, 0, 1)).astype(np.float32)/255.0
                    self.q = frame
                else:
                    # print("skip frame cam/empty", self.id)
                    self.q = None
                    time.sleep(0.01)

    def read(self):
        return self.q

    def kill(self):
        self.cap.release()

class VideoCapture:
    def __init__(self, camConfig, gpuConfig, device_id, cam_id, log, vc_device="/CPU:0"):
        threading.Thread.__init__(self)
        self.log = log      
        self.log.debug("VideoCapture start init stream object " + str(cam_id))
        self.totalFrames = 0
        self.vc_device = vc_device
        self.server_URL = "http://localhost:8080/update?"
        self.startTime = int(time.time())
        self.pipline_str = 'rtspsrc location={} latency={} ! rtph264depay ! h264parse ! queue leaky=1 ! decodebin ! videoconvert  ! appsink sync=false'
        self.cur_frame_cnt = 0
        self.proceed_frames_cnt = 0
        self.proceedTime = [0, 0]
        self.session = sessions.FuturesSession(max_workers=2)
        self.outFrame = np.array([])
        self.isDrow = False
        self.error_counter_default = 10
        self.error_counter = 10
        self.clients = []
        self.out_color = (252, 35, 240)
        self.in_color = (52, 235, 240)
        self.text_color = (255, 255, 0)
        self.intersections = {}
        self.save_video_res = None
        self.id = str(cam_id)
        self.device_id = device_id        
        self.uid = np.append(
            np.random.choice(
                np.fromstring('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', dtype=np.uint8),
                7), np.uint8(device_id))
        try:
            self.config = camConfig
            print("VideoCapture stream config", self.config)
            self.url = self.config['url']
            if 'type' in self.config: self.type = self.config['type']
            else: self.type = 0 
            self.isFromFile = self.config['isFromFile']
            self.cap = None
            self.buffer = 200
            self.img_size = int(gpuConfig['img_size'])
            print("img size", self.img_size) 
            self.cur_proceed_frame = None
            self.max_hum_w = int(self.img_size/2)
            self.GPUconfig = gpuConfig
            self.frame_res = (self.img_size, self.img_size)
            self.skip_frames = int(self.config['skip_frames'])
            self.batch_size = int(self.config['batch_size'])
            self.path_track = int(self.config['path_track'])
            self.save_video_flag = self.config['save_video_flag']
            print("save_video_flag", self.save_video_flag, cam_id)
            self.display_video_flag = self.config['display_video_flag']
            if(self.config['save_video_flag'] or self.config['display_video_flag']):
                self.isDrow = True
                self.save_video_res = tuple(self.config['save_video_res'])
            print("save video res", self.save_video_res)
            scale = self.img_size/416.0
            self.borders = self.config['borders']
            print("scale", scale, self.borders)
            if self.borders:
                for key in self.borders:
                    self.borders[key][0] = [int(self.borders[key][0][0]*scale),int(self.borders[key][0][1]*scale)]
                    self.borders[key][1] = [int(self.borders[key][1][0]*scale),int(self.borders[key][1][1]*scale)]
                    self.intersections[key] = [0, 0]
            self.out = None
            self.encoder = gdet.create_box_encoder(os.path.join('models',self.config['encoder_filename']), batch_size=self.batch_size, device=self.vc_device)
            if self.save_video_flag:            
                outFile = self.config['save_path'] +"_"+ str(self.startTime)+".avi"
                if outFile == '': outFile =  'video/'+str(self.id)+"_"+ str(self.startTime)+"_auto.avi"
                self.log.info("!!!!!!Save out video to file " + outFile)
                self.out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'XVID'), 5, self.save_video_res)
            print("VideoCapture stream start load tracker")
            self.tracker = Tracker(nn_matching.NearestNeighborDistanceMetric("cosine", self.config['max_cosine_distance'], None))
            print("VideoCapture stream tracker ok")
            if not self.isFromFile:
                #print("start read from camera")
                url = self.pipline_str.format(self.url, self.buffer)
                print("url", self.url,"\n", url)
                #self.q = VideoCaptureStream(self.url, self.frame_res)
                self.q = VideoCaptureStream(url, self.frame_res)
                print("self.q", self.q)
            else:
                #print("start read from file")
                self.q = VideoCaptureFile(self.url, self.frame_res, self.skip_frames)
                self.totalFrames = self.q.totalFrames
                print("created stream ok")
                #if client is not None:
                #    client.send_json({'OK':["startStream", self.id]})
            self.session.get(self.server_URL+'cmd=startStream&name='+self.id+'&status=OK')
            self._stopevent = threading.Event()
        except:
            self.session.get(self.server_URL+'cmd=startStream&name='+self.id+'&status=error')
            self.log.debug("VideoStream Can not start Video Stream for " + camConfig)            
            print("VideoStream err types",type(traceback.print_exception(*sys.exc_info())), type(sys.exc_info())) 
            print("VideoStream err:", sys.exc_info())
            self.id = None

    def writeVideo(self, frame):
       self.out.write(frame)

    def get_status(self):
        status = {}
        try:
            status["id"] = self.id
            status["startTime"] = self.startTime
            #status["proceedTime"] = self.proceedTime
            status["save_video_res"] = self.save_video_res
            status["device_id"] = self.device_id
            status["skip_frames"] = self.skip_frames
            status["save_video_flag"] = self.save_video_flag
            status["cur_frame_cnt"] = self.cur_frame_cnt
            status["proceed_frames_cnt"] = self.proceed_frames_cnt
            status["totalFrames"] = self.totalFrames
        except: print(sys.exc_info())
        return status

    def get_cur_stat(self):
        res = {}
        print(self.id, self.intersections, time.asctime(time.localtime()))
        if self.borders:
            for border in self.borders:
                res[border] = [self.intersections[border][0], self.intersections[border][1]]
                self.intersections[border] = [0, 0]
        return res 

    def get_cur_frame(self):
        return self.outFrame   

    def read(self):
        if self.q.isStop:
            self.session.get(self.server_URL+'cmd=stopStream&name='+self.id+'&status=OK&module=stream')
            print("stop cam", self.id)
            self.kill()
            return None
        else:
            start = time.time()
            frame = self.q.read()
            if frame is not None:
                # self.out.write(frame)
                frame = np.array(frame)
                self.cur_frame_cnt += 1
                self.proceed_frames_cnt += 1
                if self.type == 1:
                   print("start cut area")
                   #https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
                self.proceedTime[0] = time.time() - start
                return frame
            else:
                print("skip by zero", type(frame))
                return None

    def ccw(self,A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    # Return true if line segments AB and CD intersect
    def track_intersection_angle(self,A,B):   
        res = 0
        for key in self.borders:
            C = np.array(self.borders[key][0])
            D = np.array(self.borders[key][1])
            if self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D):            
                v0 = np.array(B) - np.array(A)
                v1 = np.array(D) - np.array(C)
                angle = np.math.atan2(np.linalg.det([v0,v1]), np.dot(v0,v1))
                if (angle > 0):
                    self.intersections[key][0] += 1
                    res = 1
                else:
                    self.intersections[key][1] += 1
                    res = 2
                print("intersections",self.intersections)                
        return res

    def drawBorderLines(self, frame):
        for b in self.borders:
            a = self.borders[b][0]
            b = self.borders[b][1]
            length = 40
            vX0 = b[0] - a[0]; vY0 = b[1] - a[1]
            mag = math.sqrt(vX0*vX0 + vY0*vY0)
            vX = vX0 / mag; vY = vY0 / mag
            temp = vX; vX = -vY; vY = temp
            z0 = (int(a[0]+vX0/2), int(a[1]+vY0/2))
            z1 = (int(a[0]+vX0/2 - vX * length), int(a[1] +vY0/2- vY * length))
            cv2.line(frame, (a[0],a[1]), (b[0],b[1]), (255, 255, 0), 2)
            cv2.arrowedLine(frame, z0, z1, (0, 255, 0), 2)
            cv2.putText(frame, "Out", z1, 0, 1, (0, 255, 0), 1)
        return frame

    def track(self, boxs, frame=None, features=None):
        #print("boxs", len(boxs))
        try:
            start = time.pref_counter()
            #frame = self.cur_proceed_frame
            if self.type == 0:
                #print("len(boxs)", len(boxs))
                if(len(boxs)):
                    start_3 = time.pref_counter()
                    features = self.encoder(frame, boxs)
                    #start_3 = time.time()
                    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
                    #start_4 = time.time()
                    #print("detection", start_4 -start_3)
                    self.tracker.predict()
                    #start_5 = time.time()
                    #print("predict", start_5 - start_4)
                    self.tracker.update(detections)
                    #start_6 = time.time()
                    #print("track.update", start_6 - start_5)
                    for track in self.tracker.tracks:
                        if(not track.is_confirmed() or track.time_since_update > 1):
                            # if(track.time_since_update > life_frame_limit): track.state = 3 # if missed to long than delete id
                            continue 
                        xy = track.mean[:2].astype(np.int)# tuple(())
                        clr = (255, 255, 0) # default color
                        track_name = str(track.track_id) # default name                        
                        if(hasattr(track, 'xy')):
                            if(not hasattr(track, 'calculated')):
                                res = self.track_intersection_angle(track.xy[0], xy)
                                if res > 0:
                                    # cnt_people_in[track.track_id] = 0
                                    track.calculated = str(res)+"_"
                                    track.cross_cnt = self.path_track
                                    if res == 1: track.color = self.out_color
                                    else: track.color = self.in_color
                                    # self.log.debug("intersection!! "+ self.id)
                            else:
                                clr = track.color
                                # track_name = track.calculated  + track_name
                                track.cross_cnt -= 1
                                if(track.cross_cnt < 1): track.state = 3 # delete from track list
                            track.xy.append(xy)
                            if len(track.xy) > self.path_track:
                                track.xy = track.xy[-self.path_track:]
                            # cv2.polylines(frame_sm, [np.array(track.xy)], False, clr, 3)
                        else: 
                            track.xy = [xy]
                        if(self.isDrow):
                            txy =  tuple(xy)
                            cv2.circle(frame, txy, 5, clr, -1)
                            cv2.putText(frame, track_name, txy, 0, 0.4, clr, 1)
                    #start7 = time.time()
                    #print("trakes", start7 - start_6)
                self.drawBorderLines(frame)
                txt_frame = "Frame:"+str(self.cur_frame_cnt)
                for key in self.borders:
                    txt_frame += " In:"+str(self.intersections[key][0])+" Out:"+str(self.intersections[key][1])
                cv2.putText(frame, txt_frame, (10, 340), 0, 0.4, self.text_color, 1)
                self.outFrame = np.copy(frame)
                #start8 = time.time()
                #print('self.outFrame', start8-start7)
                if self.save_video_flag:
                    frame = cv2.resize(self.outFrame,self.save_video_res)
                    self.out.write(frame)
                    #self.writeVideo(frame)
                #print("write", time.time()-start8)
                self.proceedTime[1] = time.time() - start
                #print("track", self.proceedTime[1])
            else:
                pass
                #print("start calculate people in area")
        except:
            print("except videocapture track")
            print(sys.exc_info())

    def kill(self):
        try:
            print("try to kill videoCapture")
            self._stopevent.set()
            self.isRun = False 
            self.session.get(self.server_URL+'cmd=stopStream&name='+self.id+'&status=OK&module=stream')
            if self.save_video_flag: self.out.release()
            if(self.cap):
                if(self.cap.isOpened()): self.cap.release()
        except:
            self.session.get(self.server_URL+'cmd=stopStream&name='+self.id+'&status=error&module=stream')
            self.log.error("Unexpected error while Cam stopping")
            print(sys.exc_info()[0]) 
        self.log.info("videoCapture exit done")

# if __name__ == "__main__":
#     print("start")
#
#     log = logging.getLogger('app')
#     log.setLevel(logging.DEBUG)
#     f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)
#     ch.setFormatter(f)
#     log.addHandler(ch)
#
#     streams = []
#     with open('config/Stream_file_39.yaml', encoding='utf-8') as f:
#         configStream = yaml.load(f, Loader=yaml.FullLoader)
#     with open('config/Gpu_device0.yaml', encoding='utf-8') as f:
#         configGpu = yaml.load(f, Loader=yaml.FullLoader)
#     device_id = 0
#     cam_id = 'file_39'
#     #                           camConfig, gpuConfig, device_id, cam_id, log
#     streams.append(VideoCapture(configStream, configGpu, device_id, cam_id, log))
#     time.sleep(5)
#     if len(streams) > 0:
#         while True:
#             try:
#                 for stream in streams:
#                     print("stream", stream.cur_frame_cnt, stream.totalFrames)
#                     frame = stream.read()
#                     if frame.any():
#                         cv2.imshow("preview", frame)
#                         print("dd",)
#                         if stream.proceedTime[0]:
#                             print("fps", 1.0/(stream.proceedTime[0]))
#                         if stream.proceedTime[1]:
#                             print("fps", 1.0/(stream.proceedTime[1]))
#                     else:
#                         print("skip frame")
#                 key = cv2.waitKey(2)
#                 if key & 0xFF == ord('q'): break
#             except:
#                 print("stop by exception")
#                 print(sys.exc_info())
#                 break
#     for stream in streams:
#         stream.kill()
#     cv2.destroyAllWindows()
#
#   #  read  time =  0.015438318252563477
#   #  detect  time = 1.2223217487335205
#   #  proceed  time = 1.21510648727417

#    features 0.11533737182617188
#detection 0.0002684593200683594
#predict 0.002065420150756836
#track.update 0.007570743560791016
#trakes 0.0002079010009765625
#self.outFrame 0.0007374286651611328
#write 0.0034914016723632812
#track 0.12968873977661133
#TRT cams(1) proceed time =0.14783382415771484