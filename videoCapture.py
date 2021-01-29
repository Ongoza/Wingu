import os, sys, traceback
import time
import queue, threading
import logging
import random
import json
import math
import yaml
import numpy as np
import cv2
import asyncio
from requests_futures import sessions
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
from deep_sort import nn_matching
from deep_sort import preprocessing
import deep_sort.generate_detections as gdet
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

class VideoCaptureStream:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                global is_frame
                is_frame = False
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def kill(self):
        self.q.get_nowait()
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
        self.cur_frame_cnt = 0
        self.proceed_frames_cnt = 0
        self.proceedTime = [0, 0]
        self.session = sessions.FuturesSession(max_workers=2)
        self.outFrame = np.array([])
        self.isDrow = False
        self.clients = []
        self.intersections = []
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
            self.img_size = int(gpuConfig['img_size'])
            # self.lastFrame = 100
            self.max_hum_w = int(self.img_size/4) 
            self.GPUconfig = gpuConfig
            self.frame_res = (self.img_size, self.img_size)
            self.skip_frames = int(self.config['skip_frames'])
            self.batch_size = int(self.config['batch_size'])
            self.path_track = int(self.config['path_track'])
            self.save_video_flag = self.config['save_video_flag']
            self.display_video_flag = self.config['display_video_flag']
            if(self.config['save_video_flag'] or self.config['display_video_flag']):
                self.isDrow = True
                self.save_video_res = tuple(self.config['save_video_res'])
            self.borders = self.config['borders']
            self.out = None
            if self.save_video_flag:            
                outFile = self.config['save_path'] +"_"+ str(self.startTime)+".avi"
                if outFile == '': outFile =  'video/'+str(self.id)+"_"+ str(self.startTime)+"_auto.avi"
                self.log.debug("Save out video to file " + outFile)
                self.out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'XVID'), 5, self.save_video_res)
            print("VideoCapture stream start load encoder")
            with tf.device(self.vc_device):
                self.encoder = gdet.create_box_encoder(os.path.join('models',self.config['encoder_filename']), batch_size=self.batch_size, device=self.vc_device)
            print("VideoCapture stream start load tracker")
            if self.encoder is not None:
                self.tracker = Tracker(nn_matching.NearestNeighborDistanceMetric("cosine", self.config['max_cosine_distance'], None))
                print("VideoCapture stream tracker ok")
                # self.cnt_people_in = {}
                if not self.isFromFile:
                    self.q = VideoCaptureStream(self.url)
                else:
                    self.cap = cv2.VideoCapture(self.url)
                    self.totalFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 10
                self._stopevent = threading.Event()
                #if client is not None:
                #    client.send_json({'OK':["startStream", self.id]})
            else:
                print("VideoCapture encoder is None")
                return None
            self.session.get(self.server_URL+'cmd=startStream&name='+self.id+'&status=OK')
        except:
            self.session.get(self.server_URL+'cmd=startStream&name='+self.id+'&status=error')
            self.log.debug("VideoStream Can not start Video Stream for " + camConfig)            
            print("VideoStream err types",type(traceback.print_exception(*sys.exc_info())), type(sys.exc_info())) 
            print("VideoStream err:", sys.exc_info())
            self.id = None

    def writeVideo(self):
       self.out.write(self.outFrame)

    def get_status(self):
        status = {}
        try:
            status["id"] = self.id
            status["startTime"] = self.startTime
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
        res = self.intersections.copy()
        self.intersections = []
        return res 

    def get_cur_frame(self):
        return self.outFrame   

    def read(self):
        start = time.time()
        if self.isFromFile:
            # print("proceed_frames_cnt="+str(self.proceed_frames_cnt))
            if self.proceed_frames_cnt < self.totalFrames:
                ret, frame = self.cap.read()
                self.cur_frame_cnt += 1
                self.proceed_frames_cnt = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if self.skip_frames:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.proceed_frames_cnt + self.skip_frames)
                if (ret):
                   frame = cv2.resize(frame, self.frame_res)
                   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                   if self.type == 1:
                       print("start cut area")
                else:
                    # self.log.debug("Skip frame")
                    frame = self.read()
                self.proceedTime[0] = time.time() - start
                return frame
            else:
                self.session.get(self.server_URL+'cmd=stopStream&name='+self.id+'&status=OK&module=stream')
                time.sleep(1)
                self.kill()
        else:
            start = time.time()
            frame = self.q.read()
            self.cur_frame_cnt += 1
            self.proceed_frames_cnt += 1
            frame = cv2.resize(frame, self.frame_res)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.type == 1:
               print("start cut area")
               #https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
            self.proceedTime[0] = time.time() - start
            return frame

    def ccw(self,A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    # Return true if line segments AB and CD intersect
    def track_intersection_angle(self,A,B):   
        res = {}
        for key in self.borders:
            C = np.array(self.borders[key][0])
            D = np.array(self.borders[key][1])
            if self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D):            
                v0 = np.array(B) - np.array(A)
                v1 = np.array(D) - np.array(C)
                angle = np.math.atan2(np.linalg.det([v0,v1]), np.dot(v0,v1))
                if (angle > 0):
                    res[key] = 1
                else:
                    res[key] = 0
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

    def track(self, box, score, cl, frame):
        try:
            start = time.time()
            if self.type == 0:
                res = []
                boxs = []
                confs = []
                for i in range(len(box)): 
                   if score[i] > 0:
                       if cl[i] == 0:
                         boxs.append((np.array(box[i])*self.img_size))
                         confs.append(score[i])
                if(len(boxs)):
                    with tf.device(self.vc_device):
                        features = self.encoder(frame, boxs)
                    # print("features", type (features), features[0].shape )
                    detections = [Detection(bbox, conf, feature) for bbox, conf, feature in zip(boxs, confs, features)] 
                    self.tracker.predict()
                    self.tracker.update(detections)
                    for track in self.tracker.tracks:
                        if(not track.is_confirmed() or track.time_since_update > 1):
                            # if(track.time_since_update > life_frame_limit): track.state = 3 # if missed to long than delete id
                            continue 
                        xy = track.mean[:2].astype(np.int)# tuple(())
                        clr = (255, 255, 0) # default color
                        track_name = str(track.track_id) # default name
                        if(hasattr(track, 'xy')):
                            lst_intrsc = self.track_intersection_angle(track.xy[0], xy)
                            if lst_intrsc.keys():
                                #border_line
                                if(not hasattr(track, 'calculated')):
                                    #cnt_people_in[track.track_id] = 0
                                    track.calculated = "in_"
                                    self.intersections.append(lst_intrsc)
                                    track.color = (52, 235, 240)
                                    self.log.debug("intersection!! "+ self.id)
                                    res.append(lst_intrsc)
                                    track.cross_cnt = self.path_track
                            if(hasattr(track, 'calculated')):
                                clr = track.color
                                track_name = track.calculated  + track_name
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
                self.drawBorderLines(frame)
                cv2.putText(frame, "Frame: "+str(self.cur_frame_cnt), (10, 340), 0, 0.4, (255, 255, 0), 1)
                frame = cv2.resize(frame,self.save_video_res)
                self.outFrame = np.copy(frame)
                # print('self.outFrame', self.outFrame)
                if self.save_video_flag:
                    self.writeVideo()
                self.proceedTime[1] = time.time() - start
            else:
                print("start calculate people in area")
        except:
            print(sys.exc_info())

    def kill(self):
        try:
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

if __name__ == "__main__":
    print("start")

    log = logging.getLogger('app')
    log.setLevel(logging.DEBUG)
    f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(f)
    log.addHandler(ch)
    
    streams = []
    with open('config/Stream_file_39.yaml', encoding='utf-8') as f:    
        configStream = yaml.load(f, Loader=yaml.FullLoader)  
    with open('config/Gpu_device0.yaml', encoding='utf-8') as f:    
        configGpu = yaml.load(f, Loader=yaml.FullLoader)
    device_id = 0
    cam_id = 'file_39'
    #                           camConfig, gpuConfig, device_id, cam_id, log
    streams.append(VideoCapture(configStream, configGpu, device_id, cam_id, log))
    time.sleep(5)
    if len(streams) > 0:
        while True:
            try:
                for stream in streams:
                    print("stream", stream.cur_frame_cnt, stream.totalFrames)
                    frame = stream.read()
                    if frame.any():
                        cv2.imshow("preview", frame)
                        print("dd",)
                        if stream.proceedTime[0]:
                            print("fps", 1.0/(stream.proceedTime[0]))
                        if stream.proceedTime[1]:
                            print("fps", 1.0/(stream.proceedTime[1]))
                    else:
                        print("skip frame")
                key = cv2.waitKey(2)
                if key & 0xFF == ord('q'): break
            except:
                print("stop by exception")
                print(sys.exc_info()) 
                break
    for stream in streams:
        stream.kill()
    cv2.destroyAllWindows()