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
import gpusManager

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
from deep_sort import nn_matching
from deep_sort import preprocessing
import deep_sort.generate_detections as gdet
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

class VideoCapture:
    def __init__(self, camConfigFile, gpuConfig, device_id, log):        
        print("start init stream object")
        self.totalFrames = 0
        self.cur_frame_cnt = 0
        self.log = log
        try:
            with open(camConfigFile) as f:    
                self.config = yaml.load(f, Loader=yaml.FullLoader)
            self.id = self.config['id']
            self.device_id = device_id
            self.url = self.config['url']
            self.isFromFile = self.config['isFromFile']
            self.log.debug("start init stream object")            
            self.cap = cv2.VideoCapture(self.url)
            self.img_size = gpuConfig['img_size']
            # self.lastFrame = 100
            self.max_hum_w = int(self.img_size/4) 
            self.GPUconfig = gpuConfig
            self.save_video_res = None
            self.frame_res = (self.img_size, self.img_size)
            self.skip_frames = self.config['skip_frames']
            self.batch_size = self.config['batch_size']
            self.isDrow = False
            self.proceed_frames_cnt = 0
            self.proceedTime = [0, 0]
            self.outFrame = np.array([])
            self.path_track = self.config['path_track']
            self.save_video_flag = self.config['save_video_flag']
            self.display_video_flag = self.config['display_video_flag']
            if(self.config['save_video_flag'] or self.config['display_video_flag']):
                self.isDrow = True
                self.save_video_res = tuple(self.config['save_video_res'])
            self.borders = self.config['borders']
            self.out = None
            if self.save_video_flag:            
                outFile =  self.config['save_path']
                self.log.debug("Save out video to file " + outFile)
                self.out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'XVID'), 5, self.save_video_res)
            self.encoder = gdet.create_box_encoder(self.config['encoder_filename'], batch_size=self.batch_size)
            self.tracker = Tracker(nn_matching.NearestNeighborDistanceMetric("cosine", self.config['max_cosine_distance'], None))
            # self.cnt_people_in = {}
            if not self.isFromFile:
                self.q = queue.Queue()
                t = threading.Thread(target=self._reader)
                t.daemon = True
                t.start()
            else:
                self.totalFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self._stopevent = threading.Event()
            
        except:
            self.log.debug("Can not start Vidoe Stream for " + camConfigFile)            
            print("err types",type(traceback.print_exception(*sys.exc_info())), type(sys.exc_info())) 
            print("VideoStream err:", sys.exc_info())
            self.id = None

    def get_status(self):
        return {
            "id":self.id, 
            "save_video_res":self.save_video_res, 
            "skip_frames":self.skip_frames,
            "display_video_flag":self.display_video_flag,
            "cur_frame_cnt": self.cur_frame_cnt,
            "proceed_frames_cnt": self.proceed_frames_cnt,
            "totalFrames": self.totalFrames
            }

    def get_cur_frame(self):
        return self.outFrame   

    # read frames as soon as they are available, keeping only most recent one
    # 0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
    # 1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
    def _reader(self):
        while not self._stopevent.isSet():
            if(self.cap):
                ret, frame = self.cap.read()
                if (ret):
                    if (not self.q.empty()):
                        try: self.q.get_nowait()   # discard previous (unprocessed) frame
                        except queue.Empty: pass
                    frame = cv2.resize(frame, self.frame_res)
                    self.q.put(frame)
                # else: print("skip frame", self.cur_frame)
                self.cur_frame_cnt += 1

    def read(self):
        start = time.time()
        if self.isFromFile:
            if self.cur_frame_cnt < self.totalFrames:
                ret, frame = self.cap.read()
                self.cur_frame_cnt += 1
                self.proceed_frames_cnt = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.proceed_frames_cnt + self.skip_frames)
                if (ret):
                   frame = cv2.resize(frame, self.frame_res)
                else:
                    #self.log.debug("Skip frame")
                    frame = self.read()
                self.proceedTime[0] = round(time.time() - start, 2)
                return frame
            else:
                self.exit()
        else:
            return self.q.get()

    def ccw(self,A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    

    # Return true if line segments AB and CD intersect
    def track_intersection_angle(self,A,B):   
        res = []
        for key in self.borders:
            C = np.array(self.borders[key][0])
            D = np.array(self.borders[key][1])
            if self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D):            
                v0 = np.array(B) - np.array(A)
                v1 = np.array(D) - np.array(C)
                angle = np.math.atan2(np.linalg.det([v0,v1]), np.dot(v0,v1))
                if (angle > 0):
                    res.append(key)
        return res

    def add_intersectio_event(self, border_names, id):
        self.log.debug(border_names, id)

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
        start = time.time()
        # self.log.debug("track "+ str(len(box)) +" "+ str(len(score)) +" "+ str(len(cl)))
        boxs = []
        confs = []
        for i in range(len(box)): 
           if score[i] > 0:
               if cl[i] == 0:
                 boxs.append((np.array(box[i])*self.img_size))
                 confs.append(score[i])
        if(len(boxs)):
            # self.log.debug("track 2")
            features = self.encoder(frame, boxs)
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
                # self.log.debug("track "+ track_name)
                if(hasattr(track, 'xy')):
                    lst_intrsc = self.track_intersection_angle(track.xy[0], xy)
                    if(any(lst_intrsc)):
                        #border_line
                        if(not hasattr(track, 'calculated')):
                            #cnt_people_in[track.track_id] = 0
                            track.calculated = "in_"
                            track.color = (52, 235, 240)
                            self.log.debug("intersection!! "+ track_name +" "+ self.id)
                            track.cross_cnt = self.path_track
                    if(hasattr(track, 'calculated')):
                        clr = track.color
                        track_name = track.calculated  + track_name
                        track.cross_cnt -= 1
                        if(track.cross_cnt < 1): track.state = 3 # delete from track list
                    track.xy.append(xy)
                    if len(track.xy) > self.path_track:
                        track.xy = track.xy[-self.path_track:]
                    # print("[track.xy]", [track.xy])
                    # cv2.polylines(frame_sm, [np.array(track.xy)], False, clr, 3)
                else: 
                    track.xy = [xy]
                if(self.isDrow):    
                    txy =  tuple(xy)
                    cv2.circle(frame, txy, 5, clr, -1)
                    #cv2.rectangle(frame_sm, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), clr, 1)
                    # cv2.putText(frame, str(track.track_id),(int(bbox[1]), int(bbox[0])),0, 5e-3 * 200, (0,255,0),2)
                    cv2.putText(frame, track_name, txy, 0, 0.4, clr, 1)
        # self.log.debug("--" + str(self.display_video_flag))
        if self.save_video_flag:
            self.drawBorderLines(frame)
            cv2.putText(frame, "Frame: "+str(self.cur_frame_cnt), (10, 340), 0, 0.4, (255, 255, 0), 1)
            # cv2.putText(frame, "People in: "+str(len(self.cnt_people_in)), (10, 360), 0, 0.4, (52, 235, 240), 1)
            frame = cv2.resize(frame,self.save_video_res)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.out.write(frame)
        if self.display_video_flag:
            #self.log.debug("save")
            if(not self.save_video_flag):
                frame = self.drawBorderLines(frame)
                cv2.putText(frame, "Frame: "+str(self.cur_frame_cnt), (10, 340), 0, 0.4, (255, 255, 0), 1)
                # cv2.putText(frame, "People in: "+str(len(self.cnt_people_in)), (10, 360), 0, 0.4, (52, 235, 240), 1)
                #print(self.save_video_res)
                frame = cv2.resize(frame, self.save_video_res)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #ret2, iWeb = cv2.imencode(".jpg", frame)
            #if(ret2): self.outFrame = iWeb.tobytes()
            #else: self.log.error("Can't convert img")
            self.outFrame = np.copy(frame)
        self.proceedTime[1] = time.time() - start
        

    def exit(self):
        self._stopevent.set()
        self.isRun = False
        try:
            if self.save_video_flag: self.out.release()
            if(self.cap):
                if(self.cap.isOpened()): self.cap.release()
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise 
        print("videoCapture exit done")

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
    with open('config/GPU_default.yaml') as f:    
        config = yaml.load(f, Loader=yaml.FullLoader)  
    # videoCapture.VideoCapture(camConfigFileName, config, device, log)
    streams.append(VideoCapture("config/Stream_39.yaml", config, 0, log))
    time.sleep(5)
    if len(streams) > 0:
        while True:
            try:
                print("stream", streams[0].cur_frame_cnt, streams[0].totalFrames)
                if streams[0]._stopevent.isSet(): 
                    print("stop ddd")
                    break 
                else:
                    frame = streams[0].read()
                    # frame = streams[0].get_cur_frame()
                    if frame.any():
                        cv2.imshow("preview", frame)
                    else:
                        print("skip frame")
                    key = cv2.waitKey(10)
                    if key & 0xFF == ord('q'): break
            except:
                print("stop by exception")
                break
        print("stop 10")

    for stream in streams:
        stream.exit()
    print("stop 11")
    cv2.destroyAllWindows()
    print("stop 12")
    cv2.waitKey(1)
    print("stop 13")