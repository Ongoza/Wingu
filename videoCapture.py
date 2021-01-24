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
import aiosqlite
from requests_futures import sessions
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
from deep_sort import nn_matching
from deep_sort import preprocessing
import deep_sort.generate_detections as gdet
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

class VideoCapture:
    def __init__(self, camConfig, gpuConfig, device_id, cam_id, log, client=None, vc_device="/CPU:0"):
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
        self.db_path = "db.wingu.sqlite3"
        self.db_table_name = "intersetions"
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
            self.isFromFile = self.config['isFromFile']
            self.cap = cv2.VideoCapture(self.url)
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
                outFile = self.config['save_path']
                if outFile == '': outFile =  'video/'+str(self.id)+"_auto.avi"
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
                    self.q = queue.Queue()
                    t = threading.Thread(target=self._reader)
                    t.daemon = True
                    t.start()
                else:
                    self.totalFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - self.skip_frames
                self._stopevent = threading.Event()
                #if client is not None:
                #    client.send_json({'OK':["startStream", self.id]})
            else:
                print("VideoCapture encoder is None")
                return None
        except:
            self.log.debug("VideoStream Can not start Video Stream for " + camConfig)            
            print("VideoStream err types",type(traceback.print_exception(*sys.exc_info())), type(sys.exc_info())) 
            print("VideoStream err:", sys.exc_info())
            self.id = None

    async def ws_send_data(self, cmd):
        try:
            await self.session.get(self.server_URL+'cmd='+cmd)
            #future.result()
            #print("ok")
        except:
             pass

    #async def ws_send_data(self):
    #    try:
    #        data = self.outFrame.tobytes()
    #        print("VideoStream data=", len(data))
    #        for client in self.clients:
    #            try:
    #                await client.send_bytes(data)
    #            except:
    #                print("VideoStream ws_send_data client error")
    #                print(sys.exc_info())
    #                await client.send_json({"error":["VideoStream","can not send data"]})
    #    except:
    #        print("VideoStream ws_send_data error in except")
    #        print(sys.exc_info())

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

    def stopGetStream(self, client):
          self.clients = []
          self.display_video_flag = False

    async def startGetStream(self, client):
        print("VideoCapture start stream video")
        if client not in self.clients:
            self.clients.append(client)
            self.display_video_flag = True
            print("len of clients: ", len(self.clients) )
            try:
                await client.send_json({'OK':["startGetStream", self.id]})
            except:
                print("VideoCapture exception in startGet Stream")
                await client.send_json({"error":["startGetStream", self.id, "exception on VideoCapture"]})
        else:
            await client.send_json({"error":["startGetStream", self.id, "one more times"]})            

    def send_cur_frame(self):
        return self.outFrame   

    async def save_statistic(self, borders_arr):
        print("VideoCapture start save stat in stream")
        try:
            for borders in borders_arr:
                for item in borders:
                    sql = f'INSERT INTO {self.db_table_name}(border, stream_id, time) VALUES("{item}", "{self.id}", {int(time.time())})'
                    print("sql", sql)
                    async with aiosqlite.connect(self.db_path) as db:
                        await db.execute(sql)
                        await db.commit()
        except:
            self.log.debug("VideoCapture Error save data")

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
                else:
                    #self.log.debug("Skip frame")
                    frame = self.read()
                self.proceedTime[0] = time.time() - start
                return frame
            else:
                self.kill()
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

    async def track(self, box, score, cl, frame):
        try:  
            start = time.time()
            res = []
            #self.log.debug("Track Cam "+str(self.id)+"  frame="+ str(self.proceed_frames_cnt))
            boxs = []
            confs = []
            for i in range(len(box)): 
               if score[i] > 0:
                   if cl[i] == 0:
                     boxs.append((np.array(box[i])*self.img_size))
                     confs.append(score[i])
            if(len(boxs)):
                #self.log.debug("track boxs "+ str(len(boxs)))
                with tf.device(self.vc_device):
                    features = self.encoder(frame, boxs)
                detections = [Detection(bbox, conf, feature) for bbox, conf, feature in zip(boxs, confs, features)] 
                self.tracker.predict()
                self.tracker.update(detections)
                # self.log.debug("videoCapture traks "+ str(len(self.tracker.tracks)))
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
                        if lst_intrsc:
                            #border_line
                            if(not hasattr(track, 'calculated')):
                                #cnt_people_in[track.track_id] = 0
                                track.calculated = "in_"
                                track.color = (52, 235, 240)
                                self.log.debug("intersection!! "+ self.id +" "+ str(len(lst_intrsc)))
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
                        # print("[track.xy]", [track.xy])
                        # cv2.polylines(frame_sm, [np.array(track.xy)], False, clr, 3)
                    else: 
                        track.xy = [xy]
                    if(self.isDrow):    
                        txy =  tuple(xy)
                        cv2.circle(frame, txy, 5, clr, -1)
                        # cv2.rectangle(frame_sm, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), clr, 1)
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
            if res:
                 await self.save_statistic(res)
            if self.clients:
                self.log.debug("save")
                if(not self.save_video_flag):
                    frame = self.drawBorderLines(frame)
                    cv2.putText(frame, "Frame: "+str(self.cur_frame_cnt), (10, 340), 0, 0.4, (255, 255, 0), 1)
                    # cv2.putText(frame, "People in: "+str(len(self.cnt_people_in)), (10, 360), 0, 0.4, (52, 235, 240), 1)
                    #print(self.save_video_res)
                    frame = cv2.resize(frame, self.save_video_res)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ret2, frame_jpg = cv2.imencode(".jpg", frame)
                if(ret2):
                    self.outFrame = np.copy(frame_jpg)
                    print("self.outFrame=", len(self.outFrame))
                    # self.outFrame = iWeb.tobytes()
                    #with open("video/dd__00_1.jpg",'wb') as f:
                    #    f.write(res)
                    await self.ws_send_data("frame")
            self.proceedTime[1] = time.time() - start
        except:
            print(sys.exc_info())

    def kill(self):
        try:
            self._stopevent.set()
            self.isRun = False 
            self.session.get(self.server_URL+'cmd=stopStream&name='+self.id)
            if self.save_video_flag: self.out.release()
            if(self.cap):
                if(self.cap.isOpened()): self.cap.release()
        except:
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