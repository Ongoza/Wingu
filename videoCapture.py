import os, sys, traceback
import time
import queue, threading
import logging
import random
import json
import math

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
    def __init__(self, camConfig, gpuConfig):        
        #self.totalFrames = 19
        self.id = camConfig['id']
        self.url = camConfig['url']
        self.cap = cv2.VideoCapture(self.url)

        self.totalFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.lastFrame = 100
        self.config = gpuConfig
        self.save_video_res = None
        self.frame_res = (gpuConfig['img_size'],gpuConfig['img_size'])
        self.skip_frames = gpuConfig['skip_frames']
        self.isDrow = False
        self.outFrame = np.array([])
        self.path_track = gpuConfig['path_track']
        self.save_video_flag = gpuConfig['save_video_flag']
        self.display_video_flag = gpuConfig['display_video_flag']
        if(gpuConfig['save_video_flag'] or gpuConfig['display_video_flag']):
            self.isDrow = True
            self.save_video_res = tuple(camConfig['save_video_res'])
        self.borders = camConfig['borders']
        self.out = None
        if self.save_video_flag:            
            outFile = self.url + '_res.avi'
            print("Save out video to file " + outFile)
            self.out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'XVID'), 5, self.save_video_res)
        self.tracker = Tracker(nn_matching.NearestNeighborDistanceMetric("cosine", gpuConfig['max_cosine_distance'], gpuConfig['nn_budget']))
        self.cur_frame = 0
        self.cnt_people_in = {}
        self.q = queue.Queue()
        self._stopevent = threading.Event()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while not self._stopevent.isSet():
            if self.cur_frame >= self.totalFrames:
               self.exit()
            else:
                if(self.cap):
                    ret, frame = self.cap.read()
                    if (ret):
                        frame = cv2.resize(frame, self.frame_res)
                        if (not self.q.empty()):
                            try: self.q.get_nowait()   # discard previous (unprocessed) frame
                            except queue.Empty: pass
                        self.q.put(frame)
                self.cur_frame += 1
                

    def read(self):
        return self.q.get()

    def track_intersection_angle(self, xy0, xy1):
        res = []
        for key in self.borders:
            s = np.vstack([xy0, xy1, self.borders[key][0], self.borders[key][1]])        # s for stacked
            h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
            l1 = np.cross(h[0], h[1])           # get first line
            l2 = np.cross(h[2], h[3])           # get second line
            x, y, z = np.cross(l1, l2)          # point of intersection
            if z != 0:                          # lines are parallel
                v0 = np.array(xy1) - np.array(xy0)
                v1 = np.array(self.borders[key][1]) - np.array(self.borders[key][0])
                angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
                if (angle > 0):
                    res.append(key)
            return res

    def add_intersectio_event(self, border_names, id):
        print(border_names, id)

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

    def track(self, detections, frame):
        self.tracker.predict()
        self.tracker.update(detections)
        for track in self.tracker.tracks:
            if(not track.is_confirmed() or track.time_since_update > 1):
                # if(track.time_since_update > life_frame_limit): track.state = 3 # if missed to long than delete id
                continue 
            bbox = track.to_tlwh()
            x1y1 = (int(bbox[1]+(bbox[3] - bbox[1])/2), int(bbox[0]+(bbox[2] - bbox[0])/2))
            clr = (255, 255, 0) # default color
            track_name = str(track.track_id) # default name
            if(hasattr(track, 'xy')):
                # detect direction
                #track_line = LineString([track.xy[0], x1y1])
                #if(track_line.intersection(border_line_str)):
                lst_intrsc = self.track_intersection_angle(track.xy[0], x1y1)
                if(any(lst_intrsc)):
                    #border_line
                    # print("intersection!!", track_name)
                    if(not hasattr(track, 'calculated')):
                        #if(border_line_a * (x1y1[1] - border_line[0][1]) -  border_line_b * (x1y1[0] - border_line[0][0])) > 0:
                        self.cnt_people_in[track.track_id] = 0
                        track.calculated = "in_" + str(len(self.cnt_people_in)) + "_"
                        track.color = (52, 235, 240)
                        self.add_intersectio_event(lst_intrsc, track.track_id)
                        print("inresection", track.track_id)
                        #else: # 
                        #    cnt_people_out[track.track_id] = 0
                        #    track.calculated = "out_" + str(len(cnt_people_out)) + "_"
                        #    track.color = (0, 255, 0)
                        track.cross_cnt = self.path_track
                    clr = track.color
                # else:
                        
                if(hasattr(track, 'calculated')):
                    clr = track.color
                    track_name = track.calculated  + track_name
                    track.cross_cnt -= 1
                    if(track.cross_cnt < 1): track.state = 3 # delete from track list
                track.xy = np.append(track.xy, [x1y1], axis=0)
                track.xy = track.xy[-self.path_track:]
                # cv2.arrowedLine(frame,(track.x1[0], track.y1[0]),(x1, y1),(0,255,0),4)
                if(self.isDrow):
                    cv2.polylines(frame, [track.xy], False, clr, 3)
            else: track.xy = np.array([x1y1])
            if(self.isDrow):
                cv2.circle(frame, x1y1, 5, clr, -1)
                cv2.rectangle(frame, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), clr, 1)
                # cv2.putText(frame, str(track.track_id),(int(bbox[1]), int(bbox[0])),0, 5e-3 * 200, (0,255,0),2)
                cv2.putText(frame, track_name, x1y1, 0, 0.4, clr, 1)
            if(self.save_video_flag):
                self.drawBorderLines(frame)
                #cv2.putText(frame, "FPS: "+str(round(1./(time.time()-start), 2))+" frame: "+str(counter), (10, 340), 0, 0.4, (255, 255, 0), 1)
                cv2.putText(frame, "People in: "+str(len(self.cnt_people_in)), (10, 360), 0, 0.4, (52, 235, 240), 1)
                frame = cv2.resize(frame,self.save_video_res)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.out.write(frame)
            if(self.display_video_flag):
                if(not self.save_video_flag):
                    frame = self.drawBorderLines(frame)
                    #cv2.putText(frame, "FPS: "+str(round(1./(time.time()-start), 2))+" frame: "+str(counter), (10, 340), 0, 0.4, (255, 255, 0), 1)
                    cv2.putText(frame, "People in: "+str(len(self.cnt_people_in)), (10, 360), 0, 0.4, (52, 235, 240), 1)
                    print(self.save_video_res)
                    frame = cv2.resize(frame, self.save_video_res)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.outFrame = frame
                #cv2.putText(frame, " out: "+str(len(cnt_people_out)), (43, 376), 0, 0.4, (0, 255, 0), 1)
                # print("end frame")
                #cv2.imshow("preview", frame)

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
    stream = []
    stream.append(VideoCapture("id", "video/39.avi", gpusManager.defaultBorders, gpusManager.defaultConfig))
    while True:
        print("stream", stream[0].cur_frame, stream[0].totalFrames)
        if stream[0]._stopevent.isSet(): 
            print("stop ddd")
            break 
        else: 
            frame = stream[0].read()
            #cv2.imshow("preview", frame)
            key = cv2.waitKey(10)
            if key & 0xFF == ord('q'): break
    print("stop 10")
    stream[0].exit()
    print("stop 11")
    cv2.destroyAllWindows()
    print("stop 12")
    cv2.waitKey(1)
    print("stop 13")