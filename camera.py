import os, sys, traceback
import time
import queue, threading
import asyncio
import logging
# import cv2
import numpy as np
import random
import requests
import json
# from server import log

class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        self._stopevent = threading.Event()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while not self._stopevent.isSet():
            if(self.cap):
                ret, frame = self.cap.read()
                if (ret): 
                    if (not self.q.empty()):
                        try: self.q.get_nowait()   # discard previous (unprocessed) frame
                        except queue.Empty: pass
                    self.q.put(frame)
    def read(self):
        return self.q.get()
    def exit(self):
        self._stopevent.set()
        if(self.cap):
            if(self.cap.isOpened()):self.cap.release()

class Camera(threading.Thread):
    def __init__(self, id, log):
        threading.Thread.__init__(self)
        self.id = id
        self.log = log
        self.url = ''
        self.skipFrame = 4
        self.cntFrame = 0
        self.allFrames = 0
        self.borders = []
        self.tgtResolution = []
        self.videoResResolution = []
        self.videoResPath = ''
        self.resultVideo = False
        self.displayVideo = False
        self._stopevent = threading.Event()
        self.killed = False
    
    def startVideo(self):
        print('start video: ', self.url)


    def kill(self):
        self.log.debug("kill camera")
        self._stopevent.set()
        self.killed = True

    def run(self):
        print("start camera")
        while not self._stopevent.isSet():
            # self.log.debug("hop")
            time.sleep(1)


if __name__ == "__main__":
    log = logging.getLogger('app')
    log.setLevel(logging.DEBUG)
    f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(f)
    log.addHandler(ch)

    cam = Camera("testCamera", log)
    cam.start()
    time.sleep(10)
    cam.kill()
    cam.join()




