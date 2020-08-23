import time
import threading
import asyncio
import logging
# from server import log

class Camera(threading.Thread):
    def __init__(self, id, log):
        threading.Thread.__init__(self)
        self.id = id
        self.log = log
        self._stopevent = threading.Event()
        self.killed = False
    
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




