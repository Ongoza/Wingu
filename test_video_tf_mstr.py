import os, yaml
import time
import numpy as np
import tensorflow as tf
import cv2
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import math

import videoCapture

writeVideo_flag = True
show_fh_fw = (416, 512)
conf_thresh = 0.4
nms_thresh = 0.6
video_path = 'video/39.avi'
video_path1 = 'video/43.avi'
model_name = "y4"
HEIGHT, WIDTH = (416, 416)

path_root = os.path.dirname(os.path.abspath(__file__))
#cfg_track_path = 'cfg/deep_sort.yaml'
#class_names = load_class_names('data/coco.names')
# output_video_size = 416
border_line = [(0, 400), (1200, 400)]
path_track = 20 # how many frames in path are saves
detector = YOLOv4(
    input_shape=(HEIGHT, WIDTH, 3),
    anchors=YOLOV4_ANCHORS,
    num_classes=80,
    training=False,
    yolo_max_boxes=64,
    yolo_iou_threshold=0.5,
    yolo_score_threshold=0.5,
)
detector.load_weights("models/yolov4.h5")

border_lines = {'border1':[[0, 100], [312, 104]]}

def drawBorderLines(frame):
    for b in border_lines:
        a = border_lines[b][0]
        b = border_lines[b][1]
        length = 40
        vX0 = b[0] - a[0]; vY0 = b[1] - a[1]
        mag = math.sqrt(vX0*vX0 + vY0*vY0)
        vX = vX0 / mag; vY = vY0 / mag
        temp = vX; vX = -vY; vY = temp
        z0 = (int(a[0]+vX0/2), int(a[1]+vY0/2))
        z1 = (int(a[0]+vX0/2 - vX * length), int(a[1] +vY0/2- vY * length))
        cv2.line(frame, tuple(a), tuple(b), (255, 255, 0), 2)
        cv2.arrowedLine(frame, z0, z1, (0, 255, 0), 2)
        cv2.putText(frame, "Out", z1, 0, 1, (0, 255, 0), 1)
    return frame

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def track_intersection_angle(A,B):   
    res = []
    for key in border_lines:
        C = np.array(border_lines[key][0])
        D = np.array(border_lines[key][1])
        if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):            
            v0 = np.array(B) - np.array(A)
            v1 = np.array(D) - np.array(C)
            angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
            if (angle > 0):
                res.append(key)
    return res

if __name__ == "__main__":    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    device = None
    if gpus:
        print("is cuda available", gpus)
        device = gpus[0].name
    with open('config/GPU_default.yaml') as f:    
        defaultConfig = yaml.load(f, Loader=yaml.FullLoader)    
    gpu = GpuDevice("test", device, defaultConfig, log)
    gpu.startCam('config/Stream_default.yaml')
    cams = [

        ]    
    print("frame img size", im_width, im_height)
    if writeVideo_flag:
        outFile = "mstr0_tf.avi"
        outFile1 = "mstr1_tf.avi"
        print("Save out video to file " + outFile)
        outs = [cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'XVID'), 10, show_fh_fw),
              cv2.VideoWriter(outFile1, cv2.VideoWriter_fourcc(*'XVID'), 10, show_fh_fw)]
        # out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'MP4V'), 10, show_fh_fw)

# variables
    counter = 0
    skip_counter = 4 
    cnt_people_in = {}
    cnt_people_out = {}
    path_track = 20 # how many frames in path are saves
    start = time.time()

# main loop
    while True:        
        r, frame = cap.read()
        r1, frame1 = cap1.read()
        if (not r):
            print("skip frame ", skip_counter)
            skip_counter -= 1
            if (skip_counter > 0): continue
            else: break
        start = time.time()
        counter += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame_sm = cv2.resize(frame, (HEIGHT, WIDTH))
        frame_sm1 = cv2.resize(frame1, (HEIGHT, WIDTH))
        # convert numpy opencv to tensor
        frames = np.stack([frame_sm, frame_sm1], axis=0)        
        frames_tf = tf.convert_to_tensor(frames, dtype=tf.float32, dtype_hint=None, name=None)/ 255.0
        print(frames_tf.shape)
        # shape=(416, 416, 3), dtype=float32)
        # frame_tz = tf.image.resize(frame_tz, (HEIGHT, WIDTH))
        #frames_tz = tf.expand_dims(frame_tz, axis=0)
        boxes, scores, classes, valid_detections = detector.predict(frames_tf)
        # print("boxes", boxes.shape)
        # boxes (1, 64, 4)
        for j in range(len(frames)):
            self.cam.update(boxes[j], scores[j], classes[j], frames[j]) 

        print("fps",round(1./(time.time()-start), 2))
        start = time.time()
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): break

    cap.release()    
    cap1.release()    
    if writeVideo_flag:
        for out in outs:
            out.release()        
    cv2.destroyAllWindows()
    print("Model: "+model_name+". People in: "+str(len(cnt_people_in))+", out: "+str(len(cnt_people_out)))
