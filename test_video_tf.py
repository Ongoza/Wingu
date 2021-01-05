import os, yaml
import time
import numpy as np
import tensorflow as tf
import cv2
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import math

from deep_sort import nn_matching
import deep_sort.generate_detections as gdet
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

writeVideo_flag = True
show_fh_fw = (416, 512)
#imgfile = 'data/dog.jpg'
#cfgfile = 'cfg/yolov4.cfg'
#weightfile = 'Model_data/yolov4_dark.weights'
conf_thresh = 0.4
nms_thresh = 0.6
video_path = 'video/39.avi'
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
    yolo_max_boxes=100,
    yolo_iou_threshold=0.5,
    yolo_score_threshold=0.5,
)
detector.load_weights("models/yolov4.h5")

max_cosine_distance = 0.2 # 03
encoder = gdet.create_box_encoder("models/mars-small128.pb", batch_size=64)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, None)
tracker = Tracker(metric)
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
    if gpus:
        print("is cuda available", gpus)

#  tracker
    #cfg_track = edict({})
    #if cfg_track_path is not None:
    #    assert(os.path.isfile(cfg_track_path))
    #with open(cfg_track_path, 'r') as fo:
    #    cfg_track.update(yaml.load(fo.read()))
    # print("cfg_track", cfg_track.DEEPSORT)

    encoder = gdet.create_box_encoder('models/mars-small128.pb', batch_size=64)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, None)
    tracker = Tracker(metric)
 
    # video parameters
    cap = cv2.VideoCapture(video_path)
    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("frame img size", im_width, im_height)

    if writeVideo_flag:
        outFile = video_path+"_tf.avi"
        print("Save out video to file " + outFile)
        out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'XVID'), 10, show_fh_fw)
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
        if (not r):
            print("skip frame ", skip_counter)
            skip_counter -= 1
            if (skip_counter > 0): continue
            else: break
        start = time.time()
        counter += 1

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_sm = cv2.resize(frame, (HEIGHT, WIDTH))
        # convert numpy opencv to tensor
        frame_tz = tf.convert_to_tensor(frame_sm, dtype=tf.float32, dtype_hint=None, name=None)
        # print(frame_tz)
        # shape=(416, 416, 3), dtype=float32)
        # frame_tz = tf.image.resize(frame_tz, (HEIGHT, WIDTH))
        frame_tz = tf.expand_dims(frame_tz, axis=0) / 255.0
        boxes, scores, classes, valid_detections = detector.predict(frame_tz)
        boxs = []
        confs = []
        for box, score, cl in zip(boxes[0].tolist(), scores[0].tolist(), classes[0].tolist()): 
            if score > 0:
                if cl == 0:
                    boxs.append((np.array(box)*416))
                    confs.append(score)
                    # cv2.rectangle(frame_sm,(box[0],box[1]), (box[2],box[3]), (255,255,0), 4)
                    #prob.append([score,cl])
        #print("boxs", boxs)
        # cv2.imwrite("video/frame.jpg", frame_sm)
        features = encoder(frame, boxs)
        #print(type(features[0]), type(boxs[0]))
        detections = [Detection(bbox, conf, feature) for bbox, conf, feature in zip(boxs, confs, features)] 
        # for d in  detections: print("d=", d.__dict__)
        tracker.predict()
        tracker.update(detections)
        for track in tracker.tracks:
            # print("track", track)
            if(not track.is_confirmed() or track.time_since_update > 1):
                # if(track.time_since_update > life_frame_limit): track.state = 3 # if missed to long than delete id
                continue
            # x1y1 = (int(bbox[1]+(bbox[3] - bbox[1])/2), int(bbox[0]+(bbox[2] - bbox[0])/2))
            xy = track.mean[:2].astype(np.int)# tuple(())
            clr = (255, 255, 0) # default color
            track_name = str(track.track_id) # default name
            if(hasattr(track, 'xy')):
                lst_intrsc = track_intersection_angle(track.xy[0], xy)
                if(any(lst_intrsc)):
                    #border_line
                    if(not hasattr(track, 'calculated')):
                        cnt_people_in[track.track_id] = 0
                        track.calculated = "in_" + str(len(cnt_people_in)) + "_"
                        track.color = (52, 235, 240)
                        print("intersection!!", track_name, track.track_id)
                        track.cross_cnt = path_track
                clr = track.color
                track.xy.append(xy)
                if len(track.xy) > path_track:
                    track.xy = track.xy[-path_track:]
                # print("[track.xy]", [track.xy])
                cv2.polylines(frame_sm, [np.array(track.xy)], False, clr, 3)
            else: 
                track.color = clr
                track.xy = [xy]
            
            txy =  tuple(xy)
            cv2.circle(frame_sm, txy, 5, clr, -1)
            #cv2.rectangle(frame_sm, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), clr, 1)
            # cv2.putText(frame, str(track.track_id),(int(bbox[1]), int(bbox[0])),0, 5e-3 * 200, (0,255,0),2)
            cv2.putText(frame_sm, track_name, txy, 0, 0.4, clr, 1)
        # print("--", counter. fps)
        drawBorderLines(frame_sm)
        frame_out = cv2.resize(frame_sm, (512, 416))
        cv2.putText(frame_out, "FPS: "+str(round(1./(time.time()-start), 2))+" frame: "+str(counter), (10, 340), 0, 0.4, (255, 255, 0), 1)
        cv2.putText(frame_out, "People in: "+str(len(cnt_people_in)), (10, 360), 0, 0.4, (52, 235, 240), 1)
        start = time.time()
        cv2.imshow('Yolo demo', frame_out)
        if writeVideo_flag:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): break

    cap.release()    
    if writeVideo_flag: out.release()
    cv2.destroyAllWindows()
    print("Model: "+model_name+". People in: "+str(len(cnt_people_in))+", out: "+str(len(cnt_people_out)))
