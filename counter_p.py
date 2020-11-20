# 
# based on deep sort body path tracking project
# TODO
# add intersection rule - follow to prev direction
# add rule for forbidden too fast change locations - update ids
# add tracking several videos
# check if calculated a right direction
# https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/

# from __future__ import division
import numpy as np
import cv2
import time
import os
import tensorflow as tf
import torch
import torchvision.transforms as transforms
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
from deep_sort import nn_matching
from deep_sort import preprocessing
import deep_sort.generate_detections as gdet
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
# from shapely.geometry import LineString, Point
from models.models import Darknet
from models.utils import non_max_suppression
import math

model_name = "yolov3" 
# path to input video
video_name = "39.avi"
video_path = "video/"
# border line
frame_scale = 3.84615384615 # 1600/416

border_lines = {'border1':[[int(0/frame_scale), int(400/frame_scale)], [int(1200/frame_scale), int(400/frame_scale)]]}
# save or don't the output video to the disk
writeVideo_flag = False
root_dir = os.getcwd()
# detector section
# model_def = "utils/yolov3.cfg"
model_def = "models/yolov3-tiny.cfg"

# weights_path = "Model_data/yolov3.weights"
weights_path = "models/yolov3-tiny.weights"

# wget -c https://pjreddie.com/media/files/yolov3.weights
conf_thres = 0.5
nms_thres = 0.4
img_size = 416
body_res = (256, 128)
body_min_w = 64
# end detector
# embedding pose
threshold = 0.5 # 07
nms_max_overlap = 0.9 # 1
max_cosine_distance = 0.2 # 03
# end emedding

def track_intersection_angle(xy0, xy1):
    res = []
    for key in border_lines:
        s = np.vstack([xy0, xy1, border_lines[key][0], border_lines[key][1]])        # s for stacked
        h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z != 0:                          # lines are parallel
            v0 = np.array(xy1) - np.array(xy0)
            v1 = np.array(border_lines[key][1]) - np.array(border_lines[key][0])
            angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
            if (angle > 0):
                res.append(key)
        return res

def add_intersectio_event(border_names, id):
    print(border_names, id)

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
        cv2.line(frame, (a[0],a[1]), (b[0],b[1]), (255, 255, 0), 2)
        cv2.arrowedLine(frame, z0, z1, (0, 255, 0), 2)
        cv2.putText(frame, "Out", z1, 0, 1, (0, 255, 0), 1)

def drawBorderLine(a, b):
    length = 40
    vX0 = b[0] - a[0]; vY0 = b[1] - a[1]
    mag = math.sqrt(vX0*vX0 + vY0*vY0)
    vX = vX0 / mag; vY = vY0 / mag
    temp = vX; vX = -vY; vY = temp
    z0 = (int(a[0]+vX0/2), int(a[1]+vY0/2))
    z1 = (int(a[0]+vX0/2 - vX * length), int(a[1] +vY0/2- vY * length))
    cv2.line(frame, a, b, (255, 255, 0), 2)
    cv2.arrowedLine(frame, z0, z1, (0, 255, 0), 2)
    cv2.putText(frame, "Out", z1, 0, 1, (0, 255, 0), 1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = Darknet(model_def, img_size=img_size).to(device)
    detector.load_darknet_weights(weights_path)
    tr_img = True
    nn_budget = None
    path_track = 20 # how many frames in path are saves
    cnt_people_in = {}
    #cnt_people_out = {}
    skip_counter = 10
    counter = 0
    
    # human ids model
    model_filename = 'models/mars-small128.pb'
 

    # wget https://github.com/Qidian213/deep_sort_yolov3/raw/master/model_data/mars-small128.pb
    encoder = gdet.create_box_encoder(model_filename, batch_size=64)

    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    cap = cv2.VideoCapture(video_path+video_name)
    img_size_start = (1600,1200)
    show_fh_fw = (img_size, img_size)
    max_hum_w = show_fh_fw[0]/2
    ratio_h_w = (show_fh_fw[0]/img_size, show_fh_fw[1]/img_size)
    max_hum_w = torch.tensor(max_hum_w).to(device)
    start_ratio_h_w = (img_size_start[0]/show_fh_fw[0],img_size_start[1]/(show_fh_fw[1]-104))
    #border_line = [(int(border_line[0][0]/start_ratio_h_w[0]),int(border_line[0][1]/start_ratio_h_w[1])),(int(border_line[1][0]/start_ratio_h_w[0]),int(border_line[1][1]/start_ratio_h_w[1]))]
    
    #border_line_str = LineString(border_line)
    #border_line_a = (border_line[1][0] - border_line[0][0])
    #border_line_b = (border_line[1][1] - border_line[0][1])
    out = None
    if writeVideo_flag:
        outFile = root_dir+'/video/' + model_name + '_' + video_name
        print("Save out video to file " + outFile)
        out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'XVID'), 10, show_fh_fw)
        # out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'MP4V'), 10, show_fh_fw)
    
    while True:
        r, frame = cap.read()
        if (not r):
            print("skip frame ", skip_counter)
            skip_counter -= 1
            if (skip_counter > 0): continue
            else: break
        start = time.time()
        counter += 1
        # frame = cv2.resize(frame, show_fh_fw)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.copyMakeBorder( frame, 0, 400, 0, 0, cv2.BORDER_CONSTANT)
        frame = cv2.resize(frame,(img_size, img_size))
        # input picture to Tensor
        
        # frame_cuda = torch.from_numpy(frame).float().to(device) 
        frame_cuda = transforms.ToTensor()(frame).unsqueeze(0)
        # other way
        with torch.no_grad():    
            # input_imgs = input_imgs.unsqueeze(0).permute(0, 3, 1, 2)/255.0#.permute(2, 0, 1)
            obj_detec = detector(frame_cuda)
            # print(obj_detec.shape)
            obj_detec = non_max_suppression(obj_detec, conf_thres, nms_thres)
        if not obj_detec[0] is None:
            #print("torch=",len(obj_detec[0]))
            pass
        # print(boxes)
        boxs = []
        confs = []
        for item in obj_detec:
            if item is not None:
                # print("item ", item)
                i = 0
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in item: #classes[int(cls_pred)]
                    wb = y2-y1
                    if((cls_pred == 0) and (wb < max_hum_w) and (wb > body_min_w)):
                            # boxs.append([int(y1*ratio_h_w[1]), int(x1*ratio_h_w[0]), int(y2*ratio_h_w[1]), int(x2*ratio_h_w[0])])
                            # boxs.append([int(y1), int(x1), int(y2), int(x2)])
                            boxs.append([y1, x1, y2, x2])
                            # print(conf, cls_conf)
                            # confs.append(float(conf))
                            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                            # person_photo = frame[y1:y2, x1:x2]
        # print("box=",boxs)
        # frame_tf = tf.convert_to_tensor(frame, dtype=tf.float32)
        if(len(boxs)):
            t_start2 = time.time()
            features = encoder(frame, boxs)
            start2 = t_start2-start
            # print("emd=",time.time() - t_start2)
            # emd= 0.004137516021728516
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            # detections = [Detection(bbox, conf, feature) for bbox, conf, feature in zip(boxs, confs, features)]
            start21 = time.time()-start
            # print(detections)
            boxes = np.array([d.tlwh for d in detections]) # w and h replace by x2 y2
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            # print(detections)
            tracker.predict()
            tracker.update(detections)
            for track in tracker.tracks:
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
                    lst_intrsc = track_intersection_angle(track.xy[0], x1y1)
                    if(any(lst_intrsc)):
                        #border_line
                        # print("intersection!!", track_name)
                        if(not hasattr(track, 'calculated')):
                            #if(border_line_a * (x1y1[1] - border_line[0][1]) -  border_line_b * (x1y1[0] - border_line[0][0])) > 0:
                            cnt_people_in[track.track_id] = 0
                            track.calculated = "in_" + str(len(cnt_people_in)) + "_"
                            track.color = (52, 235, 240)
                            add_intersectio_event(lst_intrsc, track.track_id)
                            #else: # 
                            #    cnt_people_out[track.track_id] = 0
                            #    track.calculated = "out_" + str(len(cnt_people_out)) + "_"
                            #    track.color = (0, 255, 0)
                            track.cross_cnt = path_track
                        clr = track.color
                    # else:
                        
                    if(hasattr(track, 'calculated')):
                        clr = track.color
                        track_name = track.calculated  + track_name
                        track.cross_cnt -= 1
                        if(track.cross_cnt < 1): track.state = 3 # delete from track list
                    track.xy = np.append(track.xy, [x1y1], axis=0)
                    track.xy = track.xy[-path_track:]
                    # cv2.arrowedLine(frame,(track.x1[0], track.y1[0]),(x1, y1),(0,255,0),4)
                    cv2.polylines(frame, [track.xy], False, clr, 3)
                else: track.xy = np.array([x1y1])
                cv2.circle(frame, x1y1, 5, clr, -1)
                cv2.rectangle(frame, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), clr, 1)
                # cv2.putText(frame, str(track.track_id),(int(bbox[1]), int(bbox[0])),0, 5e-3 * 200, (0,255,0),2)
                cv2.putText(frame, track_name, x1y1, 0, 0.4, clr, 1)
        drawBorderLines(frame)
        cv2.putText(frame, "FPS: "+str(round(1./(time.time()-start), 2))+" frame: "+str(counter), (10, 340), 0, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, "People in: "+str(len(cnt_people_in)), (10, 360), 0, 0.4, (52, 235, 240), 1)
        #cv2.putText(frame, " out: "+str(len(cnt_people_out)), (43, 376), 0, 0.4, (0, 255, 0), 1)
        # print("end frame")
        cv2.imshow("preview", frame)
        if writeVideo_flag:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): break
    cap.release()
    if writeVideo_flag: out.release()
    cv2.destroyAllWindows()
    print("Model: "+model_name+". People in: "+str(len(cnt_people_in))+", out: "+str(len(cnt_people_out)))
