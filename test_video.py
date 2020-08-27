import os, yaml
import time
import numpy as np
import torch
from tool.utils import post_processing, nms_cpu, plot_boxes_cv2, load_class_names, drawBorderLine
from tool.darknet2pytorch import Darknet
import cv2
from deep_sort import build_tracker
from easydict import EasyDict as edict
import torchvision.transforms as transforms
from shapely.geometry import LineString, Point

writeVideo_flag = False

imgfile = 'data/dog.jpg'
cfgfile = 'cfg/yolov4.cfg'
weightfile = 'Model_data/yolov4_dark.weights'
conf_thresh = 0.4
nms_thresh = 0.6
video_path = 'video/39.avi'
cfg_track_path = 'cfg/deep_sort.yaml'
class_names = load_class_names('data/coco.names')
# output_video_size = 416
border_line = [(0, 400), (1200, 400)]
path_track = 20 # how many frames in path are saves

def get_region_boxes(boxes_and_confs):
    # print('Getting boxes from boxes and confs ...')
    boxes_list = []
    confs_list = []
    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])
        # print("item[0]", item[3])
        # print("item[1]", item[1])
    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)
    return [boxes, confs]

if __name__ == "__main__":    
    use_cuda = torch.cuda.is_available()
    print("is cuda available", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
# detector
    detector = Darknet(cfgfile).to(device)
    detector.load_weights(weightfile)
    detector.eval()

#  tracker
    cfg_track = edict({})
    if cfg_track_path is not None:
        assert(os.path.isfile(cfg_track_path))
    with open(cfg_track_path, 'r') as fo:
        cfg_track.update(yaml.load(fo.read()))
    # print("cfg_track", cfg_track.DEEPSORT)

    tracker = build_tracker(cfg_track, use_cuda=use_cuda)

 # video parameters
    cap = cv2.VideoCapture(video_path)
    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("frame img size" ,im_width, im_height)

    if writeVideo_flag:
        outFile = root_dir+'/video/' + model_name + '_' + video_name
        print("Save out video to file " + outFile)
        out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'XVID'), 10, show_fh_fw)
        # out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'MP4V'), 10, show_fh_fw)

#   prepeare borders
    img_size_start = (1600,1200)
    show_fh_fw = (detector.width, detector.height)
    max_hum_w = show_fh_fw[0]/2
    ratio_h_w = (show_fh_fw[0]/detector.width, show_fh_fw[1]/detector.height)
    max_hum_w = torch.tensor(max_hum_w).to(device)
    start_ratio_h_w = (img_size_start[0]/show_fh_fw[0],img_size_start[1]/(show_fh_fw[1]-104))

    border_line = [(int(border_line[0][0]/start_ratio_h_w[0]),int(border_line[0][1]/start_ratio_h_w[1])),(int(border_line[1][0]/start_ratio_h_w[0]),int(border_line[1][1]/start_ratio_h_w[1]))]    
    border_line_str = LineString(border_line)
    border_line_a = (border_line[1][0] - border_line[0][0])
    border_line_b = (border_line[1][1] - border_line[0][1])


# variables
    counter = 0
    skip_counter = 4 
    cnt_people_in = {}
    cnt_people_out = {}

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
        # Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        # frame = cv2.imread(imgfile)
        frame_sm = cv2.resize(frame, (detector.width, detector.height))
        # frame_cuda = transforms.ToTensor()(frame_sm).to(device).unsqueeze(0)
        frame_cuda = torch.from_numpy(frame_sm.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        # for batches
        # frame_sm = torch.from_numpy(frame_sm.transpose(0, 3, 1, 2)).float().div(255.0)
        with torch.no_grad():
            detections = get_region_boxes(detector(frame_cuda))
            detections_suppression = post_processing(frame_sm, conf_thresh, nms_thresh, detections, True)

            tracks = tracker.update(detections_suppression[0], frame_sm)
            # print(outputs)
        frame_out = frame_sm
        # for box in tracks:
            # frame_out = cv2.rectangle(frame_out, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
            # frame_out = cv2.putText(frame_out,str(box[4]),(box[0],box[1]+10), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,0], 2)

        for track in tracks:
            # print(track.mean[:4])
            if(not track.is_confirmed() or track.time_since_update > 1):
                # if(track.time_since_update > life_frame_limit): track.state = 3 # if missed to long than delete id
                continue 
            x1y1 = (int(track.mean[0]), int(track.mean[1]))
            clr = (255, 255, 0) # default color
            track_name = str(track.track_id) # default name
            if(hasattr(track, 'xy')):
                # detect direction
                track_line = LineString([track.xy[0], x1y1])
                if(track_line.intersection(border_line_str)):
                    # print("intersection!!", track_name)
                    if(not hasattr(track, 'calculated')):
                        if(border_line_a * (x1y1[1] - border_line[0][1]) -  border_line_b * (x1y1[0] - border_line[0][0])) > 0:
                            cnt_people_in[track.track_id] = 0
                            track.calculated = "in_" + str(len(cnt_people_in)) + "_"
                            track.color = (52, 235, 240)
                        else: # 
                            cnt_people_out[track.track_id] = 0
                            track.calculated = "out_" + str(len(cnt_people_out)) + "_"
                            track.color = (0, 255, 0)
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
                cv2.polylines(frame_out, [track.xy], False, clr, 3)
            else: track.xy = np.array([x1y1])
            cv2.circle(frame_out, x1y1, 5, clr, -1)
            box = track.to_tlbr()
            cv2.rectangle(frame_out, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
            # cv2.rectangle(frame_out, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), clr, 1)
            # cv2.putText(frame, str(track.track_id),(int(bbox[1]), int(bbox[0])),0, 5e-3 * 200, (0,255,0),2)
            cv2.putText(frame_out, track_name, x1y1, 0, 0.4, clr, 1)
        cv2.line(frame_out, border_line[0], border_line[1], (255, 255, 0), 2)
        z0, z1 = drawBorderLine(border_line[0], border_line[1])
        cv2.arrowedLine(frame_out, z0, z1, (0, 255, 0), 2)
        cv2.putText(frame_out, "Out", z1, 0, 1, (0, 255, 0), 1)
        cv2.putText(frame_out, "FPS: "+str(round(1./(time.time()-start), 2))+" frame: "+str(counter), (10, 340), 0, 0.4, (255, 255, 0), 1)
        cv2.putText(frame_out, "People in: "+str(len(cnt_people_in)), (10, 360), 0, 0.4, (52, 235, 240), 1)
        cv2.putText(frame_out, " out: "+str(len(cnt_people_out)), (43, 376), 0, 0.4, (0, 255, 0), 1)
        # print(counter, "FPS:", round(1./(time.time()-start), 2), len(tracks))

        # print("--", counter. fps)
        frame_out = cv2.resize(frame_out, (512, 416))
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
