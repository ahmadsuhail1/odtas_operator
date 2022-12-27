from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# imports for working with FASTAPI
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil
from pydantic import BaseModel
import copy
import requests
import urllib
from simplebgc.gimbal import Gimbal, ControlMode

# imports for loading the YOLO Model
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import uvicorn
import socketio
import base64
# from gimbal_camera import single_object_tracking
# Other Imports

from threading import Thread
import concurrent.futures
import subprocess
import time, datetime
import os
import cv2
import json

from pathlib import Path
import sys
from typing import Union, List
from collections import deque




path = Path(__file__).resolve()
path = sys.path.insert(0, '/yolov5')


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'
VIDEOS = ROOT / 'videos'
OUTPUT_VIDEOS = ROOT / 'output_vidS'


if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'pysot') not in sys.path:
    sys.path.append(str(ROOT / 'pysot'))  # add pysot ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



from components.Camera import ObjectDetection
from components.alarm import run_alarm_in_thread

# from yolov5.utils.general import (LOGGER, xyxy2xywh)
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
torch.set_num_threads(2)
# from mmtrack.apis import inference_sot, init_model
# # =============================== MAIN APP =======================

# # course origin resource sharing middleware
# # App Object
app = FastAPI()

# origins = ['https://localhost:3000']
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)


# # --------------------------- global variables -----------------------------
global camera_switch, recording, detection_switch, tracker, recorder_frame, out, alert_class, is_alarm, tracking_ids, person_count, person_count_array
global operator_obj, trackingID
global incoming_tracked_obj, MOT, SOT, end_SOT
global pitch_speed, yaw_speed, yaw_position, pitch_position, YAW_MAX_LIMIT_ANGLE, YAW_MIN_LIMIT_ANGLE, PITCH_DOWNWARD_LIMIT_ANGLE, PITCH_UPWARD_LIMIT_ANGLE
global gimbal
global socketconnection
trackingID = None
operator_obj = None
camera_switch = False
recording = False
detection_switch = False
recorder_frame = False
out = None
tracker = None
alert_class = None
is_alarm = False
tracking_ids = []
incoming_tracked_obj = []
MOT = False
SOT = False
end_SOT = False
person_count_array = []
person_count = 0

pitch_speed = 0
yaw_speed = 0
pitch_position = 0
yaw_position = 0
YAW_MAX_LIMIT_ANGLE = 0
YAW_MIN_LIMIT_ANGLE = 0
PITCH_DOWNWARD_LIMIT_ANGLE = 0
PITCH_UPWARD_LIMIT_ANGLE = 0
gimbal = None

global detection_obj, model_all_names

detection_obj = ObjectDetection("yolov5n.pt")
detection_obj.start()

sio = socketio.Client()
try: 
    sio.connect('http://4.240.59.37:3000')
    socketconnection = True
except:
    print("VM Server is not running")
    socketconnection = False
# # --------------------------- functions to run detection -----------------------------

yolo_weights=WEIGHTS / 'yolov5n.pt'
strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt'
config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml'
nr_sources = 1
half = False
device = '0' if torch.cuda.is_available() else 'cpu'
hide_labels = False
hide_class = False
hide_conf = False
conf_thres=0.45,  # confidence threshold
iou_thres=0.50,  # NMS IOU threshold
device = select_device(device)
WEIGHTS.mkdir(parents=True, exist_ok=True)
VIDEOS.mkdir(parents=True,exist_ok=True)
OUTPUT_VIDEOS.mkdir(parents=True,exist_ok=True)

#=======================================================================
# SINGLE OBJECT TRACKING MODEL
# sot_config_model = Path('mmtracking/configs/sot/mixformer/mixformer_cvt_500e_got10k.py')
# sot_checkpoint_model = Path('mmtracking/checkpoints/mixformer_cvt_500e_got10k.pth')
# config_path = Path(__file__).parent / sot_config_model
# checkpoint_path = Path(__file__).parent / sot_checkpoint_model
# sot_model = init_model(str(config_path), str(checkpoint_path))

# sot_config_model = Path('pysot/experiments/siamrpn_alex_dwxcorr/config.yaml')
# sot_snapshot_model = Path('pysot/experiments/siamrpn_alex_dwxcorr/model.pth')

sot_config_model = Path('pysot/experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml')
sot_snapshot_model = Path('pysot/experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth')
cfg.merge_from_file(sot_config_model)
cfg.CUDA = torch.cuda.is_available() and cfg.CUDA

# create sot model
sot_model = ModelBuilder()
# load model
sot_model.load_state_dict(torch.load(sot_snapshot_model, map_location=lambda storage, loc: storage.cpu()))
sot_model.eval().to(device)

sot_tracker = build_tracker(sot_model)
# ======================================================================

cfg_mot = get_config()
cfg_mot.merge_from_file(config_strongsort)

tracker = StrongSORT(
    strong_sort_weights,
    device,
    half,
    max_dist=cfg_mot.STRONGSORT.MAX_DIST,
    max_iou_distance=cfg_mot.STRONGSORT.MAX_IOU_DISTANCE,
    max_age=cfg_mot.STRONGSORT.MAX_AGE,
    n_init=cfg_mot.STRONGSORT.N_INIT,
    nn_budget=cfg_mot.STRONGSORT.NN_BUDGET,
    mc_lambda=cfg_mot.STRONGSORT.MC_LAMBDA,
    ema_alpha=cfg_mot.STRONGSORT.EMA_ALPHA,
)

tracker.model.warmup()
outputs = [None] * nr_sources
#  =====================================================
 # setting the angle rotation speed in  degree/sec
# pitch means upward/downward and yaw means left/right
# roll is not used in this project since gimbal is 2 axis
pitch_speed = 30
yaw_speed = 50

# initializing the yaw and pitch position of the gimbal
yaw_position = 0
pitch_position = 0

# setting the max/min yaw and pitch angles of the gimbal
YAW_MAX_LIMIT_ANGLE = 165
YAW_MIN_LIMIT_ANGLE = -165
PITCH_UPWARD_LIMIT_ANGLE = -10
PITCH_DOWNWARD_LIMIT_ANGLE = 160
# ============================================

# The variables in base model should be same as declared in the frontend
class Data(BaseModel):
    camera: bool
    detection: bool
    tracking: bool
    recording: bool
class TrackingID(BaseModel):
    singleID:int    
class Filter(BaseModel):
    class_filteration_list: List[int] = None

class AlarmClass(BaseModel):
    alarm_class_number : int = None
    is_alarm : bool = False
    
class TrackingPoints(BaseModel): 
    width : int = None
    height : int = None
    clickX : int = None
    clickY : int = None
    
class OperatorInfo(BaseModel):
    operator_id : str = None
    operator_drone_id : str = None

# making all the gradients false, since we are doing forward pass. This will reduce the memory consumption.
@torch.no_grad()
def generate_frames():
    
    # global variables
    global tracker, outputs, tracking_ids, alert_class,incoming_tracked_obj, MOT, SOT
    global detection_obj, detection_switch, camera_switch, tracking_switch, model_all_names, recorder_frame, is_alarm, person_count_array, person_count
    global yaw_position, pitch_position, yaw_speed, pitch_speed, YAW_MAX_LIMIT_ANGLE, YAW_MIN_LIMIT_ANGLE, PITCH_UPWARD_LIMIT_ANGLE, PITCH_DOWNWARD_LIMIT_ANGLE
    global gimbal
    
    # for frame size of the attached camera
    frame  = detection_obj.frame
    
    # for the initailized model names
    names = detection_obj.model.names
    model_all_names = list(names.keys())

    # Array of center points of tracked objects for historical trajectory
    pts = [deque (maxlen=30) for _ in range(1000)]
    

    # initializing the GIMBAL
    # gimbal = Gimbal()
    
    
   
    
    # getting the total rows and columns in frame for centering the gimbal
    rows,cols, _ = frame.shape

    # initializing the center of tracked object in the frame for SOT model
    x_medium = int(cols//2)
    y_medium = int(rows//2)

    # getting the center of the frame
    center_x = int(cols//2)
    center_y = int(rows//2)

    

    # variable for single object tracking. if the object is not tracked, then it will be set to False
    tracked_obj_exist = False
    FRAME_HEIGHT = int(detection_obj.height)
    FRAME_WIDTH = int(detection_obj.width)
    


    # set True to speed up constant image size inference
    cudnn.benchmark = True  
    
    # for sot tracking initialization
    first_frame_SOT = True
    # starting the infinite loop for generating frames
    while True:     
        
        # if camera is on then read the frame
        if camera_switch:    
            # reading frame from camera
            
            real_frame = detection_obj.read()
            
            s = "" # String to be displayed on the frame
            # if recording is on then save the frame by making a copy
            
            # =======================================

            # if detection is ON and tracking switch is OFF then detect the objects in the frame from YOLO weights
            if detection_switch and not tracking_switch:
                
                # prediction scores
                results = detection_obj.score_frame(real_frame)
                # plotting the bboxes
                real_frame = detection_obj.plot_boxes(results, real_frame)
                
            
                # # Print results
                for c in results[0].cpu().numpy().astype(int):
                    n = (results[0] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)} {os.linesep}"  # add to string
                    # s += "{0} {1}{2}, {3}".format(n, (names[int(c)]),'s' * (n > 1), "\n" )  # add to string
                real_frame = cv2.putText(real_frame,"{}".format (s), (0,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (144, 238, 144),2)
                
                

                # if alarm is on then check if the detected object is in the alert class list
                # alarm will be triggered if the detected object is in the frame
                if  is_alarm and (alert_class in results[0].cpu().numpy().astype(int)) :
                    sound_alarm()
            # =======================================
            
            
            if detection_switch and tracking_switch:
                 # processing the frames for MOT model
                # print (SOT,MOT, "SOT,MOT")
                annotator = Annotator(real_frame, line_width=2, pil=not ascii, example = str(names))

                if MOT and not SOT:
                    # print("In MOT")
                    results = detection_obj.model(real_frame)

                    # extracting the predictions from YOLO model for STRONGSORT MOT model
                    det = results.pred[0]
                    
                    

                # if detection is available then process the frame for MOT model
                    if det is not None and len(det):
                        
                        # converting the x,y,w,h to x1,y1,x2,y2 format
                        # xywhs = xyxy2xywh(det[:, 0:4])
                        
                        # standard x,w,w,h format
                        xywhs = det[:, 0:4]
                        confs = det[:, 4]
                        clss = det[:, 5]
                        
                        
                        outputs = tracker.update(det, real_frame)
                        
                        # draw boxes for visualization
                        if len(outputs) > 0:
                            for c in clss.unique():
                                n = (clss == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)} \n"  # add to string
                                
                            for _, (output, conf) in enumerate(zip(outputs, confs)):
                                
                                bboxes = output[0:4]
                                id = output[4]
                                cls = output[5]
                                c = int(cls)  # integer class
                                id = int(id)  # integer id
                                
                                
                                if (id not in person_count_array) and c==0:
                                    person_count_array.append(id)
                                    person_count += 1
                                
                                if id not in tracking_ids:
                                    tracking_ids.append(id)          
                                    
                                print(trackingID, "trackingID inside loop")
                                if not trackingID:
                                    # print('abrcadabra')
                                    pass
                                else:
                                    if trackingID.singleID == id:
                                        print('inside tracking id')
                                        x_medium1, y_medium2 = int((bboxes[0] + bboxes[2])/2),int((bboxes[1] + bboxes[3])/2)
                                        cv2.line(real_frame,(x_medium1,0), (x_medium1, FRAME_HEIGHT), (255,0,0),2)
                                        cv2.line(real_frame,(0,y_medium2), (FRAME_WIDTH,y_medium2), (255,0,0),2)   
                                        if (yaw_position >= YAW_MIN_LIMIT_ANGLE and yaw_position <= YAW_MAX_LIMIT_ANGLE):
                                            # and (pitch_position >= PITCH_UPWARD_LIMIT_ANGLE and (pitch_position <= PITCH_DOWNWARD_LIMIT_ANGLE)):
                                                if x_medium1 < center_x - 70:
                                                    if yaw_position != YAW_MIN_LIMIT_ANGLE:
                                                        yaw_position -= 1
                                                        # gimbal.control(
                                                        #     pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=pitch_position,
                                                        #     yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)

                                                elif x_medium1 > center_x + 70:
                                                    if yaw_position != YAW_MAX_LIMIT_ANGLE:
                                                        yaw_position += 1
                                                        # gimbal.control(
                                                        #     pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=pitch_position,
                                                        #     yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)

                                                else:
                                                    pass
                                            
                                        if pitch_position >= PITCH_UPWARD_LIMIT_ANGLE and pitch_position <= PITCH_DOWNWARD_LIMIT_ANGLE:
                                                if y_medium2 < center_y - 70:
                                                    if pitch_position != PITCH_UPWARD_LIMIT_ANGLE:
                                                        pitch_position -=1
                                                        # gimbal.control(
                                                        #     pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=pitch_position,
                                                        #     yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)

                                                elif y_medium2 > center_y + 70:
                                                     if pitch_position != PITCH_DOWNWARD_LIMIT_ANGLE:
                                                        pitch_position +=1
                                                        # gimbal.control(
                                                        #     pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=pitch_position,
                                                        #     yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)

                                                else: 
                                                    pass     
                                
                                
                                center = (int((bboxes[0]+bboxes[2])/2), int((bboxes[1]+bboxes[3])/2))
                                pts[id].append(center)
                                
                                for j in range(1,len(pts[id])):
                                    if pts[id][j-1] is None or pts[id][j] is None:
                                        continue
                                    thickness = int(np.sqrt(64/float(j+1))*2)
                                    cv2.line(real_frame, (pts[id][j-1]), (pts[id][j]), colors(c, True), thickness)
                                
                                label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                    (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                annotator.box_label(bboxes, label, color=colors(c, True))
                                real_frame = annotator.result()
                                
                                
                                real_frame = cv2.putText(real_frame,"Total Person Count: {}".format(person_count), (0,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
                                
                                real_frame =  cv2.putText(real_frame,"{}".format (s), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
            
                    
                    else:
                        tracker.increment_ages()
                        # if the tracking object exists or if incoming_tracked_obj contains point x,y 
                        # And run only if Single Object Tracking is True
                # print(incoming_tracked_obj, "incoming_tracked_obj")
                # print(tracked_obj_exist, "tracked_obj_exist")
                # print(SOT, "SOT")
                if (incoming_tracked_obj or tracked_obj_exist) and SOT:
                    print("IN SOTTTTTTTTTT ")
                    # print(first_frame_SOT, "first_frame_SOT")
                    # print(xywhs, "xywhs")

                    if first_frame_SOT:
                        # check if the clicked point is in any bounding box, if true get the coordinates of the bbox
                        bbox = is_point_in_bbox(xywhs)
                        if bbox is not None:
                            sot_tracker.init(real_frame, bbox)
                            first_frame_SOT = False
                            MOT = False
                        if bbox is None:
                            # if the point is not in any bbox then set the SOT to False and
                            # first_frame_SOT to True so that the next time the point is clicked, it will be checked again
                            SOT = False
                            first_frame_SOT = True
                            MOT = True
                    else:
                        # bbox = is_point_in_bbox(xywhs)
                        # if bbox is None:
                        
                        sot_outputs = sot_tracker.track(real_frame)
                    
                        # setting Multi Object Tracking to False
                        # MOT = False
                        
                        # extracted bbox coordinates are in the form of x1,x2,y1,y2
                        # for initializing SOT model tracking, we need to get the center of the bbox
                        
                        # sot model is the name of the model we are using for single tracking, real_frame is the actual frame for we have provided, from where the tracking will start, bbox is the bbox coordinates of the object we want to track, tracker_initialize_id is the frame id
                        if len(sot_outputs) > 0:
                            if not end_SOT:
                                if 'polygon' in sot_outputs:
                                    polygon = np.array(sot_outputs['polygon']).astype(np.int32)
                                    cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                                True, (0, 255, 0), 3)
                                    mask = ((sot_outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                                    mask = mask.astype(np.uint8)
                                    mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                                    frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                                else:
                                    sot_bbox = list(map(int, sot_outputs['bbox']))
                                    cv2.rectangle(real_frame, (sot_bbox[0], sot_bbox[1]),
                                                (sot_bbox[0]+sot_bbox[2], sot_bbox[1]+sot_bbox[3]),
                                                (0, 255, 0), 3)
                                    x1,y1,x2,y2 = int(sot_bbox[0]),int(sot_bbox[1]),int(sot_bbox[0]+sot_bbox[2]),int(sot_bbox[1]+sot_bbox[3])
                                    x_medium, y_medium = int((x1 + x2)/2),int((y1 + y2)/2)
                                    tracked_obj_exist = True
                            else:
                                SOT = False
                                MOT = True
                                tracked_obj_exist = False
                                first_frame_SOT = True

                        else:
                            
                        # if the result is empty then set the tracked_obj_exist to False
                            # x_medium = int(cols//2)
                            # y_medium = int(rows//2)
                            SOT = False
                            MOT = True
                            tracked_obj_exist = False
                            first_frame_SOT = True


                        # frame = cv2.resize(frame, (480, 640), interpolation = cv2.INTER_LINEAR)
                        
                        # ==============================================
                        
                        # if (yaw_position >= YAW_MIN_LIMIT_ANGLE and yaw_position <= YAW_MAX_LIMIT_ANGLE):
                        # # and (pitch_position >= PITCH_UPWARD_LIMIT_ANGLE and (pitch_position <= PITCH_DOWNWARD_LIMIT_ANGLE)):
                        #     if x_medium < center_x - 70:
                        #         if yaw_position != YAW_MIN_LIMIT_ANGLE:
                        #             yaw_position -= 1
                        #             gimbal.control(
                        #                 pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=pitch_position,
                        #                 yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)

                        #     elif x_medium > center_x + 70:
                        #         if yaw_position != YAW_MAX_LIMIT_ANGLE:
                        #             yaw_position += 1
                        #             # gimbal.control(
                        #             #     pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=pitch_position,
                        #             #     yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)

                        #     else:
                        #         pass
                        
                        # if pitch_position >= PITCH_UPWARD_LIMIT_ANGLE and pitch_position <= PITCH_DOWNWARD_LIMIT_ANGLE:
                            # if y_medium < center_y - 70:
                                
                            #     if pitch_position != PITCH_UPWARD_LIMIT_ANGLE:
                            #         pitch_position -=1
                            #         # gimbal.control(
                            #         #     pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=pitch_position,
                            #         #     yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)

                            # elif y_medium > center_y + 70:
                            #     if pitch_position != PITCH_DOWNWARD_LIMIT_ANGLE:
                            #         pitch_position +=1
                            #         # gimbal.control(
                            #         #     pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=pitch_position,
                                    #     yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)

                            # else: 
                            #     pass

                        incoming_tracked_obj = []
                        # else:
                        #     MOT = True
                        #     SOT = False
                        #     tracked_obj_exist = False
                        #     first_frame_SOT = True
                            
                        
                #  ========================================
                # 
                # ===========================
            
            if tracking_switch is False:
                SOT = False
                # trackingID = None
                first_frame_SOT = True
                
                MOT = True
                
            #     person_count = 0
                        
            if recording:
                real_frame = cv2.putText(real_frame,"Recording...", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (144, 238, 144),2)
                recorder_frame = real_frame.copy()
                

            # real_frame = cv2.resize(real_frame, (480, 640))
            try:    
                _, buffer = cv2.imencode('.jpg', real_frame)
                if socketconnection:
                    data = base64.b64encode(buffer)
                    sio.emit('data', data)
                    
                frame = buffer.tobytes()
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            except Exception as e:
                print(e)
                detection_obj.stop()   
                pass
        
        else:
            pass

# ================================================================================ 

def record(out):
    global recorder_frame, recording
    while(recording):
        time.sleep(0.03)
        out.write(recorder_frame)
        

# =================================================================================

# ALARM

def sound_alarm():
    global is_alarm
    run_alarm_in_thread()
    is_alarm = False
    


def send_trackingids():
    global tracking_ids, tracking_ids_sent
    tracking_ids_sent = copy.deepcopy(tracking_ids)
    tracking_ids = []
    return tracking_ids_sent



# def clip_boxes(boxes, shape):
#     # Clip boxes (xyxy) to image shape (height, width)
#     if isinstance(boxes, torch.Tensor):  # faster individually
#         boxes[:, 0].clamp_(0, shape[1])  # x1
#         boxes[:, 1].clamp_(0, shape[0])  # y1
#         boxes[:, 2].clamp_(0, shape[1])  # x2
#         boxes[:, 3].clamp_(0, shape[0])  # y2
#     else:  # np.array (faster grouped)
#         boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
#         boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # boxes[:, [0, 2]] -= pad[0]  # x padding
    # boxes[:, [1, 3]] -= pad[1]  # y padding
    # boxes[:, :4] /= gain
    print(gain, "gain value")
    print(pad, "pad value")
    if gain != 0 and pad != 0:
        boxes[0] -= pad[0]
        boxes[1] -= pad[1]
        boxes[0] /= gain
        boxes[1] /= gain
        # clip_boxes(boxes, img0_shape)
        return boxes
    else:
        return boxes

def is_point_in_bbox(detected2DArray):
    global incoming_tracked_obj, detection_obj
    # print(incoming_tracked_obj)

    im_width = detection_obj.width
    im_height = detection_obj.height
    print(incoming_tracked_obj)
    # xywhs
    pointX = incoming_tracked_obj[2][1]
    pointY = incoming_tracked_obj[3][1]

    div_width = incoming_tracked_obj[0][1]
    div_height = incoming_tracked_obj[1][1]

    new_points = scale_boxes(list((div_width,div_height)), list((pointX,pointY)), list((im_width,im_height)))
    new_points_list =[]
    [new_points_list.append(y) for y in new_points]
    bboxes = detected2DArray[0].cpu().numpy().astype(int)
    
    # for i, bbox in enumerate(bboxes):
    #     print(type(bbox))
    #     print(bbox)
    #     bbox = bbox.tolist()
    if (new_points_list[0] >= bboxes[0] and new_points_list[0] <= bboxes[2]) and (new_points_list[1] >= bboxes[1] and new_points_list[1] <= bboxes[3]):
        print("Clicked in BBOX")

        return bboxes 
    else:
        print("Not Clicked in BBOX")
        return None
# ===============================================================================
def control_camera(movement: str):
    global gimbal, yaw_position, pitch_position, yaw_speed, pitch_speed, SOT
    if not SOT:
        if movement == "up":
            if pitch_position != PITCH_UPWARD_LIMIT_ANGLE:
                pitch_position -= 2
                # gimbal.control(pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle= pitch_position, yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)    
                print("pitch_position: ",pitch_position, " yaw_position: "  ,yaw_position)
        elif movement == "down":
            if pitch_position != PITCH_DOWNWARD_LIMIT_ANGLE:
                pitch_position += 2
                # gimbal.control(pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle= pitch_position, yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)
                print("pitch_position: ",pitch_position, " yaw_position: "  ,yaw_position)
        elif movement == "left":
            if yaw_position != YAW_MIN_LIMIT_ANGLE:
                yaw_position -= 2
                # gimbal.control(pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle= pitch_position, yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)
                print("pitch_position: ",pitch_position, " yaw_position: "  ,yaw_position)
        elif movement == "right":
            if yaw_position != YAW_MAX_LIMIT_ANGLE:
                yaw_position += 2
                # gimbal.control(pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle= pitch_position, yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)
                print("pitch_position: ",pitch_position, " yaw_position: "  ,yaw_position)
        elif movement == "reset":
            yaw_position = 0
            pitch_position = 0
            # gimbal.control(pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle= pitch_position, yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)
            print("pitch_position: ",pitch_position, " yaw_position: "  ,yaw_position)
            
        elif movement == "lock":
            # gimbal.control(roll_mode=ControlMode.no_control,pitch_mode=ControlMode.no_control,yaw_mode=ControlMode.no_control)
            print("Gimbal Locked")
        else:
            print("Invalid Movement")
    else:
        print("SOT is ON, Gimbal ")
# ================================================================================ 


def make_vidfile(file: File):
    with open(f"videos/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


def detect_vidfile(file: File):
    filepath : Union[str,Path] = Path(file)
    file_extension = Path(filepath).suffix
    if file_extension == '.mp4' or '.avi':
        source_str = 'videos/'+filepath.name
        source_path = Path(source_str)
        
        subprocess.run(["python", "yolov5/detect.py", "--weights", "weights/last_htv_23.pt",
                    "--source", source_path, "--project", "output_vids" ], shell=True)
        # subprocess.run(["python", "yolov5/detect.py", "--weights", "weights/yolov5s.pt",
        #             "--source", source_path, "--project", "output_vids" ], shell=True)
def process_VidFile(file: File):
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     executor.submit(make_vidfile, file)
    make_vidfile(file)

    file2 = os.path.join(os.getcwd(), "/videos", file.filename)
    detect_vidfile(file2)


def upload_vidfile_to_cloud(file: File):
    global operator_obj
    try:
        # send file to given url with post request and file as a parameter
        
        response = requests.post(f"http://4.240.57.37:3000/videos/postVideo/{operator_obj.operator_id}", data= {"operator": operator_obj.operator_name}, files={"file": open(file, "rb")})
        print(response)
        # , files={"file": open(file, "rb")}
        print("VM is up")
        # pass
    except:
        print("VM is down")
# =================================================================


@app.get("/")
async def read_root():

    return {"Hi": "World"}


@app.get("/video")
async def livestream():
    return StreamingResponse(generate_frames(),  media_type="multipart/x-mixed-replace;boundary=frame") 

# send json data stream to the client
@app.get("/video/trackingids")
async def trackingID_request():
    return send_trackingids()
    

 
@app.post("/video/requests")
async def handle_form(data: Data):
    global camera_switch, detection_obj, detection_switch, tracking_switch, recording, recorder_frame, out, MOT
    camera_switch = data.camera
    detection_switch = data.detection
    tracking_switch = data.tracking
    recording = data.recording
    MOT = tracking_switch
    SOT if not(tracking_switch) else MOT
    print("camera_switch: ", camera_switch, " detection_switch: ", detection_switch, " tracking_switch: ", tracking_switch, " recording: ", recording)
    if recording and out is None:
        now=datetime.datetime.now() 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        source_str = 'output_vids/'+'vid_{}.mp4'.format(str(now).replace(":",''))
        # recorded_video_path = Path(source_str)
        out = cv2.VideoWriter(source_str, fourcc, 20.0, (640, 480))
        #Start new thread for recording the video
        thread = Thread(target = record, args=[out,])
        thread.start()
    elif (not recording) and out is not None:
        out.release()
        out = None
    

@app.post("/send_operator_info")
async def recieved_operator_info(operator_info: OperatorInfo):
    global operator_obj
    operator_obj = operator_info
    
    print(operator_obj.operator_id)


@app.post("/uploadvideo")
async def upload_video(file: UploadFile = File(...)):
    os.makedirs(os.path.join("videos"), exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit(process_VidFile(file))
        # executor.submit(upload_vidfile_to_cloud(file))
    return {"filename ": file.filename}


@app.post("/video/filterdetection")
async def handle_detection_types(filteredArray: Filter):
    global detection_obj, model_all_names
    detection_obj.model.classes = filteredArray.class_filteration_list if filteredArray.class_filteration_list else model_all_names
    
@app.post("/video/setalarm")
async def handle_detection_alarm(alarm: AlarmClass):
    global alert_class, is_alarm
    alert_class = alarm.alarm_class_number
    is_alarm = alarm.is_alarm
    
@app.post("/video/trackingpoints")
async def handle_tracking_points(trackingPoints: TrackingPoints):
    global incoming_tracked_obj, SOT
    print(trackingPoints, "trackingPoints")
    incoming_tracked_obj = copy.deepcopy(list(trackingPoints))
    SOT =  True
    
    # print(SOT, "SOT when called on route")
@app.post("/video/end_sot")   
async  def end_sot():
    global end_SOT
    end_SOT = True 

@app.post("/tracking/id")   
async  def recieve_tracking_id(singleID: TrackingID):
    global trackingID
    trackingID = singleID
    print(trackingID, "trackingID when api hit")
    
    
@app.get("/cameramovement/left")
async def handle_left_movement():
    movement = "left"
    control_camera(movement)
    
@app.get("/cameramovement/right")
async def handle_right_movement():
    movement = "right"
    control_camera(movement)
    
@app.get("/cameramovement/up") 
async def handle_up_movement():
    movement = "up"
    control_camera(movement)

@app.get("/cameramovement/down")
async def handle_down_movement():
    movement = "down"
    control_camera(movement) 
    
@app.get("/cameramovement/reset")
async def handle_down_movement():
    movement = "reset"
    control_camera(movement)   

@app.get("/cameramovement/lock")
async def handle_down_movement():
    movement = "lock"
    control_camera(movement)    
    

# ========================================================================
# Server Runner

if __name__ == "__main__":
    uvicorn.run(app, host="localhost:8000", port=8000, reload=True)