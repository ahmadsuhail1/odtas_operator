# imports for working with FASTAPI
from fastapi import FastAPI, UploadFile, File, Request, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil
from pydantic import BaseModel
import copy

from simplebgc.gimbal import Gimbal, ControlMode

# imports for loading the YOLO Model
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import uvicorn

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
from typing import Union, List, Optional

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

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from components.Camera import ObjectDetection
from components.alarm import run_alarm_in_thread

from yolov5.utils.general import (LOGGER, xyxy2xywh)
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

from mmtrack.apis import inference_sot, init_model
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

global camera_switch, recording, detection_switch, tracker, recorder_frame, out, alert_class, is_alarm, tracking_ids
global incoming_tracked_obj, MOT, SOT
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
global detection_obj, model_all_names

detection_obj = ObjectDetection("http://192.168.137.111:4747/video","yolov5n.pt")


detection_obj.start()
# # --------------------------- functions to run detection -----------------------------

yolo_weights=WEIGHTS / 'yolov5n.pt'
strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt'
config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml'
nr_sources = 1
half = False
device = '0' if torch.cuda.is_available() else ''
hide_labels = False
hide_class = False
hide_conf = False
conf_thres=0.45,  # confidence threshold
iou_thres=0.50,  # NMS IOU threshold
device = select_device(device)
WEIGHTS.mkdir(parents=True, exist_ok=True)
VIDEOS.mkdir(parents=True,exist_ok=True)

# SINGLE OBJECT TRACKING MODEL
sot_config_model = Path('mmtracking/configs/sot/mixformer/mixformer_cvt_500e_got10k.py')
sot_checkpoint_model = Path('mmtracking/checkpoints/mixformer_cvt_500e_got10k.pth')
config_path = Path(__file__).parent / sot_config_model
checkpoint_path = Path(__file__).parent / sot_checkpoint_model
sot_model = init_model(str(config_path), str(checkpoint_path))
# ======================================================================

cfg = get_config()
cfg.merge_from_file(config_strongsort)

tracker = StrongSORT(
    strong_sort_weights,
    device,
    half,
    max_dist=cfg.STRONGSORT.MAX_DIST,
    max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
    max_age=cfg.STRONGSORT.MAX_AGE,
    n_init=cfg.STRONGSORT.N_INIT,
    nn_budget=cfg.STRONGSORT.NN_BUDGET,
    mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
    ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
)

tracker.model.warmup()
outputs = [None] * nr_sources

# The variables in base model should be same as declared in the frontend
class Data(BaseModel):
    camera: bool
    detection: bool
    tracking: bool
    recording: bool
    
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
    


# making all the gradients false, since we are doing forward pass. This will reduce the memory consumption.
@torch.no_grad()
def generate_frames():
    
    # global variables
    global tracker, outputs, tracking_ids, alert_class,incoming_tracked_obj, MOT, SOT
    global detection_obj, detection_switch, camera_switch, tracking_switch, model_all_names, recorder_frame, is_alarm
    
    # for frame size of the attached camera
    frame  = detection_obj.frame
    
    # for the initailized model names
    names = detection_obj.model.names
    model_all_names = list(names.keys())


    # for frame id in SOT model
    tracker_initialize_id = 0

    # initializing the GIMBAL
    # gimbal = Gimbal()
    
    
    # setting the angle rotation speed in  degree/sec
    # pitch means upward/downward and yaw means left/right
    # roll is not used in this project since gimbal is 2 axis
    pitch_speed = 30
    yaw_speed = 50
    
    
    # getting the total rows and columns in frame for centering the gimbal
    rows,cols, _ = frame.shape

    # initializing the center of tracked object in the frame for SOT model
    x_medium = int(cols//2)
    y_medium = int(rows//2)

    # getting the center of the frame
    center_x = int(cols//2)
    center_y = int(rows//2)

    # initializing the yaw and pitch position of the gimbal
    yaw_position = 0
    pitch_position = 0

    # setting the max/min yaw and pitch angles of the gimbal
    YAW_MAX_LIMIT_ANGLE = 165
    YAW_MIN_LIMIT_ANGLE = -165
    PITCH_UPWARD_LIMIT_ANGLE = -10
    PITCH_DOWNWARD_LIMIT_ANGLE = 160

    # variable for single object tracking. if the object is not tracked, then it will be set to False
    tracked_obj_exist = False
    FRAME_HEIGHT = int(detection_obj.height)
    FRAME_WIDTH = int(detection_obj.width)
    
    
    IN_VIDEO = True
    OUT_VIDEO = False
    
    # setting the fps for SOT
    fps = 30

    # initializing the variable for extracing the frame returned from SOT model
    tracked_frame = None


    # starting the infinite loop for generating frames
    while True:     
        
        # if camera is on then read the frame
        if camera_switch:
            
            # set True to speed up constant image size inference
            cudnn.benchmark = True  
            
            # reading frame from camera
            real_frame = detection_obj.read()
            
            # if recording is on then save the frame by making a copy
            if recording:
                real_frame = cv2.putText(real_frame,"Recording...", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                recorder_frame = real_frame.copy()
                
            # =======================================

            # if detection is ON and tracking switch is OFF then detect the objects in the frame from YOLO weights
            if detection_switch and not tracking_switch:
                
                # prediction scores
                results = detection_obj.score_frame(real_frame)
                # plotting the bboxes
                real_frame = detection_obj.plot_boxes(results, real_frame)

                # if alarm is on then check if the detected object is in the alert class list
                # alarm will be triggered if the detected object is in the frame
                if  is_alarm and (alert_class in results[0].numpy().astype(int)) :
                    sound_alarm()
            # =======================================
            
            
            if detection_switch and tracking_switch:
                 # processing the frames for MOT model
                annotator = Annotator(real_frame, line_width=2, pil=not ascii, example = str(names))

                
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
                    
                    # if the tracking object exists or if incoming_tracked_obj contains point x,y 
                    # And run only if Single Object Tracking is True
                    if (incoming_tracked_obj or tracked_obj_exist) and SOT:
                        
                        # setting Multi Object Tracking to False
                        print(SOT)
                        MOT = False
                        
                        # check if the clicked point is in any bounding box, if true get the coordinates of the bbox
                        bbox = is_point_in_bounding_box(xywhs)

                        # extracted bbox coordinates are in the form of x1,x2,y1,y2
                        # for initializing SOT model tracking, we need to get the center of the bbox
                        
                        # sot model is the name of the model we are using for single tracking, real_frame is the actual frame for we have provided, from where the tracking will start, bbox is the bbox coordinates of the object we want to track, tracker_initialize_id is the frame id
                        result = inference_sot(sot_model, real_frame, bbox ,tracker_initialize_id)
                        tracker_initialize_id += 1


                        # if the result is empty then set the tracked_obj_exist to False
                        if len(result) == 0:
                            x_medium = int(cols//2)
                            y_medium = int(rows//2)
                            tracked_obj_exist = False
                        
                        # if the result is not empty then set the tracked_obj_exist to True
                        if len(result) > 0:
                            result_key = result.get('track_bboxes') 
                            x1,y1,x2,y2 = int(result_key[0]),int(result_key[1]),int(result_key[2]),int(result_key[3])
                            x_medium, y_medium = int((x1 + x2)/2),int((y1 + y2)/2)
                            tracked_obj_exist = True


                        # frame = cv2.resize(frame, (480, 640), interpolation = cv2.INTER_LINEAR)
                        out_file = None
                        
                        # ==============================================
                        print(detection_obj.frame)
                        tracked_frame = sot_model.show_result(
                            detection_obj.frame,
                            result,
                            show=False,
                            wait_time=int(1000. / fps) if fps else 0,
                            out_file=out_file,
                            thickness=2)

                        print(tracked_frame)

                        if (yaw_position >= YAW_MIN_LIMIT_ANGLE and yaw_position <= YAW_MAX_LIMIT_ANGLE):
                        # and (pitch_position >= PITCH_UPWARD_LIMIT_ANGLE and (pitch_position <= PITCH_DOWNWARD_LIMIT_ANGLE)):
                            if x_medium < center_x - 70:
                                if yaw_position != YAW_MIN_LIMIT_ANGLE:
                                    yaw_position -= 1
                                    # gimbal.control(
                                    #     pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=0,
                                    #     yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)

                            elif x_medium > center_x + 70:
                                if yaw_position != YAW_MAX_LIMIT_ANGLE:
                                    yaw_position += 1
                                    # gimbal.control(
                                    #     pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=0,
                                    #     yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)

                            else:
                                pass
                        
                        if pitch_position >= PITCH_UPWARD_LIMIT_ANGLE and pitch_position <= PITCH_DOWNWARD_LIMIT_ANGLE:
                            if y_medium < center_y - 70:
                                
                                if pitch_position != PITCH_UPWARD_LIMIT_ANGLE:
                                    pitch_position -=1
                                    # gimbal.control(
                                    #     pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=pitch_position,
                                    #     yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=0)

                            elif y_medium > center_y + 70:
                                if pitch_position != PITCH_DOWNWARD_LIMIT_ANGLE:
                                    pitch_position +=1
                                    # gimbal.control(
                                    #     pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=pitch_position,
                                    #     yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=0)

                            else: 
                                pass
                        
                        # ======================================================================================
                        if not tracked_obj_exist:
                            tracker_initialize_id = 0
                            SOT = False
                            MOT = True
                            print(SOT, MOT)

                        incoming_tracked_obj = []
                    
                    if tracking_switch and MOT:
                        outputs = tracker.update(det, real_frame)
                        
                        # draw boxes for visualization
                        if len(outputs) > 0:
                            for _, (output, conf) in enumerate(zip(outputs, confs)):
            
                                bboxes = output[0:4]
                                id = output[4]
                                cls = output[5]
                                c = int(cls)  # integer class
                                id = int(id)  # integer id
                                if id not in tracking_ids:
                                    tracking_ids.append(id)
                                label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                    (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                annotator.box_label(bboxes, label, color=colors(c, True))

                else:
                    if tracking_switch:
                        tracker.increment_ages()
                
                #  ========================================
                # 
                # ===========================
                if tracking_switch and MOT:
                    real_frame = annotator.result()
                if tracking_switch and SOT:
                    real_frame = tracked_frame


            # real_frame = cv2.resize(real_frame, (480, 640))
            try:    
                _, buffer = cv2.imencode('.jpg', real_frame)
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
        time.sleep(0.05)
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
    boxes[0] -= pad[0]
    boxes[1] -= pad[1]
    boxes[0] /= gain
    boxes[1] /= gain
    # clip_boxes(boxes, img0_shape)
    return boxes

def is_point_in_bounding_box(detected2DArray):
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
    print(bboxes)
    print(type(new_points_list))
    # for i, bbox in enumerate(bboxes):
    #     print(type(bbox))
    #     print(bbox)
    #     bbox = bbox.tolist()
    if (new_points_list[0] >= bboxes[0] and new_points_list[0] <= bboxes[2]) and (new_points_list[1] >= bboxes[1] and new_points_list[1] <= bboxes[3]):
        print("Clicked in BBOX")

        return bboxes 
    
    


def single_object_tracking():
    
    pass
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
        
        subprocess.run(["python", "yolov5/detect.py", "--weights", "last_htv_23.pt",
                    "--source", source_path, "--project", "output_vids" ], shell=True)
    
def process_VidFile(file: File):
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     executor.submit(make_vidfile, file)
    make_vidfile(file)

    file2 = os.path.join(os.getcwd(), "/videos", file.filename)
    detect_vidfile(file2)

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
    
    if recording and out is None:
        now=datetime.datetime.now() 
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        source_str = 'videos/'+'vid_{}.mp4'.format(str(now).replace(":",''))
        # recorded_video_path = Path(source_str)
        out = cv2.VideoWriter(source_str, fourcc, 20.0, (640, 480))
        #Start new thread for recording the video
        thread = Thread(target = record, args=[out,])
        thread.start()
    elif (not recording) and out is not None:
        out.release()
        out = None
    


@app.post("/uploadvideo")
async def root(file: UploadFile = File(...)):
    os.makedirs(os.path.join("videos"), exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit(process_VidFile(file))
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
    print(trackingPoints)
    incoming_tracked_obj = copy.deepcopy(list(trackingPoints))
    SOT =  True

    
    

# ========================================================================
# Server Runner

if __name__ == "__main__":
    uvicorn.run(app, host="localhost:8000", port=8000, reload=True)