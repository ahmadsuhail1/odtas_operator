# imports for working with FASTAPI
from fastapi import FastAPI, UploadFile, File, Request, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil
from pydantic import BaseModel
import copy

# imports for loading the YOLO Model
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import uvicorn
import asyncio
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
from PIL import Image as im

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

camera_switch = False
recording = False
detection_switch = False
recorder_frame = False
out = None
tracker = None
alert_class = None
is_alarm = False
tracking_ids = []
global detection_obj, model_all_names

detection_obj = ObjectDetection("http://192.168.137.143:4747/video", "yolov5n.pt")
# detection_obj = ObjectDetection("http://192.168.18.211:4747/video", "yolov5n.pt")


detection_obj.start()
# # --------------------------- functions to run detection -----------------------------

yolo_weights=WEIGHTS / 'yolov5n.pt'
strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt'
config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml'
nr_sources = 1
half = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hide_labels = False
hide_class = False
hide_conf = False
conf_thres=0.45,  # confidence threshold
iou_thres=0.50,  # NMS IOU threshold
device = select_device(device)
WEIGHTS.mkdir(parents=True, exist_ok=True)
VIDEOS.mkdir(parents=True,exist_ok=True)

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
    
    


@torch.no_grad()
def generate_frames():
    global tracker, outputs, tracking_ids, alert_class
    global detection_obj, detection_switch, camera_switch, tracking_switch, model_all_names, recorder_frame, is_alarm
    names = detection_obj.model.names
    model_all_names = list(names.keys())
    
    while True:     
        if camera_switch:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            real_frame = detection_obj.read()
            
            if recording:
                real_frame = cv2.putText(real_frame,"Recording...", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                recorder_frame = real_frame.copy()
                
            # =======================================

            if detection_switch and not tracking_switch:
                
                results = detection_obj.score_frame(real_frame)
                
                real_frame = detection_obj.plot_boxes(results, real_frame)

                if  is_alarm and (alert_class in results[0].numpy().astype(int)) :
                    sound_alarm()
            # =======================================
            
            
           
            if detection_switch and tracking_switch:
                 # # process the frames
                annotator = Annotator(real_frame, line_width=2, pil=not ascii, example = str(names))
            
                results = detection_obj.model(real_frame)

                det = results.pred[0]


                if det is not None and len(det):
                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]
                    
                    if tracking_switch:
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
                
                if tracking_switch:
                    real_frame = annotator.result()

            real_frame = cv2.resize(real_frame, (840, 840))
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
        
        subprocess.run(["python", "yolov5/detect.py", "--weights", "best.pt",
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
    global camera_switch, detection_obj, detection_switch, tracking_switch, recording, recorder_frame, out
    camera_switch = data.camera
    detection_switch = data.detection
    tracking_switch = data.tracking
    recording = data.recording
    
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
    


# ========================================================================
# Server Runner

if __name__ == "__main__":
    uvicorn.run(app, host="localhost:8000", port=8000, reload=True)