from multiprocessing import Lock
from pathlib import Path
import cv2

from threading import Thread
import threading
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import sys


# from yolov5.utils.plots import Annotator
# adding Folder_2 to the system path
path = sys.path.insert(0, '/yolov5')
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
print(ROOT)

from yolov5.utils.general import (LOGGER, check_img_size, cv2)




class ObjectDetection:
    PATH = "D:\\Workspace\\FYP\\Development\\campaign-manager\\backend\\yolov5"


    def __init__(self, capture_index, model_name, imgsz = (864,864)):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        # initializing  variables for video camera stream 
        
        self.capture_index = str(capture_index)
        self.is_url = self.capture_index.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = self.capture_index.isnumeric()
        
        # Loading Model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model(model_name)
        self.stride, self.classes, self.pt  = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        
        # print(self.get_video_capture())
        self.cap = self.get_video_capture()
        # print(self.cap)
        assert self.cap.isOpened()

        self.grabbed, self.frame = self.cap.read()

        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.stopped = True

        self.t = Thread(target=self.update, args=())
        
        self.t.daemon = True

    def start(self):
        self.stopped = False
        self.t.start()

    # Changes Above =================
    
    
    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        
        
        
        
        
        
        print(self.capture_index,'--------------------------------------------')
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """

        if model_name:
            print("model name reached")
            global PATH
        
            model = torch.hub.load(self.PATH,
                                   'custom',
                                   path="D:\\Workspace\\FYP\\Development\\campaign-manager\\backend\\yolov5s.pt",
                                   source='local',
                                   force_reload=True
                                   )
            
            model.eval()
        
            # print(model)
            print("DONE")
        else:
    
            model = torch.hub.load('ultralytics/yolov5','yolov5n', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, label):
        """
        For a given label value, return corresponding string label.
        :param label: numeric label
        :return: corresponding string label
        """
        return self.classes[int(label)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        print(f"[INFO] Total {n} detections. . . ")
        print(f"[INFO] Looping through all detections. . . ")

        # looping through the detections
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(
                    row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                text_d = self.class_to_label(labels[i])
                
                if text_d == "Person":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)    
                    cv2.putText(frame, text_d, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                elif text_d == "LTV":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)    
                    cv2.putText(frame, text_d, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                elif text_d == "HTV":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)    
                    cv2.putText(frame, text_d, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                elif text_d == "Boat":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)    
                    cv2.putText(frame, text_d, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)    
                    cv2.putText(frame, text_d, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)     
                    
        return frame

    def update(self):
        print("%d : Thread in targeted thread action", threading.get_ident())
        while True:

            #  if self.stopped is false, then we are reading the next frame
            if not self.stopped:
                self.grabbed, self.frame = self.cap.read()

                if self.grabbed is False:
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.frame = np.zeros_like(self.frame)
                    self.cap.open(self.capture_index)
                    # print('[Exiting] No more frames to read')
                    # self.stopped = True
                    # break
            else:
                pass
        # self.cap.release()

    # method for returning latest read frame
    def read(self):
        return self.frame
    
    def resume(self):
        self.stopped = False
    
    def stop(self):
        self.stopped = True
