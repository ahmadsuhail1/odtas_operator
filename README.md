This project is a prototype and is in development phase. 
ODTAS is a desktop and computer vision based system aimed at solving aerial surveillance problems.

It includes modules like object detection, object tracking (SOT and MOT), picking right capturing device, save and upload videos for detections.

This repository only includes the backend of the operator side. 

The front-end repository of operator and admin portal repository (web based) will be merged later.

The desktop application for operator is built using Next JS and Electron JS.
Front-end of Drone Operator: https://github.com/uzair-muaz/Nextron_Odtas.git
Admin Web Portal: https://github.com/UmerDfarooq/ODTAS.git

# Minimum Requirements

You will need Python >= 3.8, NVIDIA graphic card, CUDA, C++ Windows Build Tools and PyTorch > 1.7 to run this application.

# 1. Installation Process

## 1.1. If you are using HDMI connected camera

1. Go to python-capture-device-list folder
2. Follow the instructions to build the c++ library for accessing the HDMI connected camera.

## 1.2. Install Pysot
1. Run the requirements.txt file of Pysot library
2. Download a model from their Model Zoo
3. Put the downloaded model in the experiment/your_downloaded_model_weight folder


## 1.3 Install SimpleBGC

1. Go to https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers?tab=downloads and download the driver for windows version 11.2.0
2. This driver is for communicating with the Gimbal device through serial programming.

## 1.4 Installing the dependencies for YOLOv5

1. Go to Yolov5 folder 
2. Follow the installation process mentioned in the Yolov5 Readme file.



Stay tuned for more updates. Thankyou.



