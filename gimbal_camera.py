import cv2
import importlib
from mmtrack.apis import inference_sot, init_model
from simplebgc.gimbal import Gimbal, ControlMode
from pathlib import Path

device =  importlib.import_module("python-capture-device-list.device")

def select_camera(last_index):
    
    number = 0
    # hint = "Select a camera (0 to " + str(last_index) + "): "
    # try:
    #     # number = int(input(hint))
    #     # number = 1
    #     # select = int(select)
    # except Exception:
    #     print("It's not a number!")
    #     return select_camera(last_index)

    # if number > last_index:
    #     print("Invalid number! Retry!")
    #     return select_camera(last_index)

    return number


def open_camera(index):
    cap = cv2.VideoCapture(index)
    return cap


# @torch.no_grad
def main():
    # print OpenCV version
    print("OpenCV version: " + cv2.__version__)
    show = True
    input_number = 1
    sot_config_model = Path('configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py')
    sot_checkpoint_model = Path('checkpoints/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth')
    config_path = Path(__file__).parent / sot_config_model
    checkpoint_path = Path(__file__).parent / sot_checkpoint_model
    
    print(config_path)
    print(checkpoint_path)
    
    # config = "mmtracking/configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py"
    # checkpoint = "mmtracking/checkpoints/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth"
    # Get camera list
    device_list = device.getDeviceList()
    # print(device_list)
    index = 0



    for camera in device_list:
        # print(str(index) + ': ' + camera[0] + ' ' + str(camera[1]))
        index += 1

    last_index = index - 1

    if last_index < 0:
        print("No device is connected")
        return


    gimbal = Gimbal()
    # degree per sec
    pitch_speed = 30
    yaw_speed = 50
    


    # Select a camera
    # camera_number = select_camera(last_index)
    camera_number = last_index
    

    # load images
    
    cap = open_camera(camera_number)
    _,frame = cap.read()
    rows,cols, _ = frame.shape

    x_medium = int(cols//2)
    y_medium = int(rows//2)

    center_x = int(cols//2)
    center_y = int(rows//2)

    yaw_position = 0
    pitch_position = 0

    YAW_MAX_LIMIT_ANGLE = 165
    YAW_MIN_LIMIT_ANGLE = -165
    PITCH_UPWARD_LIMIT_ANGLE = -10
    PITCH_DOWNWARD_LIMIT_ANGLE = 160

    tracked_obj_exist = False

    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(FRAME_HEIGHT,FRAME_WIDTH)



    # imgs = mmcv.VideoReader(camera_number)
    IN_VIDEO = True

    OUT_VIDEO = False
    fps = 30

    # fps = args.fps
    # if show or OUT_VIDEO:
    #     if fps is None and IN_VIDEO:
    #         # fps = imgs.fps
    #     if not fps:
    #         raise ValueError('Please set the FPS for the output video.')
    #     fps = int(fps)


    # build the model from a config file and a checkpoint file
    model = init_model(config_path, checkpoint_path)

    i = 0

    while True:
        
        print("STARTING POSITION: ", yaw_position , "   " , pitch_position)

        _, frame = cap.read()
    # test and show/save the images
    # for i, img in enumerate(imgs):
        if i == 0:
            init_bbox = list(cv2.selectROI("TESTING", frame, False, False))
            # convert (x1, y1, w, h) to (x1, y1, x2, y2)
            init_bbox[2] += init_bbox[0]
            init_bbox[3] += init_bbox[1]

        

        result = inference_sot(model, frame, init_bbox, frame_id=i)

        i+=1
        

        if len(result) == 0:
            x_medium = int(cols//2)
            y_medium = int(rows//2)
            tracked_obj_exist = False



        if len(result) > 0:
            result_key = result.get('track_bboxes') 
            x1,y1,x2,y2 = int(result_key[0]),int(result_key[1]),int(result_key[2]),int(result_key[3])
            x_medium, y_medium = int((x1 + x2)/2),int((y1 + y2)/2)
            tracked_obj_exist = True


        if tracked_obj_exist:
            cv2.line(frame,(x_medium,0), (x_medium, FRAME_HEIGHT), (255,0,0),2)
            cv2.line(frame,(0,y_medium), (FRAME_WIDTH,y_medium), (255,0,0),2)
        
        # print(result)
        frame = cv2.resize(frame, (780, 540), interpolation = cv2.INTER_LINEAR)
        out_file = None
        model.show_result(
            frame,
            result,
            show=show,
            wait_time=int(1000. / fps) if fps else 0,
            out_file=out_file,
            thickness=2)

        if (yaw_position >= YAW_MIN_LIMIT_ANGLE and yaw_position <= YAW_MAX_LIMIT_ANGLE):
        # and (pitch_position >= PITCH_UPWARD_LIMIT_ANGLE and (pitch_position <= PITCH_DOWNWARD_LIMIT_ANGLE)):
            if x_medium < center_x - 70:
                if yaw_position != YAW_MIN_LIMIT_ANGLE:
                    yaw_position -= 1
                    gimbal.control(
                        pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=0,
                        yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)

            elif x_medium > center_x + 70:
                if yaw_position != YAW_MAX_LIMIT_ANGLE:
                    yaw_position += 1
                    gimbal.control(
                        pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=0,
                        yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=yaw_position)

            else:
                pass
        
        if pitch_position >= PITCH_UPWARD_LIMIT_ANGLE and pitch_position <= PITCH_DOWNWARD_LIMIT_ANGLE:
            if y_medium < center_y - 70:
                
                if pitch_position != PITCH_UPWARD_LIMIT_ANGLE:
                    pitch_position -=1
                    gimbal.control(
                        pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=pitch_position,
                        yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=0)

            elif y_medium > center_y + 70:
                if pitch_position != PITCH_DOWNWARD_LIMIT_ANGLE:
                    pitch_position +=1
                    gimbal.control(
                        pitch_mode=ControlMode.angle, pitch_speed=pitch_speed, pitch_angle=pitch_position,
                        yaw_mode=ControlMode.angle, yaw_speed=yaw_speed, yaw_angle=0)

            else: 
                pass
    

        print(yaw_position , "  " , pitch_position)
    # Open camera
    # cap = open_camera(camera_number)

    # if cap.isOpened():
    #     width = cap.get(3) # Frame Width
    #     height = cap.get(4) # Frame Height
    #     print('Default width: ' + str(width) + ', height: ' + str(height))

    #     while True:
            
    #         ret, frame = cap.read()
    #         cv2.imshow("Camera List and Resolution", frame)

    #         # key: 'ESC'
    #         key = cv2.waitKey(20)
    #         if key == 27:
    #             break

    #     cap.release() 
    #     cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()