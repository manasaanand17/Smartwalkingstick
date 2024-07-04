
import serial
import time
import cv2
import numpy as np
import time
import pyttsx3  
import torch
import matplotlib.pyplot as plt 
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                        increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
import datetime
from playsound import playsound

arduino_port = '/dev/cu.usbserial-1120'  
arduino = serial.Serial(port=arduino_port, baudrate=9600, timeout=.1)
alert_sound_path = "alert_sound.mp3"  

model=torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')
cap = cv2.VideoCapture(0)
cnt=0
detected = {}
start_time=time.time()
if (cap.isOpened()== False):
    print("Error opening video file")
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        i=0
        frame=cv2.resize(frame,(640,480))
        res=model(frame)
        cv2.imshow('Frame', np.squeeze(res.render()))
        cnf=res.pandas().xyxyn[0]['confidence']
        lbl=res.pandas().xyxy[0]['name']

        data = arduino.readline().decode().strip()  
        if (type(data) == str) and (data != ""):
            data = int(data)
            engine = pyttsx3.init() 
            if data < 50:
                if len(lbl)>0:
                    center_x = frame.shape[1] / 2
                    object_x = (res.pandas().xyxy[0]['xmin'][0] + res.pandas().xyxy[0]['xmax'][0]) / 2
                    if object_x < center_x:
                        direction = "left"
                    else:
                        direction = "right"
                    name = lbl[i]
                    confidence = cnf[i]
                    if confidence>0.30:
                        if name not in detected:
                            detected[name] = time.time()
                            print(name, confidence)
                            # print(data)
                            print("\n")
                            playsound(alert_sound_path) 
                            engine.say("There is a "+name+" infront of you. Turn {} to avoid it.".format(direction)) 
                            engine.runAndWait()   
                        elif name in detected:
                            if time.time() - detected[name] > 10:  
                                detected[name] = time.time()
                                print(name + " (detected again)", confidence)
                                # print(data)
                                print("\n")
                                playsound(alert_sound_path) 
                                engine.say("There is a " + name + " infront of you, again. Turn {} to avoid it.".format(direction))  
                                engine.runAndWait()  
                        i = i+1
                else:
                    playsound(alert_sound_path) 
                    engine.say("There is something infront of you.") 
                    engine.runAndWait()
                # print(res.pandas().xyxy[0]['name'][i])
                # print(res.pandas().xyxyn[0]['confidence'][i])
                # print("\n")
                # cnt+=1

        # if cnt>3:
    #         time_stp=str(datetime.datetime.now())
    #         rl_db.child(unq_key).child(time_stp[:18]).set({"detected state":str(res.pandas().xyxy[0]['name'])})
    #         cv2.imwrite("temp.jpg",np.squeeze(res.render()))
    #         stg.child("images/"+unq_key+time_stp.split()[0]).put("temp.jpg")
    #         #img_url=stg.child("images/"+unq_key+time_stp.split()[0]).get_url(token=None)
    #         img_url="https://firebasestorage.googleapis.com/v0/b/fall-det.appspot.com/o/images"+f"%2F"+unq_key+time_stp.split()[0]+"?alt=media"
    #         sendnotification.send(img_url)
    #         cnt=0
    #         break
            if cv2.waitKey(25) & 0xFF == ord('q'):
                exit(0)
    # else:
    #     break
    # end_time=time.time()
    # if(end_time-start_time>=60):
    #     time_stp=str(datetime.datetime.now())
    #     rl_db.child(unq_key).child(time_stp[:18]).set({"detected state":str(res.pandas().xyxy[0]['name'])})
    #     start_time=end_time
cap.release()
cv2.destroyAllWindows()



