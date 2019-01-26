# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 18:54:30 2019

@author: DOLPHIN
"""
import numpy as np
import cv2
import pickle


face_cascade = cv2.CascadeClassifier("G:\python modules\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("G:\python modules\opencv\sources\data\haarcascades\haarcascade_eye.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
    o_labels = pickle.load(f)
    print(o_labels)
    labels = {v:k for k,v in o_labels.items()}
#cap = cv2.VideoCapture(0)
imgpath = "G:\\PROJECTS\\face recognition\\test4.jpg"

if(True):
    #ret,frame = cap.read()
    #if (not ret):
        #break
    frame = cv2.imread(imgpath,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(face_cascade.empty())
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,minNeighbors=3)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        
        id_ , conf = recognizer.predict(roi_gray)
        if conf :
            print(id_, labels[id_])
            
            #img_item = "my-image.png"
            #cv2.imwrite(img_item,roi_gray)
            #cv2.imwrite("color.png",roi_color)
            cv2.putText(frame,(str(round(conf,1))+" " + str(labels[id_])),(x-5,y-5),cv2.FONT_HERSHEY_SIMPLEX,.5,color=(0,0,255))
            cv2.rectangle(frame,(x,y),(x+w,y+h),color=(255,255,255),thickness=2)
            eyes = eye_cascade.detectMultiScale(frame,scaleFactor = 1.3)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),color=(0,255,0),thickness = 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyWindow('frame')
    #if cv2.waitKey(1) & 0xFF==ord('q'):
        #break
    
#cap.release()
#cv2.destroyAllWindows()