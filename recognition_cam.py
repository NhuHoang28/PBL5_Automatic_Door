import socket
import cv2
import sqlite3
import os
from facenet_pytorch import MTCNN
import torch
import pyrebase
from PIL import Image
import time
from pathlib import Path
import os
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)  
resnet = InceptionResnetV1(pretrained='vggface2').eval() 

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnnd = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)
fontface=cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

def getProfile(id):
    conn = sqlite3.connect('C:\\Users\\ththo\\Downloads\\Nhan dien guong mat\\PBL5_Automatic_Door') 
    query = "SELECT * FROM home_people WHERE ID=" + str(id)
    cursor=conn.execute(query)
    profile=None
    for row in cursor:
        profile = row
    conn.close()
    return profile

while (True): 
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        face, prob = mtcnn(frame, return_prob=True)  
        emb = resnet(face.unsqueeze(0)).detach()  
        
        saved_data = torch.load('data.pt')  
        embedding_list = saved_data[0]
        name_list = saved_data[1] 
        dist_list = []  
        
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)
            
        idx_min = dist_list.index(min(dist_list))
        label=name_list[idx_min]
        confidence=min(dist_list)    

        for box in boxes:
            bbox = list(map(int,box.tolist()))
            # [trai:trên,trai:phai]
            cv2.rectangle(frame,(bbox[0]-7,bbox[1]-7),(bbox[2]+7,bbox[3]+7),(0,255,0),2)
            try:
                # gray= cv2.resize(gray[bbox[1]:bbox[3],bbox[0]:bbox[2]],(100,100)) 
                # gray= gray[tren:duoi,trai:phai]
                gray= gray[bbox[1]-7:bbox[3]+7,bbox[0]-7:bbox[2]+7]
            except Exception as e:
                print()
            # test_img = cv2.imread('dataSet/User.0.0.jpg', cv2.IMREAD_GRAYSCALE)

            print(label,confidence)
            
            if confidence<0.6:
                profile=getProfile(label)
                # profile=1
                if(profile!=None): 
                    #ket qua nhan dang la duoc
                    Result=True
                    # current_time = datetime.now()
                    # cv2.imwrite('dataSet/'+ str(current_time)+'.jpg', frame)
                    # Record(id)
                    # cv2.putText(frame,"unknow1",(bbox[0]+10,bbox[1]+bbox[3]-80),fontface,1,(0,0,255),2)
                    
                    #-----------
                    cv2.putText(frame, "Name:" + str(profile[0]), (bbox[0] +10, bbox[1] + bbox[3]+30), fontface, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Age:" + str(profile[1]), (bbox[0]+10 , bbox[1] + bbox[3] +60), fontface, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Gender:" + str(profile[2]), (bbox[0] +10, bbox[1] + bbox[3]  +90), fontface, 1, (0, 255, 0), 2)
            # --------------đợi 1s -> mở cửa (mình đóng lại) -> đợi 3s để đóng cửa lại  
            else:
                cv2.putText(frame,"unknow",(bbox[0]+10,bbox[1]+bbox[3]-80),fontface,1,(0,0,255),2)
        
    cv2.imshow('Image',frame)
    if (cv2.waitKey(1)== ord('q')):
            break
        
cv2.destroyAllWindows()