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

def getProfile(id):
    conn = sqlite3.connect('C:\\Users\\ththo\\Desktop\\PBL5\\AdminWeb\\db.sqlite3') 
    query = "SELECT * FROM home_people WHERE ID=" + str(id)
    cursor=conn.execute(query)
    profile=None
    for row in cursor:
        profile = row
    conn.close()
    return profile
def InsertRecord(people_id, checkpeople,ok):
    conn = sqlite3.connect('C:\\Users\\ththo\\Desktop\\PBL5\\AdminWeb\\db.sqlite3') 
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    query = "Insert into home_checkpeople(checkpeople,ok,id_check,time) values('"+str(checkpeople)+"','"+str(ok)+"','"+str(people_id)+"','"+current_time+ "')"
    conn.execute(query)
    conn.commit()
    conn.close()

Result="Close"
# os.remove('anhne.jpg')
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client.connect(('10.10.26.74', 1002))

# client.send(str.encode("634769708663"))
print("da gui RFID")

while (True): 
    path_to_file = 'User.9.13.jpg'
    path = Path(path_to_file)

    if path.is_file():
        time.sleep(1)
        img = Image.open(path)
        frame = cv2.imread("User.9.13.jpg", cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        face, prob = mtcnn(img, return_prob=True)  
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
              
        if True:
            boxes, _ = mtcnnd.detect(frame)
            if boxes is not None:
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
                    
                    if confidence<42:
                        profile=getProfile(label)
                        if(profile!=None): 
                            #ket qua nhan dang la duoc
                            Result="Open"
                            
                            # Record(id)
                            InsertRecord(label,1,1)
                            #-----------
                            cv2.putText(frame, "Name:" + str(profile[0]), (bbox[0] +10, bbox[1] + bbox[3]+30), fontface, 1, (0, 255, 0), 2)
                            cv2.putText(frame, "Age:" + str(profile[1]), (bbox[0]+10 , bbox[1] + bbox[3] +60), fontface, 1, (0, 255, 0), 2)
                            cv2.putText(frame, "Gender:" + str(profile[2]), (bbox[0] +10, bbox[1] + bbox[3]  +90), fontface, 1, (0, 255, 0), 2)
                    # --------------đợi 1s -> mở cửa (mình đóng lại) -> đợi 3s để đóng cửa lại  
                    else:
                        InsertRecord(label,1,0)
                        cv2.putText(frame,"unknow",(bbox[0]+10,bbox[1]+bbox[3]-80),fontface,1,(0,0,255),2)
        
        cv2.imshow('Image',frame)
        print("Chuan bi gui ket qua")
        # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # client.connect(('10.10.26.74', 1002))
        # client.send(str.encode(Result))
        # print("Da gui ket qua ")
        # client.close()
        # Result ="Close"
    if (cv2.waitKey(1)== ord('q')):
        break  
    #nhan anh tu module camera
    
    # s = socket.socket()
    # s.bind(("10.10.27.133",1003))
    # s.listen(10) # Accepts up to 10 incoming connections..

    # print('Server opened, waiting for image...')
    # # while (True):
    # sc, address = s.accept()
    # #dung 5s
    # print("Recieve from IP: " )
    
    # print(address)
    # f = open('anhne' + ".jpg",'wb') # Open in binary

    # # We receive and write to the file.
    # l = sc.recv(1024)
    # while (l):
    #     f.write(l)
    #     l = sc.recv(1024)
    # l = sc.recv(2048)
    # with open("anhne.jpg", "wb") as f:
    #     print("Open file")
    #     while(l):
    #         f.write(l)
    #         l = sc.recv(2048)
    #     print("Close file")
    # # f.close()
    # sc.close()
    # s.close()
    
    
    #-------------------
        
    # else:
    #     print("Wait module camera send image")
    #     time.sleep(6)
# cap.release()
cv2.destroyAllWindows()