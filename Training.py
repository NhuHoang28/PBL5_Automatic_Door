from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import sqlite3
import base64

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)  
resnet = InceptionResnetV1(pretrained='vggface2').eval()  

def conver_image_into_binary(filename):
    with open(filename, 'rb') as file:
        photo_image = base64.b64encode(file.read())
    return photo_image

def getProfile(id):
    conn = sqlite3.connect('E:\\University_course\\Nam4Ky1\\Ma nguon mo\\PBL5_Automatic_Door\\db.sqlite3') 
    query = "SELECT * FROM home_people WHERE ID=" + str(id)
    cursor=conn.execute(query)
    profile=None
    for row in cursor:
        profile = row
    conn.close()
    return profile

def update(ID):
    conn = sqlite3.connect('E:\\University_course\\Nam4Ky1\\Ma nguon mo\\PBL5_Automatic_Door\\db.sqlite3') 
    update_photo   = conver_image_into_binary("dataset/"+str(ID)+"/User."+ID+'.1.jpg')
    conn.execute(""" UPDATE home_people SET image=(?) WHERE ID=(?)""",(update_photo,ID))
    conn.commit()
    conn.close()
        

def insert(ID,name):
    conn = sqlite3.connect('E:\\University_course\\Nam4Ky1\\Ma nguon mo\\PBL5_Automatic_Door\\db.sqlite3') 
    insert_photo   = conver_image_into_binary("dataset/"+str(ID)+"/User."+ID+'.1.jpg')
    conn.execute(""" INSERT INTO home_people ("ID", "Name", "image") values (?,?,?)""",(str(ID), name, insert_photo))
    conn.commit()
    conn.close()

dataset=datasets.ImageFolder('dataset')  
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} 

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

face_list = []  
name_list = []  
embedding_list = []  

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True) 
    if face is not None and prob>0.90:  
        emb = resnet(face.unsqueeze(0))  
        embedding_list.append(emb.detach()) 
        name_list.append(idx_to_class[idx]) 
        profile=getProfile(idx_to_class[idx])
        if(profile!=None): 
            update(idx_to_class[idx])
        else:
            name = input("Enter name of id %s " %idx_to_class[idx])
            insert(idx_to_class[idx], name)
data = [embedding_list, name_list]
torch.save(data, 'data.pt')  