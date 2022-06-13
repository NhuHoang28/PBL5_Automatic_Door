from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import sqlite3
import base64
def conver_image_into_binary(filename):
    with open(filename, 'rb') as file:
        photo_image = base64.b64encode(file.read())
    return photo_image
def update(ID):
    conn = sqlite3.connect('C:\\Users\\ththo\\Desktop\\PBL563\\AdminWeb\\db1.sqlite3') 
   
     
    isRecordExist = 0
 
    if (isRecordExist == 0):
        insert_photo   = conver_image_into_binary("dataset/"+str(ID)+"/User."+ID+'.1.jpg')
        conn.execute(""" UPDATE home_people SET image=(?) WHERE ID=(?)""",(insert_photo,ID))
 
        conn.commit()
        conn.close()


mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)  
resnet = InceptionResnetV1(pretrained='vggface2').eval()  



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
        update(idx_to_class[idx])
data = [embedding_list, name_list]
torch.save(data, 'data.pt')  