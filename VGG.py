 
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.python.keras
from tensorflow.python.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
 


mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)  
resnet = InceptionResnetV1(pretrained='vggface2').eval()  


dataset=datasets.ImageFolder('anh lung tung')  
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} 

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

face_list = []  
name_list = []  
embedding_list = []  

#Lay dl
for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True) 
    if face is not None and prob>0.90:  
        face_list.append(face)
        name_list.append(idx_to_class[idx]) 
#Chia bo test
(trainX, testX, trainY, testY) = train_test_split(face_list,
	name_list, test_size=0.3, stratify=name_list, random_state=42)    
 #train bo train
for face in trainX:
    emb = resnet(face.unsqueeze(0))   
    embedding_list.append(emb.detach() )  
    
data = [embedding_list, name_list]
 
torch.save(data, 'data.pt')  


# predictions = []
# for i in range(0, len(testX)):
#     # classify the face and update the list of predictions and
# 	# confidence scores
#     dist_list = []
#     emb = resnet(testX[i].unsqueeze(0)).detach()  
#     for idx, emb_db in enumerate(embedding_list):
#         dist = torch.dist(emb, emb_db).item()
#         dist_list.append(dist)
#     idx_min = dist_list.index(min(dist_list))
#     predictions.append(trainY[idx_min])
# # show the classification report
# print(classification_report(testY, predictions))

#svm  

