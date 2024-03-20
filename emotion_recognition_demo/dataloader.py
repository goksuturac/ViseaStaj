import cv2
import random
import torch
import numpy as np
import os 

class Dataloader():
   
    def __init__(self, image_paths, batch_size=32, shuffle = True): 
        image_paths = [os.path.join(image_paths, file) for file in os.listdir(image_paths) if file.endswith('.jpg')]

        self.image_paths = image_paths
        self.batch_size = batch_size
        self.shuffle = shuffle 
        
    def __len__(self):
        return len(self.image_paths)
    
    def __on_epoch_end__(self):
        if self.shuffle:
            print("end of epoch")
            random.shuffle(self.image_paths)
    
    def __getitem__(self, idx):
        data_path = self.image_paths[idx * self.batch_size : self.batch_size*idx + self.batch_size]
        
        for i, path in enumerate(data_path):
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (224, 224))
            image = np.expand_dims(image, -1)

            label = int(path.split('\\\\')[-2])
            

            x = torch.tensor(image/255.0, dtype= torch.float32 ).cuda().permute(2,0,1)
            y = torch.tensor(int(label), dtype= torch.float32).cuda()
            
            if i == 0:
                X = x.clone().unsqueeze(0)
                Y = y.clone().unsqueeze(0)
            else:
                #torch.cat() can be seen as an inverse operation for torch.split() and torch.chunk()
                X = torch.cat((X, x.clone().unsqueeze(0)), dim=0)
                Y = torch.cat((Y, y.clone().unsqueeze(1)), dim=0)
            try:
                return X, Y
            except:
                print("error")