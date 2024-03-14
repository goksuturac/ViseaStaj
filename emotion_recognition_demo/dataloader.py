import cv2
import random
import torch
import numpy as np

class Dataloader:
   
    def __init__(self, image_paths, batch_size=1, shuffle = True): 
        with open(image_paths) as f:
            image_paths = f.read().split('\n')
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.shuffle = shuffle 
        
    def __len__(self):
        return len(self.image_paths)
    
    def __on_epoch_end__(self):
        if self.shuffle:
            print("on epoch end")
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
                XAll = x.clone().unsqueeze(0)
                YAll = y.clone().unsqueeze(0)
            else:
                XAll = torch.cat((XAll, x.clone().unsqueeze(0)), dim=0)
                YAll = torch.cat((YAll, y.clone().unsqueeze(1)), dim=0)
        
            try:
                return XAll, YAll
            except:
                print("error")