import cv2
import random
import torch
import numpy as np

class Dataloader():
<<<<<<< HEAD
    def __init__(self, image_paths, batch_size=1, shuffle = True): 
=======
    def __init__(self, image_paths, batch_size=1, shuffle=True): 

>>>>>>> 09eef5c5f39cbf0ac993056c6e2666ee23790dbd
        with open(image_paths) as f:
            image_paths = f.read().split('\n')
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
        
        images = []
        labels = []
        for i, path in enumerate(data_path):
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
<<<<<<< HEAD
            image = np.expand_dims(image, -1)
            label = int(path.split('\\\\')[-2])
            

            x = torch.tensor(image/255.0, dtype= torch.float32 ).cuda().permute(2,0,1)
            y = torch.tensor(int(label), dtype= torch.float32).cuda()
            
=======
            # image = np.expand_dims(image, -1)

            # label = int(path.split('\\\\')[-2]) 
            label = int(path.split("\\")[3:4][0])
            x = torch.tensor(image/255.0, dtype= torch.float32 ).permute(2,0,1)
            y = torch.tensor(int(label), dtype= torch.long)
>>>>>>> 09eef5c5f39cbf0ac993056c6e2666ee23790dbd
            if i == 0:
                #global XAll, YAll
                #unsqueeze adds dimension to tensor
                XAll = x.clone().unsqueeze(0)
                YAll = y.clone().unsqueeze(0)
            else:
                #torch.cat combines 2 tensor with dimension if dim is 0 it combines column wise if dim is 1 it combines it row wise
                XAll = torch.cat((XAll, x.clone().unsqueeze(0)), dim=0)
                YAll = torch.cat((YAll, y.clone().unsqueeze(1)), dim=0)
        
            try:
                return XAll, YAll
            except:
                print("error")

        return images, labels