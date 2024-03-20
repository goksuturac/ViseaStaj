import torch 
from dataloader import Dataloader
from torchvision import models
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data_path = "ViseaStaj/emotion_recognition_demo/dataset/train"


train_loader = Dataloader(image_paths = train_data_path, batch_size = 32, shuffle = True)

model = models.vgg16(pretrained=True)

inlayer = nn.Conv2d(1, 34, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
outlayer = nn.Sequential(nn.Linear(4096, 7, bias = True), nn.Softmax(dim=1))

model.features[0] = inlayer
model.classifier[-1] = outlayer
model.cuda()

optimizer =torch.optim.Adam(model.parameters(), lr=0.0001)
loss_func = nn.CrossEntropyLoss().cuda()

for epoch in range(100):
    loss = 0
    model.train()

    for i in range(len(train_loader)):
        input,label = train_loader[i]
        optimizer.zero_grad()
        outputs= nn.softmax(model(input), dim =1)
    
        loss= loss_func(outputs, label.long())
        
        loss += loss
        loss.backward()
        optimizer.step()

        
        
        print("Epoch: %d loss: %.2f avg_loss: %.2f Train Accuracy: %.2f Train f1 score: %.2f  Train precision:%.2f  Train recall: %.2f " %(epoch, loss, avg_loss/len(train_loader), accuracy,f1_score, precision, recall), end="\r")
        






# import torch.nn as nn
# import torchvision.models as models

# class CustomVGG16(nn.Module):
#     def __init__(self, num_classes):
#         super(CustomVGG16, self).__init__()
#         self.vgg16 = models.vgg16(pretrained=True)
#         #vgg16 imagenet veriseti kullanılarak eğitilmiştir bundan dolayı 1000 farklı sınıflıdır. 
#         self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

#         # Eğitilecek parametreleri belirleyin (önceden eğitilmiş katmanların ağırlıklarını dondurun)
#         # for param in self.vgg16.features.parameters():
#             # param.requires_grad = False

#     def forward(self, x):
#         return self.vgg16(x)

