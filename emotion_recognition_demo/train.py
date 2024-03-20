import torch
import os
import torch.nn as nn
import torch.optim as optim
from dataloader import Dataloader
from model import CustomVGG16


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# eğitim verisi yyüklendi
train_loader_path = "emotion_recognition_demo\\dataset\\train.txt"
train_loader = Dataloader(image_paths=train_loader_path, batch_size=1, shuffle=True)

# model oluştu
num_classes = 7
new_model = CustomVGG16(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(new_model.parameters(), lr=0.0001)

# train 
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = new_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Eğitim tamamlandı')
torch.save(new_model.state_dict(), 'custom_vgg16.pth')
