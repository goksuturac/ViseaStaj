import torch
import os
import data_path
import torch.nn as nn
import torch.optim as optim
from dataloader import Dataloader
from model import CustomVGG16

# eğitim verisi yyüklendi
train_loader_path = data_path.train_image_paths
train_loader = Dataloader(image_paths=train_loader_path, batch_size=32, shuffle=True)

# model oluştu
num_classes = 7 
model = CustomVGG16(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# train 
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Eğitim tamamlandı')
torch.save(model.state_dict(), 'custom_vgg16.pth')
