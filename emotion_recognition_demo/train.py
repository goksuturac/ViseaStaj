import torch
import torch.nn as nn
import torch.optim as optim
import dataloader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from model import CustomVGG16

train_loader = dataloader.Dataloader()

num_classes = 7 
model = CustomVGG16(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

num_epochs = 1000
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
        if i % 1000 == 9:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Eğitim tamamlandı')
torch.save(model.state_dict(), 'custom_vgg16.pth')
