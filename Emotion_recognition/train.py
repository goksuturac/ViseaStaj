import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from models import CustomVGG16

transform = transforms.Compose([
    #torch içinde kullanılan tensör nesnesine çevrildi
    transforms.ToTensor(),
    #normalizasyon işlemi yapıldı. piksel değerleri 0-1 aralığına indirgendi
    transforms.Normalize((0.5,), (0.5,))
])

train_data = ImageFolder(root='Emotion_recognition\\dataset\\train', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

num_classes = len(train_data.classes)
model = CustomVGG16(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
