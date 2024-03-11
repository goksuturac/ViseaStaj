import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import random
import numpy as np

transform = transforms.Compose([
    #resim yeniden boyutlandırıldı
    transforms.Resize((224,224)),
    #torch içinde kullanılan tensör nesnesine çevrildi
    transforms.ToTensor(),
    #normalizasyon işlemi yapıldı. piksel değerleri 0-1 aralığına indirgendi
    transforms.Normalize((0.5,), (0.5,))
])


train_data = ImageFolder(root='Emotion_recognition\\dataset\\train', transform=transform)
validation_data = ImageFolder(root='Emotion_recognition\\dataset\\test', transform=transform)

batch_size = 32

train_loader = DataLoader(train_data, batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size, shuffle=False)

idx = random.randint(0,10)
tensor = train_data.__getitem__(idx)[0]
image = np.squeeze(tensor.numpy())
image = (image - np.min(image)) / (np.max(image) - np.min(image))
image = image.transpose((1, 2, 0))
plt.imshow(image)
plt.show()
