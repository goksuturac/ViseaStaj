import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from models import CustomVGG16

transform = transforms.Compose([
    #resim yeniden boyutlandırıldı
    transforms.Resize((224,224)),
    #torch içinde kullanılan tensör nesnesine çevrildi
    transforms.ToTensor(),
    #normalizasyon işlemi yapıldı. piksel değerleri 0-1 aralığına indirgendi
    transforms.Normalize((0.5,), (0.5,))
])

validation_data = ImageFolder(root='Emotion_recognition\\dataset\\test', transform=transform)
validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False)

model = CustomVGG16(num_classes=len(validation_data.classes))
model.load_state_dict(torch.load('custom_vgg16.pth'))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in validation_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Doğruluk: %d %%' % (100 * correct / total))
