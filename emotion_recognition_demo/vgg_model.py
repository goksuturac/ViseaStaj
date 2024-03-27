import torch.nn as nn
import torchvision.models as models
import warnings

# vgg16(weights=True) kısmı hakkında update uyarısı veriyordu. ignore edildi.
warnings.filterwarnings('ignore')

class CustomVGG16(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG16, self).__init__()
        self.vgg16 = models.vgg16(weights=True)
        # self.vgg16.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        #vgg16 imagenet veriseti kullanılarak eğitilmiştir bundan dolayı 1000 farklı sınıflıdır. 
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

        # for param in self.vgg16.features.parameters():
            # param.requires_grad = False

    def forward(self, x):
            return self.vgg16(x)


