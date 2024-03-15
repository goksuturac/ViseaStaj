import torch.nn as nn
import torchvision.models as models

class CustomVGG16(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        #vgg16 imagenet veriseti kullanılarak eğitilmiştir bundan dolayı 1000 farklı sınıflıdır. 
        self.vgg16.classifier[7] = nn.Linear(4096, num_classes)

        # Eğitilecek parametreleri belirleyin (önceden eğitilmiş katmanların ağırlıklarını dondurun)
        # for param in self.vgg16.features.parameters():
            # param.requires_grad = False

    def forward(self, x):
        return self.vgg16(x)

