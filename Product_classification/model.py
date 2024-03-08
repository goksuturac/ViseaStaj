import torch
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import pandas as pd

# print("cuda") if torch.cuda.is_available() else print("cpu")

# print(f"torch version: {torch.__version__} \ntorchvision version: {torchvision.__version__}")


#pytorch içinden kütüphane çektik
train = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None,
)
test = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


image, label = train[0]
# print(image, label)
print(image.shape)
print(len(train), len(test))
print(len(train.data), len(train.targets), len(test.data), len(test.targets)
)

#veri içindeki sınıfları öğrendik.
class_names = train.classes
print(class_names)

# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.show()

torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows= 5
columns= 5
for i in range (1, rows * columns+ 1):
    random_id = torch.randint(0,len(train),size=[1]).item()
    image, label= train[random_id]
    fig.add_subplot(rows, columns, i)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
plt.show()


batch_size = 32

train_dataloader = DataLoader(train,
                              batch_size,
                              shuffle=True)


test_dataloader = DataLoader(test,
                             batch_size,
                             shuffle=False)



print(f"Dataloaders: {train_dataloader, test_dataloader}") 
print(f"Length of train dataloader: {len(train_dataloader)} batches of {batch_size}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {batch_size}")

train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)

# MODEL BUILDING

flatten_model = nn.Flatten()
x = train_features_batch[0]

output= flatten_model(x)

print(f"before flattening: {x.shape} -> [color_channels, height, width]")
print(f"after flattening: {output.shape} -> [color_channels, height*width]")

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int,  hidden_layers: int, output_shape: int):
        super().__init__()
        layers = []
        layers.append(nn.Flatten())
        
        # ilk gizli katman
        layers.append(nn.Linear(in_features=input_shape, out_features=hidden_units))
        layers.append(nn.Softmax()) 
        
        # sonraki gizli katmanlar
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(in_features=hidden_units, out_features=hidden_units))
            layers.append(nn.Softmax())  
        
        # çıkış katmanı
        layers.append(nn.Linear(in_features=hidden_units, out_features=output_shape))
        
        self.layer_stack = nn.Sequential(*layers)
    def forward(self, x):
        return self.layer_stack(x)


model = FashionMNISTModelV0(input_shape=784,
                            hidden_layers=3,
                            hidden_units=20,
                            output_shape=len(class_names)
)


#model özeti yazdırıldı
print(model)

#model parametreleri yazdırıldı
# print(model.state_dict())


import requests
from pathlib import Path 
#vision kursunda bulunan dosya eklendi.
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)
    
    
from helper_functions import accuracy_fn 
# optimizasyon işlemlerinin yapılması.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)