import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import Dataloader
from vgg_model import CustomVGG16
from torchmetrics import Precision, Recall, Accuracy, F1Score

# test verisinin yollarının bulunduğu dosyanın yolu
test_file_path = "emotion_recognition_demo\\dataset\\test.txt"

# test verisini yükle
test_loader = Dataloader(image_paths=test_file_path, batch_size=1, shuffle=False)

model_state_dict = torch.load("emotion_recognition_demo/custom_vgg16.pth")

num_classes = 6
new_model = CustomVGG16(num_classes)
new_model.load_state_dict(model_state_dict)
new_model.eval()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(new_model.parameters(), lr=0.0001)

precision_metrics = Precision(task='multiclass', num_classes=6)
recall_metrics = Recall(task='multiclass', num_classes=6)
accuracy_metrics = Accuracy(task='multiclass', num_classes=6)
f1score_metrics = F1Score(task='multiclass', num_classes=6)

for epoch in range(10):
    epoch_loss = 0.0
    
    for input, label in test_loader:
        optimizer.zero_grad()
        
        outputs = torch.softmax(new_model(input), dim=1) 
        loss = loss_func(outputs, label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        precision_metrics.update(outputs, label)
        recall_metrics.update(outputs, label)
        accuracy_metrics.update(outputs, label)
        f1score_metrics.update(outputs, label)
        
    epoch_loss /= len(test_loader)
    
    precision = precision_metrics.compute()
    recall = recall_metrics.compute()
    accuracy = accuracy_metrics.compute()
    f1_score = f1score_metrics.compute()
    
    print("Epoch: %d  Loss: %.2f Test precision: %.2f  Test recall: %.2f Test Accuracy: %.2f  Test f1 score: %.2f" %
          (epoch+1, epoch_loss, precision, recall, accuracy, f1_score))

    precision_metrics.reset()
    recall_metrics.reset()
    accuracy_metrics.reset()
    f1score_metrics.reset()
