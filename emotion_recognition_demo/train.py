import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import Dataloader
from vgg_model import CustomVGG16
<<<<<<< HEAD
=======
from torchmetrics import Precision, Recall, Accuracy, F1Score

# eğitim verisinin yollarının bulunduğu dosyanın yolu
train_file_path = "emotion_recognition_demo\\dataset\\train.txt"
>>>>>>> 09eef5c5f39cbf0ac993056c6e2666ee23790dbd

# eğitim verisini yükle
train_loader = Dataloader(image_paths=train_file_path, batch_size=1, shuffle=True)

<<<<<<< HEAD
# eğitim verisi yyüklendi
train_loader_path = "emotion_recognition_demo\\dataset\\train.txt"
train_loader = Dataloader(image_paths=train_loader_path, batch_size=1, shuffle=True)

# model oluştu
num_classes = 6
new_model = CustomVGG16(num_classes)
=======
num_classes = 7
#modeli tanımla
new_model = CustomVGG16(num_classes)
#loss ve optimizer belirledik
>>>>>>> 09eef5c5f39cbf0ac993056c6e2666ee23790dbd
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(new_model.parameters(), lr=0.0001)

# metrikler
precision_metrics = Precision(task='multiclass', num_classes = num_classes)
recall_metrics = Recall(task='multiclass', num_classes = num_classes)
accuracy_metrics = Accuracy(task='multiclass', num_classes = num_classes)
f1score_metrics = F1Score(task='multiclass', num_classes = num_classes)

# eğitim döngüsü
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    new_model.train()

    # eğitim verisini dolaş
    """buraya enumerate ||| in range """
    for i , (input, label) in enumerate(train_loader):
    # for i in range(len(train_loader)):
        input,label = train_loader[i]
    # for input, label in train_loader:
        # modeli eğit
        optimizer.zero_grad()
<<<<<<< HEAD
        outputs = new_model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Eğitim tamamlandı')
=======

        #forward pass
        outputs = new_model(input)
        loss = loss_func(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        precision_metrics.update(outputs, label)
        recall_metrics.update(outputs, label)
        accuracy_metrics.update(outputs, label)
        f1score_metrics.update(outputs, label)

    epoch_loss = running_loss/ len(train_loader)
    print('[Epoch %d] loss: %.3f' % (epoch + 1, epoch_loss))

    precision = precision_metrics.compute()
    recall = recall_metrics.compute()
    accuracy = accuracy_metrics.compute()
    f1 = f1score_metrics.compute()
    print('Train precision:%.2f  Train recall: %.2f Train Accuracy: %.2f Train f1 score: %.2f' % (precision, recall, accuracy, f1), end="\n")

new_model.eval()
>>>>>>> 09eef5c5f39cbf0ac993056c6e2666ee23790dbd
torch.save(new_model.state_dict(), 'emotion_recognition_demo\\custom_vgg16.pth')
