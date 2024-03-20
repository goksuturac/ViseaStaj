import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from model import FashionMNISTModelV0
from pathlib import Path

def train(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    # 1. creating file directory
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # 2. creating file save path
    MODEL_NAME = "03_computer_vision.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # 3. save the file state dictionary
    torch.save(model.state_dict(),MODEL_SAVE_PATH)


if __name__ == "__main__":
    batch_size = 32
    num_epochs = 10

    train_loader, _ = get_dataloaders(batch_size)
    model = FashionMNISTModelV0(input_shape=784, hidden_units=40, hidden_layers=10, output_shape=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, optimizer, criterion, num_epochs)