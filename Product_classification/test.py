import torch
import torch.nn as nn
from data_loader import get_dataloaders
from model import FashionMNISTModelV0

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")

if __name__ == "__main__":
    batch_size = 64

    _, test_loader = get_dataloaders(batch_size)
    model = FashionMNISTModelV0(input_shape=784, hidden_units=20, hidden_layers=2, output_shape=10)
    torch.save(model.state_dict(), "model.pth")
    model.load_state_dict(torch.load("model.pth"))
    test(model, test_loader)
