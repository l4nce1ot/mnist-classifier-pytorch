import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

def get_data_loaders(batch_size: int = 64, num_workers: int = 4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

def train_model(model, loader, device, criterion, optimizer, epochs: int):
    losses, accs = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            predicted = outputs.argmax(dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        losses.append(running_loss / total)
        accs.append(correct / total)
    return losses, accs

def evaluate_model(model, loader, device, criterion):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item() * data.size(0)
            predicted = outputs.argmax(dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return test_loss / total, correct / total

def plt_distribution(dataset):
    label_list = [label for _, label in dataset]
    plt.hist(label_list, bins=range(11), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
    plt.xlabel('Цифра')
    plt.ylabel('Количество')
    plt.title('Распределение классов в обучающей выборке MNIST')
    plt.xticks(range(10))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plt_metrics(loss1, acc1, loss2, acc2, epochs):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, loss1, '--', label='Loss Simple')
    plt.plot(epochs_range, loss2, '-.', label='Loss Improved')
    plt.plot(epochs_range, acc1, '--', label='Accuracy Simple')
    plt.plot(epochs_range, acc2, '-.', label='Accuracy Improved')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Сравнение метрик обучения')
    plt.legend()
    plt.show()

def plt_results(test_loss, test_acc):
    labels = ['Simple', 'Improved']
    losses = [test_loss[0], test_loss[1]]
    accs = [test_acc[0], test_acc[1]]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].bar(labels, losses)
    axes[0].set_title('Loss')
    axes[1].bar(labels, accs)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Accuracy')
    plt.tight_layout()
    plt.show()

def predict_samples(model, loader, device, n_samples: int = 6, title: str = "Предсказания"):
    model.eval()
    data_iter = iter(loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    predictions = outputs.argmax(dim=1)

    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    for i, ax in enumerate(axes.flatten()[:n_samples]):
        ax.imshow(images[i][0].cpu(), cmap='gray')
        ax.set_title(f"Истинно: {labels[i].item()}\nПредсказано: {predictions[i].item()}")
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    batch_size = 64
    num_workers = 4
    lr = 1e-3
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_data_loaders(batch_size, num_workers)
    plt_distribution(train_loader.dataset)

    simple = SimpleCNN().to(device)
    improved = ImprovedCNN().to(device)
    optimizer_simple = optim.Adam(simple.parameters(), lr=lr)
    optimizer_improved = optim.Adam(improved.parameters(), lr=lr)

    print("Обучение SimpleCNN")
    start_simple = time.time()
    loss_s, acc_s = train_model(simple, train_loader, device, criterion, optimizer_simple, epochs)
    end_simple = time.time()
    time_simple = end_simple - start_simple

    print("Обучение ImprovedCNN")
    start_improved = time.time()
    loss_i, acc_i = train_model(improved, train_loader, device, criterion, optimizer_improved, epochs)
    end_improved = time.time()
    time_improved = end_improved - start_improved

    print(f"\nВремя обучения SimpleCNN: {time_simple:.2f} секунд")
    print(f"Время обучения ImprovedCNN: {time_improved:.2f} секунд")
    print(f"Общее время обучения: {time_simple + time_improved:.2f} секунд")

    predict_samples(simple, test_loader, device, title="SimpleCNN: предсказания")
    predict_samples(improved, test_loader, device, title="ImprovedCNN: предсказания")
    test_loss_s, test_acc_s = evaluate_model(simple, test_loader, device, criterion)
    test_loss_i, test_acc_i = evaluate_model(improved, test_loader, device, criterion)
    plt_metrics(loss_s, acc_s, loss_i, acc_i, epochs)
    plt_results((test_loss_s, test_loss_i), (test_acc_s, test_acc_i))

if __name__ == '__main__':
    main()
