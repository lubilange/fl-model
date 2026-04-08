import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Définition du modèle MLP ---
class Net(nn.Module):
    def __init__(self, input_dim=6, output_dim=2):  # 6 features venant de WhatsApp
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # pas de sigmoid

# --- Entraînement local (optionnel pour tests) ---
def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch_x, batch_y in trainloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_loss = running_loss / (epochs * len(trainloader))
    return avg_loss

# --- Évaluation locale (optionnel pour tests) ---
def test(net, testloader, device):
    net.to(device)
    net.eval()
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_x, batch_y in testloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = net(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == batch_y).sum().item()
    accuracy = correct / len(testloader.dataset)
    avg_loss = total_loss / len(testloader)
    return avg_loss, accuracy
