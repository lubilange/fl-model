import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# --- Charger le dataset centralisé ou local pour client ---
DATA_PATH = os.environ.get("DATA_PATH", "df_final.csv")
df_final = pd.read_csv(DATA_PATH)

FEATURES = [
    'Age', 'RespiratoryRate', 'O2Saturation', 'PulseRate', 
    'AdmissionGCS', 'Creatinine', 'HasClinicalExaminationBeenCompleted'
]
TARGET = "RequiredICUAdmission"

# --- Normalisation des features ---
scaler = StandardScaler()
df_final[FEATURES] = scaler.fit_transform(df_final[FEATURES])

# --- Définition du modèle MLP ---
class Net(nn.Module):
    def __init__(self, input_dim=len(FEATURES), output_dim=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # pas de sigmoid

# --- Charger le dataset pour un client donné ---
def load_data(partition_id: int, num_partitions: int, batch_size: int = 16):
    df_client = np.array_split(df_final.sample(frac=1, random_state=42), num_partitions)[partition_id]
    if len(df_client) == 0:
        raise ValueError(f"Client {partition_id} n'a pas de données !")

    X = df_client[FEATURES].values.astype(np.float32)
    y = df_client[TARGET].values.astype(np.int64)

    split_idx = int(0.8 * len(df_client))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    trainloader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                             batch_size=batch_size, shuffle=True)
    testloader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
                            batch_size=batch_size)
    return trainloader, testloader

# --- Charger tout le dataset centralisé ---
def load_centralized_dataset(batch_size: int = 16):
    X = df_final[FEATURES].values.astype(np.float32)
    y = df_final[TARGET].values.astype(np.int64)
    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                        batch_size=batch_size)
    return loader

# --- Entraînement ---
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

# --- Évaluation ---
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