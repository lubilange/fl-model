import torch
import torch.nn as nn
import torch.nn.functional as F

# Features utilisées pour le modèle
FEATURES = [
    'Age', 'RespiratoryRate', 'O2Saturation', 'PulseRate',
    'AdmissionGCS', 'Creatinine', 'HasClinicalExaminationBeenCompleted'
]

# Définition du modèle MLP pour données tabulaires
class Net(nn.Module):
    def __init__(self, input_dim=len(FEATURES), output_dim=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits, pas de sigmoid
