from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from task import Net, FEATURES

app = FastAPI()

# Charger le modèle global
model = Net()
model.load_state_dict(torch.load("final_model.pt", map_location=torch.device('cpu')))
model.eval()

# Définir le format attendu pour les données patient
class PatientData(BaseModel):
    Age: float
    RespiratoryRate: float
    O2Saturation: float
    PulseRate: float
    AdmissionGCS: float
    Creatinine: float
    HasClinicalExaminationBeenCompleted: int

# Endpoint de prédiction
@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.dict()])

    # Normalisation des features (idéalement, réutiliser le scaler du training)
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    # Convertir en tensor
    x = torch.tensor(df[FEATURES].values.astype(float), dtype=torch.float32)

    # Prédiction
    with torch.no_grad():
        outputs = model(x)
        predicted_class = outputs.argmax(1).item()

    return {"prediction": int(predicted_class)}