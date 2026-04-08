"""authexample: A Flower client app for federated learning with WA data."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from authexample.task import Net, test as test_fn, train as train_fn
import requests

# =============================
# CONFIGURATION
# =============================
FL_SERVER_URL = "https://fl-model.onrender.com"
WA_TRAINING_URL = "https://federatedlearning.onrender.com/training-data"

CLIENT_ID = "hospital_1"
CLIENT_GROUP = "group_1"

# =============================
# HELPERS
# =============================
def fetch_training_data():
    """Récupère les features depuis le microservice WA et les transforme en tensors."""
    resp = requests.get(WA_TRAINING_URL)
    resp.raise_for_status()
    data = resp.json()

    dataset = data["data"]  # liste de dictionnaires
    if not dataset:
        raise ValueError("Pas de données disponibles pour l'entraînement")

    X_list = []
    y_list = []

    for d in dataset:
        x = [
            d.get("Age", 0),
            d.get("RespiratoryRate", 0) or 0,
            d.get("O2Saturation", 0) or 0,
            d.get("PulseRate", 0) or 0,
            d.get("AdmissionGCS", 0) or 0,
            d.get("Creatinine", 0) or 0,
            d.get("HasClinicalExaminationBeenCompleted", 0),
        ]
        y = d.get("RequiredICUAdmission", 0)
        X_list.append(x)
        y_list.append(y)

    X_tensor = torch.tensor(X_list, dtype=torch.float32)
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    return X_tensor, y_tensor

# =============================
# CLIENT APP
# =============================
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X, y = fetch_training_data()
    dataset = torch.utils.data.TensorDataset(X, y)
    batch_size = context.run_config["batch-size"]
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loss = train_fn(model, trainloader, context.run_config["local-epochs"], msg.content["config"]["lr"], device)

    model_record = ArrayRecord(model.state_dict())
    metrics = MetricRecord({"train_loss": train_loss, "num-examples": len(dataset)})
    content = RecordDict({"arrays": model_record, "metrics": metrics})
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X, y = fetch_training_data()
    dataset = torch.utils.data.TensorDataset(X, y)
    batch_size = context.run_config["batch-size"]
    valloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = MetricRecord({"eval_loss": eval_loss, "eval_acc": eval_acc, "num-examples": len(dataset)})
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)

# =============================
# LANCEMENT DU CLIENT
# =============================
if __name__ == "__main__":
    print(f"Démarrage du client FL {CLIENT_ID}...")
    app.run(FL_SERVER_URL)
