import torch
import requests
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from authexample.task import Net, train as train_fn, test as test_fn, update_last_train, should_retrain

# URL du microservice WhatsApp qui renvoie les données du client
WHATSAPP_SERVICE_URL = "https://federatedlearning.onrender.com/training-data"

# Crée ClientApp
app = ClientApp()

def fetch_client_data():
    """Récupère les données du client depuis le microservice WhatsApp."""
    resp = requests.get(WHATSAPP_SERVICE_URL)
    resp.raise_for_status()
    json_data = resp.json()
    dataset = json_data["data"]

    X = []
    y = []

    for row in dataset:
        # Exemple simple: convertir dict de features en vecteur
        features = [
            row.get("Age", 0),
            row.get("RespiratoryRate") or 0,
            row.get("O2Saturation") or 0,
            row.get("PulseRate") or 0,
            row.get("AdmissionGCS") or 0,
            row.get("Creatinine") or 0,
        ]
        label = row.get("RequiredICUAdmission", 0)  # par ex
        X.append(features)
        y.append(label)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

@app.train()
def train(msg: Message, context: Context):
    """Train le modèle sur les données du client WhatsApp."""
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X, y = fetch_client_data()
    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=context.run_config["batch-size"]
    )

    if should_retrain():
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device
        )
        update_last_train()
    else:
        train_loss = 0.0

    model_record = ArrayRecord(model.state_dict())
    metrics = MetricRecord({"train_loss": train_loss, "num-examples": len(trainloader.dataset)})
    content = RecordDict({"arrays": model_record, "metrics": metrics})
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Évalue le modèle sur les données locales WhatsApp."""
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X, y = fetch_client_data()
    valloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=context.run_config["batch-size"]
    )

    eval_loss, eval_acc = test_fn(model, valloader, device)
    metrics = MetricRecord({"eval_loss": eval_loss, "eval_acc": eval_acc, "num-examples": len(valloader.dataset)})
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)