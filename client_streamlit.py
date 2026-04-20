import streamlit as st
import pandas as pd
import torch
import flwr as fl

from torch.utils.data import DataLoader, TensorDataset
from authexample.task import Net, train, test

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="FL Client", layout="wide")
st.title("Federated Learning Client (Flower)")

SERVER_ADDRESS = "fl-model.onrender.com:8080"

# =========================
# DATA LOADER
# =========================
def create_dataloader_from_df(df, batch_size=32):
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

# =========================
# FLOWER CLIENT
# =========================
class FLClient(fl.client.NumPyClient):

    def __init__(self, model, trainloader, device, epochs, lr):
        self.model = model
        self.trainloader = trainloader
        self.device = device
        self.epochs = epochs
        self.lr = lr

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        loss = train(
            self.model,
            self.trainloader,
            self.epochs,
            self.lr,
            self.device
        )

        return self.get_parameters(config), len(self.trainloader.dataset), {"loss": float(loss)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss, acc = test(self.model, self.trainloader, self.device)

        return float(loss), len(self.trainloader.dataset), {"accuracy": float(acc)}

# =========================
# UI
# =========================
uploaded_file = st.file_uploader("📂 Upload CSV dataset", type="csv")

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    with col1:
        batch_size = st.number_input("Batch size", 1, value=16)

    with col2:
        epochs = st.number_input("Local epochs", 1, value=3)

    with col3:
        lr = st.number_input("Learning rate", value=0.001, format="%.4f")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # START FLOWER CLIENT
    # =========================
    if st.button("🚀 Start Federated Learning"):

        try:
            st.info("Connecting to Flower server...")

            trainloader = create_dataloader_from_df(df, batch_size)
            model = Net().to(device)

            client = FLClient(model, trainloader, device, epochs, lr)

            fl.client.start_numpy_client(
                server_address=SERVER_ADDRESS,
                client=client,
            )

            st.success("Training finished ✔")

        except Exception as e:
            st.error(f"Error: {e}")
