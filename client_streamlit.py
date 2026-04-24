import streamlit as st
import pandas as pd
import torch
import requests
import io
import os
import random
import plotly.graph_objects as go

from supabase import create_client, Client
from authexample.task import Net
from torch.utils.data import DataLoader, TensorDataset

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="WP4 FL Dashboard", layout="wide")

# =========================
# 🎨 STYLE CSS (AJOUT)
# =========================
st.markdown("""
<style>

/* BACKGROUND */
body {
    background-color: #eef1f5;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #1e3c5a;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* CARDS */
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* TITLES */
h1, h2, h3 {
    color: #1e3c5a;
}

</style>
""", unsafe_allow_html=True)

st.title(" Dashboard Médical + Federated Learning")

SERVER_URL = "https://fl-model.onrender.com"

# =========================
# SUPABASE
# =========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# SESSION STATE
# =========================
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False

if "trained" not in st.session_state:
    st.session_state["trained"] = False

if "metrics" not in st.session_state:
    st.session_state["metrics"] = {}

if "history" not in st.session_state:
    st.session_state["history"] = []

# =========================
# MENU (AMÉLIORÉ)
# =========================
menu = st.sidebar.radio(
    " Menu",
    [
        "Entraînement FL",
        "Dashboard Clinique",
        "Dashboard Recherche",
        "Export Anonymisé"
    ]
)

# =========================
# SUPABASE SAFE FETCH
# =========================
def safe_fetch(table):
    try:
        return supabase.table(table).select("*").execute().data or []
    except:
        return []

# =========================
# DATA SOURCES
# =========================
patients = pd.DataFrame(safe_fetch("patients"))
conditions = pd.DataFrame(safe_fetch("conditions"))
observations = pd.DataFrame(safe_fetch("observations"))
treatments = pd.DataFrame(safe_fetch("treatments"))
adherence_logs = pd.DataFrame(safe_fetch("adherence_logs"))
nurses = pd.DataFrame(safe_fetch("nurses"))

# =========================
# FL UTILS
# =========================
def create_dataloader_from_df(df, batch_size=32):
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_fn(model, dataloader, epochs, lr, device):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    last_loss = 0

    for _ in range(epochs):
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            last_loss = float(loss)

    return last_loss


def test_fn(model, dataloader, device):
    model.to(device)
    model.eval()

    correct, total, loss = 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss += float(criterion(out, y))
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    acc = correct / total
    return loss / len(dataloader), acc


# =========================================================
# 🧠 TRAINING
# =========================================================
if menu == "Entraînement FL":

    st.subheader(" Entraînement Federated Learning")

    uploaded_file = st.file_uploader(" Dataset CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)

        with col1:
            batch_size = st.number_input("Batch size", 1, value=16)

        with col2:
            epochs = st.number_input("Epochs", 1, value=5)

        with col3:
            lr = st.number_input("Learning rate", value=0.001, format="%.3f")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net().to(device)

        dataloader = create_dataloader_from_df(df, batch_size)

        if st.button(" Télécharger modèle global"):
            response = requests.get(f"{SERVER_URL}/get_model")
            buffer = io.BytesIO(response.content)
            model.load_state_dict(torch.load(buffer, map_location=device))
            st.session_state["model_loaded"] = True
            st.success("Modèle chargé")

        if st.button(" Entraîner"):
            if not st.session_state["model_loaded"]:
                st.warning("Télécharge le modèle d'abord")
            else:
                loss = train_fn(model, dataloader, epochs, lr, device)
                test_loss, acc = test_fn(model, dataloader, device)

                st.success(f"Loss: {loss:.4f}")
                st.info(f"Accuracy: {acc:.4f}")

                st.session_state["metrics"] = {
                    "loss": loss,
                    "accuracy": acc,
                    "dataset_size": len(df)
                }

                st.session_state["model_state"] = model.state_dict()
                st.session_state["trained"] = True

        if st.button(" Envoyer poids"):
            if not st.session_state.get("trained", False):
                st.error("Entraîne d'abord")
            else:
                buffer = io.BytesIO()
                torch.save(st.session_state["model_state"], buffer)
                buffer.seek(0)

                response = requests.post(
                    f"{SERVER_URL}/submit_weights",
                    files={"weights": buffer}
                )

                if response.status_code == 200:
                    st.success("Poids envoyés")
                else:
                    st.error("Erreur serveur")

# =========================================================
# 📊 DASHBOARD CLINIQUE
# =========================================================
elif menu == "Dashboard Clinique":

    st.subheader(" Vue Clinique")

    # ===== CARDS =====
    col1, col2, col3, col4 = st.columns(4)

    hommes = len(patients[patients["gender"] == "male"]) if not patients.empty else 0
    femmes = len(patients[patients["gender"] == "female"]) if not patients.empty else 0
    sympto = len(observations) if not observations.empty else 0

    with col1:
        st.markdown(f'<div class="card"><h3>Total</h3><h2>{len(patients)}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><h3>Hommes</h3><h2>{hommes}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><h3>Femmes</h3><h2>{femmes}</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="card"><h3>Symptômes</h3><h2>{sympto}</h2></div>', unsafe_allow_html=True)

    st.divider()

    # ===== GRAPH =====
    if not patients.empty:
        gender_dist = patients["gender"].value_counts()

        fig = go.Figure()
        fig.add_bar(x=gender_dist.index, y=gender_dist.values)

        fig.update_layout(title="Répartition Patients", template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📈 RESEARCH
# =========================================================
elif menu == "Dashboard Recherche":

    st.subheader(" Analyse")

    if not conditions.empty:
        risk_dist = conditions["severity"].value_counts()

        fig = go.Figure()
        fig.add_trace(go.Pie(labels=risk_dist.index, values=risk_dist.values))

        st.plotly_chart(fig)

# =========================================================
# 📤 EXPORT
# =========================================================
elif menu == "Export Anonymisé":

    metrics = st.session_state.get("metrics", {})

    if metrics:
        export = pd.DataFrame([metrics])
        st.dataframe(export)

        st.download_button(
            " Télécharger CSV",
            export.to_csv(index=False),
            "export.csv"
        )
