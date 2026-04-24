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
# 🎨 STYLE CSS (DASHBOARD PRO)
# =========================
st.markdown("""
<style>

/* BACKGROUND */
body {
    background-color: #f5f7fb;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3c72, #2a5298);
    color: white;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* TITLES */
h1, h2, h3 {
    font-weight: 700;
}

/* KPI CARDS */
.kpi-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    text-align: center;
}

.kpi-title {
    font-size: 14px;
    color: gray;
}

.kpi-value {
    font-size: 28px;
    font-weight: bold;
    color: #1e3c72;
}

/* BLOCK CARD */
.block-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* BUTTON */
.stButton button {
    background-color: #2a5298;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton button:hover {
    background-color: #1e3c72;
    color: white;
}

</style>
""", unsafe_allow_html=True)

st.title("Dashboard + Federated Learning")

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
# MENU
# =========================
menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Entraînement FL",
        "Dashboard Clinique",
        "Dashboard Recherche",
        "Export Anonymisé"
    ]
)

# =========================
# FETCH
# =========================
def safe_fetch(table):
    try:
        return supabase.table(table).select("*").execute().data or []
    except:
        return []

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
# 🏠 TRAINING
# =========================================================
if menu == "Entraînement FL":

    uploaded_file = st.file_uploader("📂 Dataset CSV", type="csv")

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

        if st.button("📥 Télécharger modèle global"):
            response = requests.get(f"{SERVER_URL}/get_model")
            buffer = io.BytesIO(response.content)
            model.load_state_dict(torch.load(buffer, map_location=device))
            st.session_state["model_loaded"] = True
            st.success("Modèle chargé ✔")

        if st.button("🧠 Entraîner"):

            if not st.session_state["model_loaded"]:
                st.warning("Télécharge le modèle global")
            else:
                loss = train_fn(model, dataloader, epochs, lr, device)
                test_loss, acc = test_fn(model, dataloader, device)

                st.success(f"Loss: {loss:.4f}")
                st.info(f"Accuracy: {acc:.4f}")

                st.session_state["metrics"] = {
                    "loss": loss,
                    "test_loss": test_loss,
                    "accuracy": acc,
                    "dataset_size": len(df)
                }

                st.session_state["model_state"] = model.state_dict()
                st.session_state["trained"] = True

# =========================================================
# 📊 DASHBOARD CLINIQUE
# =========================================================
elif menu == "Dashboard Clinique":

    st.subheader("🏥 Vue clinique")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Patients</div><div class="kpi-value">{len(patients)}</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Conditions</div><div class="kpi-value">{len(conditions)}</div></div>', unsafe_allow_html=True)

    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Symptômes</div><div class="kpi-value">{len(observations)}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown("### 🚨 Niveau d'alerte")
    if not conditions.empty:
        st.bar_chart(conditions["severity"].value_counts())
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Symptômes")
    if not observations.empty and "severity" in observations.columns:
        st.bar_chart(observations["severity"].value_counts())
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 📈 RESEARCH
# =========================================================
elif menu == "Dashboard Recherche":

    st.subheader("Recherche")

    if not patients.empty:
        st.bar_chart(patients["gender"].value_counts())

    if not conditions.empty:
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=conditions["severity"].value_counts().index,
            values=conditions["severity"].value_counts().values
        ))
        st.plotly_chart(fig)

# =========================================================
# EXPORT
# =========================================================
elif menu == "Export Anonymisé":

    metrics = st.session_state.get("metrics", {})

    if metrics:
        export = pd.DataFrame([metrics])
        st.dataframe(export)

        csv = export.to_csv(index=False).encode("utf-8")

        st.download_button("Télécharger", csv, "export.csv")
