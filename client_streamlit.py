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
# 🎨 STYLE CSS (sidebar + cartes)
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Fond général clair */
    .main {
        background-color: #eef1f5;
    }

    /* Sidebar façon image */
    section[data-testid="stSidebar"] {
        background-color: #1e3c5a;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Cartes blanches avec ombre */
    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }

    /* Titres */
    h1, h2, h3 {
        color: #1e3c5a;
    }
</style>
""", unsafe_allow_html=True)

st.title("Dashboard + Federated Learning ")

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
# MENU (style sidebar demandé)
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
# 🧠 TRAINING (inchangé)
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
            try:
                response = requests.get(f"{SERVER_URL}/get_model")
                response.raise_for_status()
                buffer = io.BytesIO(response.content)
                global_state = torch.load(buffer, map_location=device)
                model.load_state_dict(global_state)
                st.session_state["model_loaded"] = True
                st.success("Modèle global chargé ✔")
            except Exception as e:
                st.error(e)

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
                st.session_state["history"].append(acc)
                st.session_state["model_state"] = model.state_dict()
                st.session_state["trained"] = True

        if st.button("📤 Envoyer poids"):
            if not st.session_state.get("trained", False):
                st.error("Entraîne d'abord le modèle")
            else:
                buffer = io.BytesIO()
                torch.save(st.session_state["model_state"], buffer)
                buffer.seek(0)
                files = {"weights": ("client_weights.pt", buffer)}
                TOKEN = os.environ.get("FL_CLIENT_TOKEN", "SHARED_TOKEN")
                headers = {"Authorization": f"Bearer {TOKEN}"}
                response = requests.post(
                    f"{SERVER_URL}/submit_weights",
                    files=files,
                    headers=headers
                )
                if response.status_code == 200:
                    st.success("Poids envoyés ✔")
                else:
                    st.error(response.text)

# =========================================================
# 📊 DASHBOARD CLINIQUE (style conservé, KPIs en cartes)
# =========================================================
elif menu == "Dashboard Clinique":
    st.subheader("🏥 Vue clinique en temps réel")

    # KPIs sous forme de cartes (style, pas de contenu factice)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="card"><h3>👥 Patients</h3><h2>{len(patients)}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><h3>🧾 Conditions FHIR</h3><h2>{len(conditions)}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><h3>🩺 Symptômes</h3><h2>{len(observations)}</h2></div>', unsafe_allow_html=True)

    st.divider()

    # ================= TRIAGE =================
    st.markdown("### 🚨 Niveau d'alerte")
    if not conditions.empty:
        st.bar_chart(conditions["severity"].value_counts())

    st.divider()

    # ================= TENDANCE =================
    st.markdown("### 📈 Répartition des symptômes")
    if not observations.empty and "severity" in observations.columns:
        trend = observations["severity"].value_counts().reset_index()
        trend.columns = ["severity", "count"]
        st.bar_chart(trend.set_index("severity"))

    st.divider()

    # ================= SUPPORT =================
    st.markdown("### 👩‍⚕️ Support infirmier")
    if not nurses.empty:
        st.bar_chart(nurses["status"].value_counts())

    st.divider()

    # ================= SIMULATION =================
    st.markdown("###  cas de simulations")
    sim = pd.DataFrame([
        {"glycémie": 4.5, "niveau": "normal"},
        {"glycémie": 8.2, "niveau": "élevé"},
        {"glycémie": 6.8, "niveau": "modéré"}
    ])
    sim["prediction"] = sim["glycémie"].apply(lambda x: "élevé" if x > 7 else "normal")
    st.dataframe(sim)

# =========================================================
# 📈 DASHBOARD RECHERCHE
# =========================================================
elif menu == "Dashboard Recherche":
    st.subheader("Graphique pour Recherche Analytique")
    st.markdown("### Répartition patients")
    if not patients.empty:
        gender_dist = patients["gender"].value_counts()
        st.bar_chart(gender_dist)
        st.write("Distribution patients par genre")
    else:
        st.info("Aucun patient")

    st.markdown("###  Répartition des risques")
    if not conditions.empty:
        risk_dist = conditions["severity"].value_counts()
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=risk_dist.index,
            values=risk_dist.values
        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune condition")

# =========================================================
# 🔬 EXPORT ANONYMIZED
# =========================================================
elif menu == "Export Anonymisé":
    st.subheader("Export des performances FL")
    metrics = st.session_state.get("metrics", {})
    if metrics:
        export = pd.DataFrame([{
            "id": "RECH_" + str(random.randint(1000, 9999)),
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "dataset_size": metrics["dataset_size"]
        }])
        st.dataframe(export)
        csv = export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Télécharger export",
            csv,
            "wp4_export.csv",
            "text/csv"
        )
    else:
        st.info("Aucune donnée disponible")
