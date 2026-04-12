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
st.title("🧠 Client FL + Dashboard Clinique (Supabase)")

SERVER_URL = "https://fl-model.onrender.com"

# =========================
# 🔒 SUPABASE CONFIG SÉCURISÉ
# =========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Supabase non configuré. Ajoute les variables d'environnement.")
    st.stop()

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
    "📌 Navigation",
    ["🏠 Entraînement", "📊 Dashboard WP4", "🔬 Export & Recherche"]
)

# =========================
# SUPABASE FUNCTIONS
# =========================
def safe_fetch(table):
    try:
        data = supabase.table(table).select("*").execute().data
        return data or []
    except Exception as e:
        st.error(f"Erreur chargement {table}: {e}")
        return []

def get_patients():
    return safe_fetch("patients")

def get_conditions():
    return safe_fetch("conditions")

def get_observations():
    return safe_fetch("observations")

def get_medications():
    return safe_fetch("medications")

def get_reminders():
    return safe_fetch("reminders")

# =========================
# ML UTILS
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

    acc = correct / total if total > 0 else 0
    return loss / len(dataloader), acc

# =========================
# 🏠 TRAINING
# =========================
if menu == "🏠 Entraînement":

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
                    "accuracy": acc,
                    "dataset_size": len(df)
                }

                st.session_state["history"].append(acc)
                st.session_state["model_state"] = model.state_dict()
                st.session_state["trained"] = True

        if st.button("📤 Envoyer poids"):
            if not st.session_state.get("trained"):
                st.error("Entraîne d'abord le modèle")
            else:
                buffer = io.BytesIO()
                torch.save(st.session_state["model_state"], buffer)
                buffer.seek(0)

                files = {"weights": ("client_weights.pt", buffer)}
                headers = {"Authorization": "Bearer SHARED_TOKEN"}

                response = requests.post(
                    f"{SERVER_URL}/submit_weights",
                    files=files,
                    headers=headers
                )

                if response.status_code == 200:
                    st.success("Poids envoyés ✔")
                else:
                    st.error(response.text)

# =========================
# 📊 DASHBOARD
# =========================
elif menu == "📊 Dashboard WP4":

    st.subheader("🏥 Clinical Dashboard (READ-ONLY)")

    patients = pd.DataFrame(get_patients())
    conditions = pd.DataFrame(get_conditions())
    observations = pd.DataFrame(get_observations())
    treatments = pd.DataFrame(safe_fetch("treatments"))
    adherence_logs = pd.DataFrame(safe_fetch("adherence_logs"))
    nurses = pd.DataFrame(safe_fetch("nurses"))

    # =========================
    # KPI GLOBALS
    # =========================
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("👥 Patients", len(patients))
    c2.metric("⚠️ Conditions", len(conditions))
    c3.metric("🧪 Observations", len(observations))
    c4.metric("💊 Treatments", len(treatments))

    st.divider()

    # =========================
    # PATIENTS
    # =========================
    st.markdown("### 👥 Patients")

    if not patients.empty:
        st.dataframe(patients[["phone", "name", "gender", "birth_date", "onboarded"]])
    else:
        st.info("Aucun patient")

    # =========================
    # CONDITIONS (TRIAGE BACKEND ONLY)
    # =========================
    st.markdown("### 🚨 Triage clinique (backend engine)")

    if not conditions.empty:
        st.dataframe(
            conditions[["patient_id", "label", "severity", "status", "created_at"]]
        )

        st.bar_chart(conditions["severity"].value_counts())
    else:
        st.info("Aucune condition")

    # =========================
    # SYMPTÔMES (FHIR OBSERVATIONS)
    # =========================
    st.markdown("### 🧪 Observations (FHIR synced)")

    if not observations.empty:
        st.dataframe(
            observations[["patient_id", "text", "severity", "created_at"]]
        )
    else:
        st.info("Aucune observation")

    # =========================
    # TREATMENTS
    # =========================
    st.markdown("### 💊 Traitements")

    if not treatments.empty:
        st.dataframe(
            treatments[["patient_id", "text", "status", "created_at"]]
        )
    else:
        st.info("Aucun traitement")

    # =========================
    # ADHERENCE (BACKEND LOGS ONLY)
    # =========================
    st.markdown("### 💊 Adhérence (logs backend)")

    if not adherence_logs.empty:

        taken = len(adherence_logs[adherence_logs["status"] == "taken"])
        total = len(adherence_logs)

        adherence_rate = taken / total if total > 0 else 0

        st.metric("Adhérence globale", f"{adherence_rate:.2f}")

        st.bar_chart(adherence_logs["status"].value_counts())

    else:
        st.info("Aucun log d’adhérence")

    # =========================
    # NURSES OPS
    # =========================
    st.markdown("### 👩‍⚕️ Équipe infirmière")

    if not nurses.empty:
        st.bar_chart(nurses["status"].value_counts())
    else:
        st.info("Aucune donnée infirmière")

    # =========================
    # FL HISTORY (OK)
    # =========================
    st.markdown("### 🧠 Federated Learning")

    if st.session_state["history"]:
        st.line_chart(pd.DataFrame({
            "accuracy": st.session_state["history"]
        }))
# =========================
# 🔬 EXPORT
# =========================
elif menu == "🔬 Export & Recherche":

    st.subheader("🔬 Export anonymisé")

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
            "⬇️ Télécharger",
            csv,
            "export.csv",
            "text/csv"
        )
    else:
        st.info("Aucune donnée disponible")
