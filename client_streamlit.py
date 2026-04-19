import streamlit as st
import pandas as pd
import torch
import requests
import io
import os
import random
import plotly.graph_objects as go
from authexample.task import Net, train, test
from supabase import create_client, Client
from torch.utils.data import DataLoader, TensorDataset

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="WP4 FL Dashboard", layout="wide")
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
# SUPABASE SAFE FETCH
# =========================
def safe_fetch(table):
    try:
        return supabase.table(table).select("*").execute().data or []
    except:
        return []
# =========================
# DATA SOURCES (COHERENT WITH YOUR BACKEND)
# =========================
patients = pd.DataFrame(safe_fetch("patients"))
conditions = pd.DataFrame(safe_fetch("conditions"))
observations = pd.DataFrame(safe_fetch("observations"))
treatments = pd.DataFrame(safe_fetch("treatments"))
adherence_logs = pd.DataFrame(safe_fetch("adherence_logs"))
nurses = pd.DataFrame(safe_fetch("nurses"))
# =========================
# DATA LOADER
# =========================
def create_dataloader_from_df(df, batch_size=32):
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

# =========================
# UI
# =========================
menu = st.sidebar.selectbox(
    "Navigation",
    ["Entraînement FL", "Dashboard Clinique", "Dashboard Recherche"]
)

# =========================================================
# 🏠 TRAINING FL
# =========================================================
if menu == "Entraînement FL":

    uploaded_file = st.file_uploader("📂 Dataset CSV", type="csv")

    if uploaded_file:
        df = __import__("pandas").read_csv(uploaded_file)
        st.dataframe(df.head())
        col1, col2, col3 = st.columns(3)

        with col1:
            batch_size = st.number_input("Batch size", 1, value=16)

        with col2:
            epochs = st.number_input("Epochs", 1, value=5)

        with col3:
            lr = st.number_input("Learning rate", value=0.001, format="%.3f")


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = create_dataloader_from_df(df, batch_size)

        # =========================
        # GET GLOBAL MODEL
        # =========================
        if st.button("📥 Télécharger modèle global"):
            try:
                r = requests.get(f"{SERVER_URL}/get_model", timeout=20)
                r.raise_for_status()

                state = torch.load(io.BytesIO(r.content), map_location=device)

                model = Net().to(device)
                model.load_state_dict(state)

                st.session_state.model = model
                st.session_state.model_loaded = True
                st.session_state.trained = False

                st.success("Modèle global chargé ✔")

            except Exception as e:
                st.error(f"Erreur serveur: {e}")

        # =========================
        # TRAIN LOCAL
        # =========================
        if st.button("🧠 Entraîner"):

            if not st.session_state.model_loaded:
                st.warning("Charge d'abord le modèle global")
            else:
                model = st.session_state.model

                loss = train(model, dataloader, epochs, lr, device)
                test_loss, acc = test(model, dataloader, device)

                st.session_state.metrics = {
                    "loss": loss,
                    "test_loss": test_loss,
                    "accuracy": acc,
                    "dataset_size": len(df)
                }

                st.session_state.trained = True

                st.success(f"Loss: {loss:.4f}")
                st.info(f"Accuracy: {acc:.4f}")

        # =========================
        # SEND WEIGHTS
        # =========================
        if st.button("📤 Envoyer poids (FL)"):

            if not st.session_state.trained:
                st.error("Entraîne d'abord le modèle")
            else:
                buffer = io.BytesIO()
                torch.save(st.session_state.model.state_dict(), buffer)
                buffer.seek(0)

                files = {"weights": ("client.pt", buffer)}

                TOKEN = os.environ.get("FL_CLIENT_TOKEN", "SHARED_TOKEN")
                headers = {"Authorization": f"Bearer {TOKEN}"}

                try:
                    r = requests.post(
                        f"{SERVER_URL}/submit_weights",
                        files=files,
                        headers=headers,
                        timeout=30
                    )

                    if r.status_code == 200:
                        st.success("✔ Poids envoyés au serveur FL")
                        st.json(r.json())

                        st.info("Aggregation gérée automatiquement par Flower côté serveur")

                        st.session_state.trained = False

                    else:
                        st.error(r.text)

                except Exception as e:
                    st.error(f"Erreur réseau: {e}")
# =========================================================
# 📊 CLINICAL DASHBOARD (REAL + AI + ANALYTICS)
# =========================================================

elif menu == "Dashboard Clinique":

    st.subheader("🏥Vue clinique en temps réel")

    # ================= KPI =================
    col1, col2, col3 = st.columns(3)

    col1.metric("👥 Patients", len(patients))
    col2.metric("🧾 Conditions FHIR", len(conditions))
    col3.metric("🩺 Symptômes", len(observations))

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
# 📈 RESEARCH DASHBOARD (BI)
# =========================================================
elif menu == "Dashboard Recherche":

    st.subheader("Graphique pour Recherche Analytique")

    # =========================
    # 1. COHORTE PATIENTS
    # =========================
    st.markdown("### Répartition patients")

    if not patients.empty:
        gender_dist = patients["gender"].value_counts()

        st.bar_chart(gender_dist)

        st.write("Distribution patients par genre")
    else:
        st.info("Aucun patient")

    # =========================
    # 2. RISQUE CLINIQUE (conditions backend)
    # =========================
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
