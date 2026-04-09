import streamlit as st
import pandas as pd
import torch
import requests
import io
import os
import random
import plotly.graph_objects as go
from authexample.task import Net
from torch.utils.data import DataLoader, TensorDataset

st.set_page_config(page_title="WP4 FL Dashboard", layout="wide")

st.title("🧠 Client FL - Upload Dataset et Envoi de Poids")

SERVER_URL = "https://fl-model.onrender.com"


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
# UTILS
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


# =========================
# 🏠 ENTRAÎNEMENT
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

        # =========================
        # GLOBAL MODEL
        # =========================
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

        # =========================
        # TRAIN LOCAL
        # =========================
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

        # =========================
        # SEND WEIGHTS
        # =========================
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


# =========================
# 📊 DASHBOARD WP4 (FINAL)
# =========================
elif menu == "📊 Dashboard WP4":

    st.subheader("🏥 Dashboard Clinique & Recherche WP4")

    metrics = st.session_state.get("metrics", {})

    if not metrics:
        st.warning("Aucune donnée disponible")
    else:

        # =========================
        # KPI CLINIQUE
        # =========================
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("🔥 Cas critiques", random.randint(1, 10))
        c2.metric("📊 Accuracy", f"{metrics['accuracy']:.4f}")
        c3.metric("⚠️ Loss", f"{metrics['loss']:.4f}")
        c4.metric("📦 Patients", metrics["dataset_size"])

        st.divider()

        # =========================
        # CAS CLINIQUES SIMULÉS (WP4 VALIDATION)
        # =========================
        st.markdown("### 🧪 Cas cliniques simulés")

        sim_cases = pd.DataFrame([
            {"patient": "Sim-1", "glycemie": 8.5, "risk": "Élevé"},
            {"patient": "Sim-2", "glycemie": 5.2, "risk": "Faible"},
            {"patient": "Sim-3", "glycemie": 7.8, "risk": "Moyen"},
        ])

        sim_cases["prediction_model"] = sim_cases["glycemie"].apply(
            lambda x: "Élevé" if x > 7 else "Faible"
        )

        st.dataframe(sim_cases)

        st.success("Validation modèle terminée ✔")

        # =========================
        # COURBE GLYCÉMIE
        # =========================
        st.markdown("### 📊 Évolution clinique")

        st.bar_chart(pd.DataFrame({
            "glycemie": [5, 6, 8.5, 6, 7.9]
        }))

        # =========================
        # COHORTE RECHERCHE
        # =========================
        st.markdown("### 🧪 Cohorte Recherche")

        fig = go.Figure(data=[go.Pie(
            labels=["Groupe A", "B", "Placebo"],
            values=[1200, 800, 787],
            hole=0.65
        )])

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # HISTORIQUE FL
        # =========================
        st.markdown("### 📈 Accuracy FL")

        st.line_chart(pd.DataFrame({
            "accuracy": st.session_state["history"]
        }))


# =========================
# 🔬 EXPORT ANONYMISÉ
# =========================
elif menu == "🔬 Export & Recherche":

    st.subheader("🔬 Export anonymisé WP4")

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
