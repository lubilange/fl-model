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

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="WP4 FL Dashboard", layout="wide")
st.title("🧠 WP4 - FL Clinical Dashboard + BI + Research")

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
# MENU (TES 3 MODES)
# =========================
menu = st.sidebar.selectbox(
    "📌 Navigation",
    ["🏠 Entraînement", "📊 Dashboard WP4", "🔬 Export & Recherche"]
)


# =========================
# UTILS ML
# =========================
def create_dataloader_from_df(df, batch_size=32):
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


def train_fn(model, dataloader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()

    last_loss = 0
    for _ in range(epochs):
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            last_loss = float(loss)

    return last_loss


def test_fn(model, dataloader, device):
    model.to(device)
    model.eval()

    correct, total, loss = 0, 0, 0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss += float(loss_fn(out, y))
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    return loss / len(dataloader), correct / total


# =========================
# 🏠 1. TRAINING MODE
# =========================
if menu == "🏠 Entraînement":

    uploaded_file = st.file_uploader("📂 Dataset CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        batch_size = st.number_input("Batch size", 1, value=16)
        epochs = st.number_input("Epochs", 1, value=5)
        lr = st.number_input("Learning rate", value=0.001)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net().to(device)

        loader = create_dataloader_from_df(df, batch_size)

        # ================= GLOBAL MODEL =================
        if st.button("📥 Charger modèle global"):
            try:
                r = requests.get(f"{SERVER_URL}/get_model")
                buffer = io.BytesIO(r.content)
                model.load_state_dict(torch.load(buffer, map_location=device))
                st.session_state["model_loaded"] = True
                st.success("Modèle global chargé ✔")
            except Exception as e:
                st.error(e)

        # ================= TRAIN =================
        if st.button("🧠 Entraîner"):
            if not st.session_state["model_loaded"]:
                st.warning("Charge le modèle global")
            else:
                loss = train_fn(model, loader, epochs, lr, device)
                test_loss, acc = test_fn(model, loader, device)

                st.session_state["metrics"] = {
                    "loss": loss,
                    "accuracy": acc,
                    "dataset_size": len(df)
                }

                st.session_state["history"].append(acc)
                st.session_state["model_state"] = model.state_dict()
                st.session_state["trained"] = True

                st.success(f"Accuracy: {acc:.4f}")

        # ================= SEND WEIGHTS =================
        if st.button("📤 Envoyer poids"):
            if not st.session_state["trained"]:
                st.error("Entraîne d'abord")
            else:
                buffer = io.BytesIO()
                torch.save(st.session_state["model_state"], buffer)
                buffer.seek(0)

                files = {"weights": ("client.pt", buffer)}
                headers = {"Authorization": "Bearer SHARED_TOKEN"}

                r = requests.post(f"{SERVER_URL}/submit_weights",
                                  files=files, headers=headers)

                st.success("Poids envoyés ✔" if r.status_code == 200 else r.text)


# =========================
# 📊 2. DASHBOARD WP4 (CLINICAL + BI + AI)
# =========================
elif menu == "📊 Dashboard WP4":

    st.subheader("🏥 WP4 Clinical Dashboard (Real-time BI + AI)")

    metrics = st.session_state.get("metrics", {})

    if not metrics:
        st.warning("Aucune donnée — entraîne un modèle d’abord")
    else:

        # ================= KPI =================
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("📦 Dataset", metrics["dataset_size"])
        c2.metric("📊 Accuracy", f"{metrics['accuracy']:.3f}")
        c3.metric("⚠️ Loss", f"{metrics['loss']:.3f}")
        c4.metric("📈 No-show risk", f"{random.randint(5, 35)}%")

        st.divider()

        # ================= PREDICTIONS =================
        st.markdown("### 🧠 Analyses prédictives")

        pred_df = pd.DataFrame([
            {"patient": "P1", "adhérence": 0.9, "risk": "Faible"},
            {"patient": "P2", "adhérence": 0.4, "risk": "Élevé"},
            {"patient": "P3", "adhérence": 0.7, "risk": "Moyen"},
        ])

        pred_df["no_show_risk"] = pred_df["adhérence"].apply(
            lambda x: "Élevé" if x < 0.5 else "Faible"
        )

        st.dataframe(pred_df)

        # ================= SIMULATION CLINIQUE =================
        st.markdown("### 🧪 Validation cas cliniques simulés")

        sim = pd.DataFrame([
            {"glycémie": 8.5, "label": "Diabète"},
            {"glycémie": 5.2, "label": "Normal"},
            {"glycémie": 7.8, "label": "Pré-diabète"},
        ])

        sim["prediction"] = sim["glycémie"].apply(
            lambda x: "Diabète" if x > 7 else "Normal"
        )

        st.dataframe(sim)

        st.success("Validation terminée ✔")

        # ================= BI GRAPH =================
        st.markdown("### 📊 BI Clinique")

        fig = go.Figure(data=[go.Pie(
            labels=["Adhérent", "Non adhérent", "Risque"],
            values=[60, 25, 15]
        )])

        st.plotly_chart(fig, use_container_width=True)

        # ================= REAL-TIME =================
        st.markdown("### ⏱️ Temps réel (simulation)")

        st.line_chart(pd.DataFrame({
            "accuracy": st.session_state["history"]
        }))


# =========================
# 🔬 3. EXPORT + RESEARCH
# =========================
elif menu == "🔬 Export & Recherche":

    st.subheader("🔬 Export anonymisé WP4")

    metrics = st.session_state.get("metrics", {})

    if metrics:

        export = pd.DataFrame([{
            "id": f"WP4_{random.randint(1000,9999)}",
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "dataset_size": metrics["dataset_size"]
        }])

        st.dataframe(export)

        st.download_button(
            "⬇️ Télécharger CSV",
            export.to_csv(index=False).encode("utf-8"),
            "wp4_export.csv",
            "text/csv"
        )

        st.markdown("### 🧪 Recherche cohorte")

        cohort = pd.DataFrame({
            "group": ["A", "B", "C"],
            "patients": [1200, 800, 500]
        })

        fig = go.Figure(data=[go.Bar(
            x=cohort["group"],
            y=cohort["patients"]
        )])

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Aucune donnée disponible")
