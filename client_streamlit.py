import streamlit as st
import pandas as pd
import torch
import requests
import io
import os
from authexample.task import Net
from torch.utils.data import DataLoader, TensorDataset

st.title("Client FL - Upload Dataset et Envoi de Poids")

SERVER_URL = "https://fl-model.onrender.com"

# --- état global ---
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False

if "trained" not in st.session_state:
    st.session_state["trained"] = False

if "metrics" not in st.session_state:
    st.session_state["metrics"] = {}


# =========================
# 📊 MENU DASHBOARD
# =========================
menu = st.sidebar.selectbox(
    "📌 Navigation",
    ["🏠 Entraînement", "📊 Dashboard", "🔬 Export & Recherche"]
)

# --- utils ---
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

    for _ in range(epochs):
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

    return float(loss)


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
# 🏠 PAGE ENTRAÎNEMENT
# =========================
if menu == "🏠 Entraînement":

    uploaded_file = st.file_uploader("Choisissez votre dataset CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Aperçu du dataset :")
        st.dataframe(df.head())

        batch_size = st.number_input("Batch size", 1, value=16)
        epochs = st.number_input("Local epochs", 1, value=5)
        lr = st.number_input("Learning rate", value=0.001, format="%.3f")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net().to(device)

        dataloader = create_dataloader_from_df(df, batch_size)

        # =========================
        # DOWNLOAD MODEL GLOBAL
        # =========================
        if st.button("📥 Télécharger le modèle global"):
            try:
                response = requests.get(f"{SERVER_URL}/get_model")
                response.raise_for_status()

                buffer = io.BytesIO(response.content)
                global_state = torch.load(buffer, map_location=device)

                model.load_state_dict(global_state)
                st.session_state["model_state"] = global_state
                st.session_state["model_loaded"] = True

                st.success("Modèle global téléchargé avec succès !")

            except Exception as e:
                st.error(f"Erreur téléchargement : {e}")

        # =========================
        # TRAIN LOCAL
        # =========================
        if st.button("🧠 Entraîner le modèle localement"):

            if not st.session_state["model_loaded"]:
                st.warning("⚠️ Télécharge d'abord le modèle global !")
            else:
                loss = train_fn(model, dataloader, epochs, lr, device)
                test_loss, test_acc = test_fn(model, dataloader, device)

                st.success(f"Entraînement terminé ! Loss: {loss:.4f}")
                st.info(f"Accuracy: {test_acc:.4f}")

                # 💾 stock metrics pour dashboard
                st.session_state["metrics"] = {
                    "loss": loss,
                    "test_loss": test_loss,
                    "accuracy": test_acc,
                    "dataset_size": len(df)
                }

                st.session_state["model_state"] = model.state_dict()
                st.session_state["trained"] = True

                st.write("Poids prêts pour envoi.")

        # =========================
        # SEND WEIGHTS
        # =========================
        if st.button("📤 Envoyer les poids au serveur"):

            if not st.session_state.get("trained", False):
                st.error("⚠️ Tu dois d'abord entraîner le modèle !")
            else:
                try:
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

                    try:
                        data = response.json()
                        if response.status_code == 200:
                            st.success(data.get("message", "Poids envoyés avec succès !"))
                        else:
                            st.error(data.get("error", response.text))
                    except:
                        st.success(response.text)

                except Exception as e:
                    st.error(f"Erreur envoi : {e}")


# =========================
# 📊 DASHBOARD
# =========================
elif menu == "📊 Dashboard":

    st.subheader("📊 Dashboard Clinicien / Recherche")

    metrics = st.session_state.get("metrics", {})

    if not metrics:
        st.warning("Aucune métrique disponible. Entraîne un modèle d'abord.")
    else:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        st.metric("Loss", f"{metrics.get('loss', 0):.4f}")
        st.metric("Dataset size", metrics.get("dataset_size", 0))

        st.success("Résumé de l'entraînement disponible.")

# =========================
# 🔬 EXPORT & RECHERCHE
# =========================
elif menu == "🔬 Export & Recherche":

    st.subheader("🔬 Export anonymisé (WP4)")

    if st.session_state.get("metrics"):
        df_export = pd.DataFrame([st.session_state["metrics"]])

        csv = df_export.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Télécharger export anonymisé",
            csv,
            "fl_metrics_export.csv",
            "text/csv"
        )

    else:
        st.info("Aucune donnée à exporter.")
