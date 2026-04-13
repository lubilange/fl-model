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
st.title("🧠 WP4 Clinical AI + Federated Learning Dashboard")

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
for k in ["model_loaded", "trained", "metrics", "history"]:
    if k not in st.session_state:
        st.session_state[k] = False if k in ["model_loaded", "trained"] else {}

if "history" not in st.session_state:
    st.session_state["history"] = []

# =========================
# MENU (RESTORED + CLEAN)
# =========================
menu = st.sidebar.selectbox(
    "📌 Navigation",
    [
        "🏠 Entraînement FL",
        "📊 Dashboard Clinique",
        "📈 Dashboard Recherche",
        "🔬 Export Anonymisé"
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
# DATA SOURCES (COHERENT WITH YOUR BACKEND)
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
def create_loader(df, batch=16):
    if df.empty:
        return None
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)

def train_fn(model, loader, epochs, lr, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    last = 0

    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
            last = float(loss)

    return last

def test_fn(model, loader, device):
    model.eval()
    correct = total = loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss += float(loss_fn(out, y))
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    return loss / max(1, len(loader)), correct / max(1, total)

# =========================================================
# 🏠 TRAINING
# =========================================================
if menu == "🏠 Entraînement FL":

    file = st.file_uploader("Dataset CSV", type="csv")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        batch = st.number_input("Batch", 8, 128, 16)
        epochs = st.number_input("Epochs", 1, 10, 5)
        lr = st.number_input("LR", 0.0001, 0.01, 0.001)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net().to(device)

        loader = create_loader(df, batch)

        if st.button("📥 Load Global Model"):
            r = requests.get(f"{SERVER_URL}/get_model")
            model.load_state_dict(torch.load(io.BytesIO(r.content), map_location=device))
            st.session_state["model_loaded"] = True

        if st.button("🧠 Train Local"):
            if not st.session_state["model_loaded"]:
                st.warning("Load global model first")
            else:
                loss = train_fn(model, loader, epochs, lr, device)
                test_loss, acc = test_fn(model, loader, device)

                st.session_state["metrics"] = {
                    "loss": loss,
                    "accuracy": acc,
                    "size": len(df)
                }

                st.session_state["history"].append(acc)
                st.session_state["model_state"] = model.state_dict()
                st.session_state["trained"] = True

                st.success(f"Loss {loss:.3f} | Acc {acc:.3f}")

        if st.button("📤 Send Weights"):
            if st.session_state["trained"]:
                buffer = io.BytesIO()
                torch.save(st.session_state["model_state"], buffer)
                buffer.seek(0)

                requests.post(
                    f"{SERVER_URL}/submit_weights",
                    files={"weights": ("w.pt", buffer)},
                    headers={"Authorization": "Bearer SHARED_TOKEN"}
                )

                st.success("Weights sent ✔")

# =========================================================
# 📊 CLINICAL DASHBOARD (REAL + AI + ANALYTICS)
# =========================================================
elif menu == "📊 Dashboard Clinique":

    st.subheader("🏥 Vue clinique en temps réel")

    # ================= KPI =================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Patients", len(patients))
    col2.metric("🧾 Infos santé(Conditions FHIR)", len(conditions))
    col3.metric("🩺 Nombres Symptômes(Observations FHIR)", len(observations))

    adherence_rate = 0
    if len(adherence_logs) > 0:
        adherence_rate = len(adherence_logs[adherence_logs["status"] == "taken"]) / len(adherence_logs)

    col4.metric("💊 Suivi traitement", f"{adherence_rate:.0%}")

    st.divider()

    # ================= PATIENTS =================
    st.markdown("### 👥 Liste des patients")

    if not patients.empty:
        patients_view = patients.copy()

        # remplace id par téléphone (plus humain)
        if "phone" in patients_view.columns:
            patients_view = patients_view.rename(columns={"phone": "📱 Téléphone"})

        # cacher ID si présent
        if "patient_id" in patients_view.columns:
            patients_view = patients_view.drop(columns=["patient_id"])

        st.write("Données disponibles (export possible ci-dessous)")

        st.download_button(
            "⬇️ Télécharger les patients",
            patients_view.to_csv(index=False).encode("utf-8"),
            "patients.csv",
            "text/csv"
        )

    st.divider()

    # ================= TRIAGE =================
    st.markdown("### 🚨 Niveau d'alerte des patients")

    if not conditions.empty:
        st.bar_chart(conditions["severity"].value_counts())

        st.download_button(
            "⬇️ Télécharger les alertes",
            conditions.to_csv(index=False).encode("utf-8"),
            "alertes.csv",
            "text/csv"
        )

    st.divider()

    # ================= SYMPTÔMES =================
  st.markdown("### 🩺 Suivi des symptômes")

  if not observations.empty:
    
        obs_view = observations.copy()
    
        # ================= GARDER UNIQUEMENT SEVERITY =================
        if "severity" in obs_view.columns:
            obs_view = obs_view[["severity"]]
        else:
            obs_view = pd.DataFrame({"severity": []})
    
        # ================= NETTOYAGE =================
        obs_view["severity"] = obs_view["severity"].fillna("unknown")
    
        # ================= DOWNLOAD =================
        st.download_button(
            "⬇️ Télécharger les symptômes",
            obs_view.to_csv(index=False).encode("utf-8"),
            "symptomes.csv",
            "text/csv"
        )

    st.divider()

    # ================= ADHÉRENCE =================
    st.markdown("### 💊 Suivi des traitements")

    if not adherence_logs.empty:

        st.bar_chart(adherence_logs["status"].value_counts())

        no_show_rate = len(adherence_logs[adherence_logs["status"] == "no_response"]) / len(adherence_logs)
        st.metric("📉 Non-réponse", f"{no_show_rate:.0%}")

        st.download_button(
            "⬇️ Télécharger suivi traitement",
            adherence_logs.to_csv(index=False).encode("utf-8"),
            "adherence.csv",
            "text/csv"
        )

    st.divider()

    # ================= TENDANCE =================
    st.markdown("### 📈 Fréquence des symptômes")

    if not observations.empty:
        trend = observations.groupby("patient_id").size().reset_index(name="nb_symptomes")

        # ajouter téléphone
        if "phone" in patients.columns:
            trend = trend.merge(patients[["patient_id", "phone"]], on="patient_id", how="left")
            trend = trend.drop(columns=["patient_id"])
            trend = trend.rename(columns={"phone": "📱 Téléphone"})

        st.bar_chart(trend.set_index("📱 Téléphone"))

    st.divider()

    # ================= WORKLOAD =================
    st.markdown("### 👩‍⚕️ Charge des soignants")

    if not nurses.empty:
        st.bar_chart(nurses["status"].value_counts())

    st.divider()

    # ================= FL =================
    st.markdown("### 🧠 Performance IA (apprentissage fédéré)")

    if st.session_state.get("history"):
        st.line_chart(pd.DataFrame({"précision": st.session_state["history"]}))

    st.divider()

    # ================= SIMULATION =================
    st.markdown("### 🧪 Tests de validation")

    sim = pd.DataFrame([
        {"glycémie": 4.5, "niveau": "normal"},
        {"glycémie": 8.2, "niveau": "élevé"},
        {"glycémie": 6.8, "niveau": "modéré"}
    ])

    sim["prédiction"] = sim["glycémie"].apply(lambda x: "élevé" if x > 7 else "normal")

    st.dataframe(sim)

# =========================================================
# 📈 RESEARCH DASHBOARD (BI)
# =========================================================
elif menu == "📈 Dashboard Recherche":

    st.subheader("📊 BI & Research Analytics (REAL DATA)")

    # =========================
    # 1. COHORTE PATIENTS
    # =========================
    st.markdown("### 👥 Cohorte patients")

    if not patients.empty:
        gender_dist = patients["gender"].value_counts()

        st.bar_chart(gender_dist)

        st.write("Distribution patients par genre")
    else:
        st.info("Aucun patient")

    # =========================
    # 2. RISQUE CLINIQUE (conditions backend)
    # =========================
    st.markdown("### 🚨 Répartition des risques")

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

    # =========================
    # 3. ADHÉRENCE RECHERCHE
    # =========================
    st.markdown("### 💊 Adhérence traitement (study cohort)")

    if not adherence_logs.empty:
        adherence_dist = adherence_logs["status"].value_counts()

        st.bar_chart(adherence_dist)

        adherence_rate = len(adherence_logs[adherence_logs["status"] == "taken"]) / len(adherence_logs)
        st.metric("Adhérence globale", f"{adherence_rate:.2f}")
    else:
        st.info("Aucun log")

    # =========================
    # 4. PROGRESSION MALADIE (SYMPTÔMES)
    # =========================
    st.markdown("### 📈 Progression symptômes (research proxy)")

    if not observations.empty:

        progression = observations.groupby("patient_id").size().reset_index(name="symptom_count")

        st.bar_chart(progression.set_index("patient_id"))

        st.write("Nombre de symptômes par patient (proxy progression)")
    else:
        st.info("Aucune observation")

    # =========================
    # 5. NO-SHOW ANALYSIS
    # =========================
    st.markdown("### ⛔ No-show analysis")

    if not adherence_logs.empty:
        no_show_rate = len(adherence_logs[adherence_logs["status"] == "no_response"]) / len(adherence_logs)

        st.metric("No-show rate", f"{no_show_rate:.2f}")
    else:
        st.info("Aucun data no-show")
# =========================================================
# 🔬 EXPORT ANONYMIZED
# =========================================================
elif menu == "🔬 Export Anonymisé":

    st.subheader("🔬 Export research-ready")

    metrics = st.session_state.get("metrics", {})

    if metrics:
        export = pd.DataFrame([{
            "id": f"WP4_{random.randint(1000,9999)}",
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "dataset_size": metrics["size"]
        }])

        st.dataframe(export)

        st.download_button(
            "⬇ Export CSV",
            export.to_csv(index=False).encode(),
            "wp4.csv",
            "text/csv"
        )
