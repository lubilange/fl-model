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

# --- AJOUT STYLE IMAGE : CSS personnalisé pour reproduire le design ---
st.markdown("""
<style>
    /* Import de la police (similaire à l'image) */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Fond général plus clair */
    .main {
        background-color: #f7f8fc;
    }

    /* Cartes avec fond blanc, ombre, coins arrondis */
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 20px 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #2e2e4d;
    }
    .metric-label {
        font-size: 14px;
        color: #7d7d9e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-icon {
        font-size: 24px;
        margin-bottom: 5px;
    }
    /* Barre latérale */
    [data-testid="stSidebar"] {
        background-color: #1e1e2f;
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: #cfcfdf !important;
    }
    [data-testid="stSidebar"] .stSelectbox label, 
    [data-testid="stSidebar"] .stSelectbox div {
        color: white !important;
    }
    .sidebar-menu {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-around;
        margin-top: 30px;
    }
    .sidebar-menu a {
        color: #cfcfdf;
        text-decoration: none;
        font-size: 20px;
        padding: 10px;
        border-radius: 12px;
        transition: 0.2s;
    }
    .sidebar-menu a:hover {
        background-color: #2e2e4d;
    }
    /* Section profil */
    .profile-box {
        display: flex;
        align-items: center;
        gap: 15px;
        background: white;
        border-radius: 20px;
        padding: 15px 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .profile-avatar {
        width: 55px;
        height: 55px;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 24px;
    }
    .profile-name {
        font-weight: 600;
        font-size: 18px;
        color: #2e2e4d;
    }
    .profile-email {
        font-size: 13px;
        color: #7d7d9e;
    }
</style>
""", unsafe_allow_html=True)
# --- FIN AJOUT STYLE IMAGE ---

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
# MENU (RESTORED + CLEAN)
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

# --- AJOUT STYLE IMAGE : Menu rapide avec icônes dans la sidebar ---
st.sidebar.markdown("### Menu rapide")
cols_icons = st.sidebar.columns(6)
icon_list = ["🏠", "📁", "💬", "🔔", "📍", "📊"]
for i, icon in enumerate(icon_list):
    with cols_icons[i]:
        st.markdown(f"<div style='text-align:center; font-size:20px; padding:5px; cursor:pointer;'>{icon}</div>", unsafe_allow_html=True)
# --- FIN AJOUT STYLE IMAGE ---

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
# FL UTILS (inchangées)
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
# 🏠 TRAINING (ENTIÈREMENT CONSERVÉ)
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
# 📊 DASHBOARD CLINIQUE (STYLE IMAGE INTÉGRÉ)
# =========================================================
elif menu == "Dashboard Clinique":

    # --- AJOUT STYLE IMAGE : Section supérieure avec métriques et profil ---
    st.subheader("🏥 Vue clinique en temps réel")

    # 4 cartes de métriques (données fictives pour le style)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">💰</div>
            <div class="metric-value">$ 628</div>
            <div class="metric-label">Earning</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🔗</div>
            <div class="metric-value">2434</div>
            <div class="metric-label">Share</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">❤️</div>
            <div class="metric-value">1259</div>
            <div class="metric-label">Likes</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">⭐</div>
            <div class="metric-value">8,5</div>
            <div class="metric-label">Rating</div>
        </div>
        """, unsafe_allow_html=True)

    # Profil utilisateur
    st.markdown("""
    <div class="profile-box">
        <div class="profile-avatar">JD</div>
        <div>
            <div class="profile-name">JOHN DON</div>
            <div class="profile-email">johndon@company.com</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Graphique circulaire "Result 45%"
    fig_result = go.Figure(go.Pie(
        labels=["Result", ""],
        values=[45, 55],
        hole=0.7,
        marker_colors=["#6c5ce7", "#f0f0f5"],
        textinfo="none"
    ))
    fig_result.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        height=200,
        annotations=[dict(text="45%", x=0.5, y=0.5, font_size=28, showarrow=False, font_color="#2e2e4d")]
    )
    st.plotly_chart(fig_result, use_container_width=False, config={'displayModeBar': False})
    st.markdown("<div style='text-align:center; font-weight:600; color:#7d7d9e;'>Result</div>", unsafe_allow_html=True)

    # --- FIN AJOUT STYLE IMAGE ---

    # ================= CONSERVATION DES KPIs ORIGINAUX =================
    st.markdown("---")
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
# 📈 DASHBOARD RECHERCHE (INCHANGÉ)
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
# 🔬 EXPORT ANONYMIZED (INCHANGÉ)
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
