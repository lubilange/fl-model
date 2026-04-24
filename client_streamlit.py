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
st.set_page_config(page_title="Dashboard", layout="wide")

# =========================
# 🎨 STYLE CSS (sidebar + cartes)
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        background-color: #eef1f5;
    }

    section[data-testid="stSidebar"] {
        background-color: #1e3c5a;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }

    h1, h2, h3 {
        color: #1e3c5a;
    }
</style>
""", unsafe_allow_html=True)



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
menu = st.sidebar.radio(
    " Menu",
    [
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

# =========================================================
# 📊 DASHBOARD CLINIQUE
# =========================================================
if menu == "Dashboard Clinique":
    st.subheader("🏥 Vue clinique en temps réel")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="card"><h3>👥 Patients</h3><h2>{len(patients)}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><h3>🧾 Conditions FHIR</h3><h2>{len(conditions)}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><h3>🩺 Symptômes</h3><h2>{len(observations)}</h2></div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🚨 Niveau d'alerte")
    if not conditions.empty:
        st.bar_chart(conditions["severity"].value_counts())

    st.divider()

    st.markdown("### 📈 Répartition des symptômes")
    if not observations.empty and "severity" in observations.columns:
        trend = observations["severity"].value_counts().reset_index()
        trend.columns = ["severity", "count"]
        st.bar_chart(trend.set_index("severity"))

    st.divider()

    st.markdown("### 👩‍⚕️ Support infirmier")
    if not nurses.empty:
        st.bar_chart(nurses["status"].value_counts())

    st.divider()

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
# 🔬 EXPORT ANONYMISÉ (NOUVEAU)
# =========================================================
elif menu == "Export Anonymisé":
    st.subheader("📤 Export anonymisé pour analyses")

    # Construction d'un DataFrame agrégé avec des statistiques non identifiantes
    data = []

    # Patients
    data.append({"Catégorie": "Patients", "Indicateur": "Nombre total", "Valeur": len(patients)})
    if not patients.empty and "gender" in patients.columns:
        for genre, nb in patients["gender"].value_counts().items():
            data.append({"Catégorie": "Patients", "Indicateur": f"Genre {genre}", "Valeur": nb})

    # Observations
    data.append({"Catégorie": "Observations", "Indicateur": "Nombre total", "Valeur": len(observations)})
    if not observations.empty and "severity" in observations.columns:
        for sev, nb in observations["severity"].value_counts().items():
            data.append({"Catégorie": "Observations", "Indicateur": f"Sévérité {sev}", "Valeur": nb})

    # Conditions
    data.append({"Catégorie": "Conditions", "Indicateur": "Nombre total", "Valeur": len(conditions)})
    if not conditions.empty and "severity" in conditions.columns:
        for sev, nb in conditions["severity"].value_counts().items():
            data.append({"Catégorie": "Conditions", "Indicateur": f"Sévérité {sev}", "Valeur": nb})

    # Infirmiers
    data.append({"Catégorie": "Infirmiers", "Indicateur": "Nombre total", "Valeur": len(nurses)})
    if not nurses.empty and "status" in nurses.columns:
        for stat, nb in nurses["status"].value_counts().items():
            data.append({"Catégorie": "Infirmiers", "Indicateur": f"Statut {stat}", "Valeur": nb})

    # Adhésion (adherence_logs)
    data.append({"Catégorie": "Adhésion", "Indicateur": "Nombre total de logs", "Valeur": len(adherence_logs)})
    if not adherence_logs.empty and "status" in adherence_logs.columns:
        for stat, nb in adherence_logs["status"].value_counts().items():
            data.append({"Catégorie": "Adhésion", "Indicateur": f"Statut {stat}", "Valeur": nb})

    # Traitements
    data.append({"Catégorie": "Traitements", "Indicateur": "Nombre total", "Valeur": len(treatments)})

    df_export = pd.DataFrame(data)

    # Affichage du tableau
    st.dataframe(df_export, use_container_width=True)

    # Téléchargement CSV
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Télécharger l'export anonymisé (CSV)",
        data=csv,
        file_name="export_anonymise.csv",
        mime="text/csv"
    )
