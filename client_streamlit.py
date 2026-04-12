import streamlit as st
import pandas as pd
import torch
import requests
import io
import random
import plotly.graph_objects as go

from supabase import create_client, Client
from authexample.task import Net
from torch.utils.data import DataLoader, TensorDataset

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="WP4 FL Dashboard", layout="wide")
st.title("🧠 WP4 Clinical AI Dashboard (Supabase + FHIR + FL)")

SERVER_URL = "https://fl-model.onrender.com"

# =========================
# SUPABASE
# =========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# MODE (IMPORTANT WP4)
# =========================
mode = st.sidebar.radio(
    "Mode",
    ["👨‍⚕️ Clinicien", "🔬 Recherche", "🧪 Simulation"]
)

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state["history"] = []

# =========================
# SAFE FETCH
# =========================
def safe_fetch(table):
    try:
        return supabase.table(table).select("*").execute().data or []
    except:
        return []

# =========================
# DATA
# =========================
patients = pd.DataFrame(safe_fetch("patients"))
conditions = pd.DataFrame(safe_fetch("conditions"))
observations = pd.DataFrame(safe_fetch("observations"))
treatments = pd.DataFrame(safe_fetch("treatments"))
adherence_logs = pd.DataFrame(safe_fetch("adherence_logs"))
nurses = pd.DataFrame(safe_fetch("nurses"))

# =========================
# KPIs
# =========================
st.subheader("📊 KPIs globaux")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", len(patients))
c2.metric("Conditions", len(conditions))
c3.metric("Observations", len(observations))
c4.metric("Traitements", len(treatments))

st.divider()

# =========================
# CLINICIEN VIEW
# =========================
if mode == "👨‍⚕️ Clinicien":

    st.markdown("## 🏥 Vue Clinique")

    st.markdown("### 👥 Patients")
    if not patients.empty:
        st.dataframe(patients[["phone", "name", "gender", "birth_date", "onboarded"]])

    st.markdown("### 🚨 Triage (Backend ONLY)")
    if not conditions.empty:
        st.dataframe(conditions[["patient_id", "label", "severity", "status", "created_at"]])
        st.bar_chart(conditions["severity"].value_counts())

    st.markdown("### 🧪 Observations")
    if not observations.empty:
        st.dataframe(observations[["patient_id", "text", "severity", "created_at"]])

    st.markdown("### 💊 Traitements")
    if not treatments.empty:
        st.dataframe(treatments[["patient_id", "text", "status", "created_at"]])

    st.markdown("### 💊 Adhérence (backend logs)")
    if not adherence_logs.empty:
        taken = len(adherence_logs[adherence_logs["status"] == "taken"])
        rate = taken / len(adherence_logs)

        st.metric("Adhérence globale", f"{rate:.2f}")
        st.bar_chart(adherence_logs["status"].value_counts())

    st.markdown("### 👩‍⚕️ Nurses")
    if not nurses.empty:
        st.bar_chart(nurses["status"].value_counts())

# =========================
# RESEARCH VIEW
# =========================
if mode == "🔬 Recherche":

    st.markdown("## 🔬 Export anonymisé")

    def anonymize(df):
        df = df.copy()
        if "phone" in df.columns:
            df["phone"] = df["phone"].apply(lambda x: "P_" + str(hash(str(x)))[:6])
        if "name" in df.columns:
            df["name"] = "ANONYMIZED"
        if "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].astype(str).apply(lambda x: "PID_" + x)
        return df

    export = {
        "patients": anonymize(patients),
        "conditions": anonymize(conditions),
        "observations": anonymize(observations),
        "treatments": anonymize(treatments)
    }

    for k, v in export.items():
        st.markdown(f"### {k}")
        st.dataframe(v)

    if not patients.empty:
        csv = anonymize(patients).to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Export anonymisé patients",
            csv,
            "wp4_export.csv"
        )

# =========================
# SIMULATION (VALIDATION SYSTEM)
# =========================
if mode == "🧪 Simulation":

    st.markdown("## 🧪 Cas cliniques simulés")

    sim = pd.DataFrame([
        {"input": "chest_pain + dyspnea", "expected": "EMERGENCY"},
        {"input": "fever + vomiting", "expected": "SEVERE"},
        {"input": "loss_of_consciousness", "expected": "CRITICAL"},
        {"input": "headache", "expected": "MILD"}
    ])

    st.dataframe(sim)

    st.info("👉 Validation backend riskEngine (WhatsApp bot)")

# =========================
# ANALYTICS (PRÉDICTIF WP4)
# =========================

st.divider()
st.markdown("## 📈 Analytics avancées")

# ---- symptom trend
if not observations.empty:
    observations["date"] = pd.to_datetime(observations["created_at"])
    trend = observations.groupby("date").size()
    st.markdown("### 📈 Symptômes progression")
    st.line_chart(trend)

# ---- adherence trend
if not adherence_logs.empty:
    adherence_logs["date"] = pd.to_datetime(adherence_logs["created_at"])
    trend2 = adherence_logs.groupby("date")["status"].apply(
        lambda x: (x == "taken").mean()
    )
    st.markdown("### 💊 Adhérence trend")
    st.line_chart(trend2)

# ---- no-show prediction simple
def no_show_risk(row):
    risk = 0.1
    try:
        if row.get("gender") == "male":
            risk += 0.1
        if row.get("birth_date"):
            age = 2026 - int(row["birth_date"][:4])
            if age > 60:
                risk += 0.2
    except:
        pass
    return min(risk, 1.0)

if not patients.empty:
    patients["no_show_risk"] = patients.apply(no_show_risk, axis=1)
    st.markdown("### 🔮 No-show prediction")
    st.dataframe(patients[["phone", "name", "no_show_risk"]])
    st.bar_chart(patients["no_show_risk"])

# =========================
# REAL-TIME SIMULATION
# =========================
st.divider()
if st.button("🔄 Refresh data"):
    st.rerun()
